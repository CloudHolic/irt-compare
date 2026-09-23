"""Builds a dataset directory from the database: persons, split, items, tables, manifest."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .config import DatasetConfig
from .manifest import content_hash, make_manifest
from .query import connect, fetch_cells, fetch_ingest_log, fetch_view_definitions
from .sampling import allocate, sample_items, spawn_rngs, split_persons, subsample_persons
from .tables import build_tables, count_train_responses


def build(config: DatasetConfig, dsn: str, out_root: Path) -> Path:
	"""Builds the dataset and returns its directory."""
	rngs = spawn_rngs(config.seed)
	with connect(dsn) as conn:
		cells = fetch_cells(conn)
		view_definitions = fetch_view_definitions(conn)
		ingest_log = fetch_ingest_log(conn)

	pool = np.sort(cells["user_id"].unique().to_numpy())
	drawn = subsample_persons(pool, config.n_persons, rngs["persons"])
	test = split_persons(len(drawn), config.test_fraction, rngs["split"])
	split = pl.DataFrame({"user_id": drawn, "split": np.where(test, "test", "train")})
	cells = cells.join(split, on="user_id")

	candidates = count_train_responses(cells)
	eligible_items = candidates.filter(pl.col("n_train") >= config.min_item_responses)
	eligible = dict(eligible_items.group_by("keys").len().iter_rows())
	allocation = allocate(eligible, config.n_items)
	picked = sample_items(eligible_items, allocation, rngs["items"])
	items, persons, responses = build_tables(picked, cells)

	core = make_manifest(
		config,
		view_definitions,
		ingest_log,
		eligible,
		allocation,
		len(pool),
		len(drawn),
		items,
		persons,
		responses,
	)
	tables = {"items": items, "persons": persons, "responses": responses}
	digest = content_hash(tables, core)

	target = out_root / config.name / digest
	if target.exists():
		return target
	staging = out_root / config.name / f".staging-{digest}-{os.getpid()}"
	staging.mkdir(parents=True)
	for name, table in tables.items():
		table.write_parquet(staging / f"{name}.parquet")
	manifest = {**core, "hash": digest, "built_at": datetime.now(UTC).isoformat()}
	with (staging / "manifest.json").open("w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2, ensure_ascii=False)
	staging.rename(target)
	return target
