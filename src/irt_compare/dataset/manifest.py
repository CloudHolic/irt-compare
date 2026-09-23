"""Dataset manifest and the content hash that names the dataset directory."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

import polars as pl

from .config import DatasetConfig
from .query import CELLS_COPY
from .sampling import STREAMS
from .tables import RESPONSES

RULES = {
	"source": "Database views v_response_acc and v_response_score; definitions under `views`.",
	"population": "All plays of users with at least one play from a random-pool dump.",
	"mods": "Plays whose mods lie within NF TD HD SD DT HT NC PF (bitmask 17261).",
	"item": "(beatmap_id, rate_group); DT covers DT and NC, HT is its own group, else NM.",
	"score": "score / (1e6 * 0.5 if NF * 0.5 if HT).",
	"dedup": "Per (user, item) each response keeps its own best play: highest accuracy for acc, "
	"highest normalized score for score. The two may come from different plays.",
	"persons": "Drawn uniformly from the random pool and split into train and test; drawn "
	"persons without a response on the picked items are then dropped.",
	"eligibility": "An item is eligible when it has at least min_item_responses train responses.",
	"boundary": "A response is an endpoint iff it equals 0.0 or 1.0 exactly; no tolerance.",
}

LIMITATIONS = [
	"Keeping one best play per cell selects on the outcome and breaks the iid assumption.",
	"Players choose which charts to play, so missingness is not at random.",
]


def make_manifest(
	config: DatasetConfig,
	view_definitions: dict[str, str],
	ingest_log: list[dict[str, Any]],
	eligible: dict[int, int],
	allocation: dict[int, int],
	pool_size: int,
	n_drawn: int,
	items: pl.DataFrame,
	persons: pl.DataFrame,
	responses: pl.DataFrame,
) -> dict[str, Any]:
	"""The hashed part of the manifest: everything but the hash and build time."""
	split = responses.join(persons.select("person_idx", "split"), on="person_idx")
	train = split.filter(pl.col("split") == "train")
	return {
		"config": asdict(config),
		"seed_streams": list(STREAMS),
		"rules": RULES,
		"query": CELLS_COPY,
		"views": view_definitions,
		"ingest_log": ingest_log,
		"keys": {
			str(k): {"eligible": eligible[k], "allocated": n} for k, n in sorted(allocation.items())
		},
		"counts": {
			"items": items.height,
			"persons": {
				"pool": pool_size,
				"drawn": n_drawn,
				**{s: persons.filter(pl.col("split") == s).height for s in ("train", "test")},
			},
			"responses": {s: split.filter(pl.col("split") == s).height for s in ("train", "test")},
		},
		"train_endpoint_fraction": {
			name: {
				"zero": (train[name] == 0.0).mean(),
				"one": (train[name] == 1.0).mean(),
			}
			for name in RESPONSES
		},
		"limitations": LIMITATIONS,
	}


def content_hash(tables: dict[str, pl.DataFrame], core: dict[str, Any]) -> str:
	"""sha256 over column values and the canonical manifest, independent of parquet encoding."""
	h = hashlib.sha256()
	for name in sorted(tables):
		table = tables[name]
		for col in sorted(table.columns):
			h.update(f"{name}.{col}\0".encode())
			series = table[col]
			if series.dtype == pl.String:
				h.update("\0".join(series.to_list()).encode())
			elif series.dtype.is_integer():
				h.update(series.cast(pl.Int64).to_numpy().astype("<i8").tobytes())
			elif series.dtype.is_float():
				h.update(series.cast(pl.Float64).to_numpy().astype("<f8").tobytes())
			else:
				raise TypeError(f"cannot hash {name}.{col} of type {series.dtype}")
	h.update(json.dumps(core, sort_keys=True, ensure_ascii=False).encode())
	return h.hexdigest()[:12]
