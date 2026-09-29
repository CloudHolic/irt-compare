"""A finished run read back from its artifacts as a point fit: a Model and its parameters."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml
from jax import Array

from ..estimators.map_aghq import Model, zoi_params
from ..models.transform import TransformSpec
from ..models.zoi import COORDS


@dataclass(frozen=True)
class Fitted:
	"""A named point fit."""

	name: str
	model: Model
	params: dict[str, Array]
	kind: str  # map | nuts | em


def parse_run(spec: str) -> tuple[str, Path, tuple[int, ...] | None]:
	"""`name=path` or `name=path#0,2,3` (NUTS posterior means over those chains only)."""
	name, _, rest = spec.partition("=")
	if not rest:
		raise ValueError(f"expected name=path[#chains], got {spec!r}")
	path, _, chains = rest.partition("#")
	return name, Path(path), tuple(int(c) for c in chains.split(",")) if chains else None


def load_run(name: str, path: Path, chains: tuple[int, ...] | None = None) -> Fitted:
	"""Reads one run directory, whatever estimator produced it."""
	config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
	family = config["family"]

	if config.get("estimator") == "map":
		items = pl.read_parquet(path / "items.parquet").sort("item_idx")
		if family == "cnrm":
			transform = json.loads((path / "transform.json").read_text(encoding="utf-8"))
			spec = TransformSpec(transform["kind"], tuple(transform["knots"]))
			raw = np.asarray(transform["raw"]).reshape(spec.init().shape)
			return Fitted(name, Model("cnrm", spec), _cnrm(items, raw), "map")
		return Fitted(name, Model(family), _zoi(family, items), "map")

	if family == "cnrm":
		items = pl.read_parquet(path / "items.parquet").sort("item_idx")
		spec = TransformSpec("linear")
		return Fitted(name, Model("cnrm", spec), _cnrm(items, spec.init()), "em")

	if chains is None:
		summary = pl.read_parquet(path / "items_summary.parquet")
		wide = summary.pivot(on="coord", index="item_idx", values="mean")
	else:
		draws = pl.read_parquet(path / "draws_items.parquet").filter(pl.col("chain").is_in(chains))
		means = draws.group_by("item_idx", "coord").agg(pl.col("value").mean())
		wide = means.pivot(on="coord", index="item_idx", values="value")
	return Fitted(name, Model(family), _zoi(family, wide.sort("item_idx")), "nuts")


def _zoi(family: str, wide: pl.DataFrame) -> dict[str, Array]:
	return zoi_params(*(wide[c].to_numpy() for c in COORDS[family]))


def _cnrm(items: pl.DataFrame, raw: np.ndarray | Array) -> dict[str, Array]:
	return {
		"alpha": jnp.asarray(items["alpha"].to_numpy()),
		"beta": jnp.asarray(items["beta"].to_numpy()),
		"log_sigma": jnp.asarray(np.log(items["sigma"].to_numpy())),
		"transform": jnp.asarray(raw),
	}
