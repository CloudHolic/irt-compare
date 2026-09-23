"""Reads a built dataset for fitting."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .tables import RESPONSES


@dataclass(frozen=True)
class TrainData:
	"""Train cells of one response variable."""

	person: np.ndarray
	item: np.ndarray
	x: np.ndarray
	n_persons: int
	n_items: int
	person_idx: np.ndarray
	dataset_hash: str
	dataset_path: Path


def load_train(path: Path, response: str) -> TrainData:
	"""Loads the train split."""
	if response not in RESPONSES:
		raise ValueError(f"response must be one of {RESPONSES}, got {response!r}")

	manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
	if manifest["hash"] != path.name:
		raise ValueError(f"manifest hash {manifest['hash']} does not match directory {path.name}")

	persons = (
		pl.read_parquet(path / "persons.parquet")
		.filter(pl.col("split") == "train")
		.sort("person_idx")
		.with_row_index("p")
		.with_columns(pl.col("p").cast(pl.Int64))
	)
	cells = (
		pl.read_parquet(path / "responses.parquet")
		.join(persons.select("p", "person_idx"), on="person_idx")
		.sort("p", "item_idx")
	)

	return TrainData(
		person=cells["p"].to_numpy(),
		item=cells["item_idx"].to_numpy(),
		x=cells[response].to_numpy(),
		n_persons=persons.height,
		n_items=manifest["counts"]["items"],
		person_idx=persons["person_idx"].to_numpy(),
		dataset_hash=manifest["hash"],
		dataset_path=path,
	)
