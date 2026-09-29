"""Reads a built dataset for fitting."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .tables import RESPONSES

SPLITS = ("train", "test")


@dataclass(frozen=True)
class TrainData:
	"""The cells of one split and one response variable. `person` indexes the split's persons."""

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
	return load_split(path, response, "train")


def load_split(path: Path, response: str, split: str) -> TrainData:
	"""Loads one split; the test persons never enter a fit and are what held-out scores use."""
	if response not in RESPONSES:
		raise ValueError(f"response must be one of {RESPONSES}, got {response!r}")
	if split not in SPLITS:
		raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

	manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
	if manifest["hash"] != path.name:
		raise ValueError(f"manifest hash {manifest['hash']} does not match directory {path.name}")

	persons = (
		pl.read_parquet(path / "persons.parquet")
		.filter(pl.col("split") == split)
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
