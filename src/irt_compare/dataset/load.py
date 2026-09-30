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


def split_persons(data: TrainData, fraction: float, seed: int) -> tuple[TrainData, TrainData]:
	"""(kept, held out): a random `fraction` of the persons and all their cells held out.

	Used to tune settings on train data alone, so the test split stays untouched until the
	final comparison. Persons are re-indexed from 0 within each part; items keep their indices.
	"""
	if not 0.0 < fraction < 1.0:
		raise ValueError(f"fraction must be in (0, 1), got {fraction}")
	rng = np.random.default_rng(seed)
	held = np.zeros(data.n_persons, dtype=bool)
	held[rng.choice(data.n_persons, size=round(fraction * data.n_persons), replace=False)] = True
	return _subset(data, ~held), _subset(data, held)


def _subset(data: TrainData, keep: np.ndarray) -> TrainData:
	new_index = np.cumsum(keep) - 1
	cell = keep[data.person]
	return TrainData(
		person=new_index[data.person[cell]],
		item=data.item[cell],
		x=data.x[cell],
		n_persons=int(keep.sum()),
		n_items=data.n_items,
		person_idx=data.person_idx[keep],
		dataset_hash=data.dataset_hash,
		dataset_path=data.dataset_path,
	)
