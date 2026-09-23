"""Dataset build configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
	"""Everything that determines a dataset build, given the database contents."""

	name: str
	n_items: int
	min_item_responses: int
	n_persons: int | None
	test_fraction: float
	seed: int

	def __post_init__(self) -> None:
		if self.n_items < 1:
			raise ValueError(f"n_items must be positive, got {self.n_items}")
		if self.min_item_responses < 1:
			raise ValueError(f"min_item_responses must be positive, got {self.min_item_responses}")
		if self.n_persons is not None and self.n_persons < 1:
			raise ValueError(f"n_persons must be positive or null, got {self.n_persons}")
		if not 0.0 < self.test_fraction < 1.0:
			raise ValueError(f"test_fraction must be in (0, 1), got {self.test_fraction}")


def load_config(path: Path) -> DatasetConfig:
	"""Reads a dataset config from YAML."""
	with path.open(encoding="utf-8") as f:
		raw = yaml.safe_load(f)

	return DatasetConfig(**raw)
