"""Endpoint partition shared by every family."""

from typing import NamedTuple

from jax import Array


class Partition(NamedTuple):
	"""Boolean masks over responses; exactly one is true for each response."""

	zero: Array
	interior: Array
	one: Array


def partition(x: Array) -> Partition:
	"""Exact equality, the boundary rule recorded in the dataset manifest."""
	zero = x == 0.0
	one = x == 1.0
	return Partition(zero, ~(zero | one), one)
