"""Seeded draws: key-mode allocation, items within key modes, persons, and the person split."""

import numpy as np
import polars as pl

STREAMS = ("items", "persons", "split")


def spawn_rngs(seed: int) -> dict[str, np.random.Generator]:
	"""One generator per stage, so changing one stage leaves the other stages' draws intact."""
	children = np.random.SeedSequence(seed).spawn(len(STREAMS))
	return {
		name: np.random.default_rng(child) for name, child in zip(STREAMS, children, strict=True)
	}


def allocate(eligible: dict[int, int], n_items: int) -> dict[int, int]:
	"""Splits n_items across key modes in proportion to their eligible item counts."""
	keys = sorted(k for k, count in eligible.items() if count > 0)
	if len(keys) > n_items:
		raise ValueError(f"{len(keys)} key modes cannot each get an item out of {n_items}")

	total = sum(eligible[k] for k in keys)
	if total < n_items:
		raise ValueError(f"only {total} eligible items for {n_items} requested")

	quotas = {k: n_items * eligible[k] / total for k in keys}
	alloc = {k: int(quotas[k]) for k in keys}
	shortfall = n_items - sum(alloc.values())
	by_remainder = sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), k))

	for k in by_remainder[:shortfall]:
		alloc[k] += 1

	for k in keys:
		if alloc[k] == 0:
			donor = max(keys, key=lambda d: (alloc[d], -d))
			alloc[donor] -= 1
			alloc[k] = 1

	return alloc


def sample_items(
	eligible: pl.DataFrame, allocation: dict[int, int], rng: np.random.Generator
) -> pl.DataFrame:
	"""Draws allocation[k] items uniformly without replacement within each key mode."""
	picked = []

	for k in sorted(allocation):
		pool = eligible.filter(pl.col("keys") == k)
		rows = np.sort(rng.choice(pool.height, size=allocation[k], replace=False))
		picked.append(pool[rows])

	return pl.concat(picked).sort("keys", "beatmap_id", "rate_group")


def subsample_persons(
	user_ids: np.ndarray, n_persons: int | None, rng: np.random.Generator
) -> np.ndarray:
	"""Uniform subsample of sorted unique user ids; all of them when n_persons is None."""
	if n_persons is None:
		return user_ids
	if n_persons > len(user_ids):
		raise ValueError(f"n_persons={n_persons} exceeds the {len(user_ids)} available")

	return np.sort(rng.choice(user_ids, size=n_persons, replace=False))


def split_persons(n: int, test_fraction: float, rng: np.random.Generator) -> np.ndarray:
	"""Boolean mask over n persons, True for the test split."""
	test = np.zeros(n, dtype=bool)
	test[rng.permutation(n)[: round(n * test_fraction)]] = True
	return test
