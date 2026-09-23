"""Turns the drawn cells into the item, person and response tables."""

import polars as pl

RESPONSES = ("acc", "score")


def count_train_responses(cells: pl.DataFrame) -> pl.DataFrame:
	"""Train responses per (beatmap, rate_group)."""
	return (
		cells.filter(pl.col("split") == "train")
		.group_by("beatmap_id", "rate_group", "keys")
		.agg(pl.len().cast(pl.Int64).alias("n_train"))
		.sort("keys", "beatmap_id", "rate_group")
	)


def build_tables(
	picked: pl.DataFrame, cells: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
	"""Returns (items, persons, responses) indexed by contiguous item_idx and person_idx."""
	items = (
		picked.select("beatmap_id", "rate_group", "keys")
		.sort("keys", "beatmap_id", "rate_group")
		.with_row_index("item_idx")
		.with_columns(pl.col("item_idx").cast(pl.Int64))
	)
	kept = cells.join(
		items.select("item_idx", "beatmap_id", "rate_group"), on=["beatmap_id", "rate_group"]
	)
	persons = (
		kept.select("user_id", "split")
		.unique()
		.sort("user_id")
		.with_row_index("person_idx")
		.with_columns(pl.col("person_idx").cast(pl.Int64))
	)
	responses = kept.join(persons.select("person_idx", "user_id"), on="user_id").sort(
		"person_idx", "item_idx"
	)

	train_stats = (
		responses.filter(pl.col("split") == "train")
		.group_by("item_idx")
		.agg(
			pl.len().alias("n_train"),
			*(
				(pl.col(name) == value).sum().alias(f"{name}_{label}")
				for name in RESPONSES
				for value, label in ((0.0, "zero"), (1.0, "one"))
			),
		)
	)
	test_counts = (
		responses.filter(pl.col("split") == "test")
		.group_by("item_idx")
		.agg(pl.len().alias("n_test"))
	)
	count_cols = ["n_train", "n_test", "acc_zero", "acc_one", "score_zero", "score_one"]
	items = (
		items.join(train_stats, on="item_idx", how="left")
		.join(test_counts, on="item_idx", how="left")
		.with_columns(pl.col(count_cols).fill_null(0).cast(pl.Int64))
		.select("item_idx", "beatmap_id", "rate_group", "keys", *count_cols)
		.sort("item_idx")
	)

	return items, persons, responses.select("person_idx", "item_idx", "acc", "score")
