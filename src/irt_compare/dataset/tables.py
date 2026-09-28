"""Turns the drawn cells into the item, person and response tables."""

import polars as pl

RESPONSES = ("acc", "score")


def k_core(
	cells: pl.DataFrame, min_item: int, min_person: int, item_counted: pl.Expr
) -> pl.DataFrame:
	"""Drops items and persons below their response floors until neither changes."""
	item_key = ["beatmap_id", "rate_group"]
	while True:
		item_n = cells.filter(item_counted).group_by(item_key).len()
		kept = cells.join(
			item_n.filter(pl.col("len") >= min_item).select(item_key), on=item_key, how="semi"
		)

		person_n = kept.group_by("user_id").len()
		kept = kept.join(
			person_n.filter(pl.col("len") >= min_person).select("user_id"), on="user_id", how="semi"
		)

		if kept.height == cells.height:
			return kept

		cells = kept


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
