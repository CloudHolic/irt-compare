"""How each fit spreads the distance to the ceiling, u = 1 - x, as its mean changes."""

import numpy as np
import polars as pl

QUINTILES = 5
MIN_GROUP_CELLS = 50  # per item, interior cells needed before it is split into quintiles


def groups(cells: pl.DataFrame, reference_theta: np.ndarray) -> pl.DataFrame:
	"""Interior cells tagged with (item, quintile of the reference theta within the item)."""
	theta = pl.DataFrame({"person": np.arange(reference_theta.size), "theta": reference_theta})
	df = (
		cells.select("item", "person", "x")
		.with_row_index("cell")
		.join(theta, on="person", how="left")
		.filter((pl.col("x") > 0.0) & (pl.col("x") < 1.0))
	)
	df = df.filter(pl.len().over("item") >= MIN_GROUP_CELLS)
	rank = pl.col("theta").rank("ordinal").over("item") - 1
	return df.with_columns(q=(rank * QUINTILES // pl.len().over("item")).cast(pl.Int64))


def compare(tagged: pl.DataFrame, predictive: dict[str, pl.DataFrame]) -> pl.DataFrame:
	"""Per group: observed mean/CV of u, and each fit's predicted mean/CV (total variance)."""
	frame = tagged.with_columns(u=1.0 - pl.col("x"))
	aggs = [
		pl.len().alias("n"),
		pl.col("u").mean().alias("obs_mean"),
		(pl.col("u").std() / pl.col("u").mean()).alias("obs_cv"),
	]
	for name, pred in predictive.items():
		frame = frame.with_columns(
			pl.Series(f"eu_{name}", pred["e_u"].to_numpy()[tagged["cell"].to_numpy()]),
			pl.Series(f"vu_{name}", pred["var_u"].to_numpy()[tagged["cell"].to_numpy()]),
		)
		mean = pl.col(f"eu_{name}").mean()
		# law of total variance over the group's cells
		var = pl.col(f"vu_{name}").mean() + pl.col(f"eu_{name}").var(ddof=0)
		aggs += [mean.alias(f"{name}_mean"), (var.sqrt() / mean).alias(f"{name}_cv")]
	return frame.group_by("item", "q").agg(aggs).sort("item", "q")


def within_item_slope(grouped: pl.DataFrame, mean_col: str, cv_col: str) -> float:
	"""Slope of log CV on log mean with item fixed effects."""
	d = grouped.select("item", lx=pl.col(mean_col).log(), ly=pl.col(cv_col).log()).with_columns(
		lx=pl.col("lx") - pl.col("lx").mean().over("item"),
		ly=pl.col("ly") - pl.col("ly").mean().over("item"),
	)
	lx = np.asarray(d["lx"].to_numpy(), dtype=float)
	ly = np.asarray(d["ly"].to_numpy(), dtype=float)
	return float(np.dot(lx, ly) / np.dot(lx, lx))


def summary(grouped: pl.DataFrame, names: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
	"""(per-fit slope and median log CV / mean ratios, median log CV ratio by distance quartile)."""
	rows = [{"source": "observed", "slope": within_item_slope(grouped, "obs_mean", "obs_cv")}]
	for name in names:
		ratio_cv = (grouped[f"{name}_cv"] / grouped["obs_cv"]).log()
		ratio_mean = (grouped[f"{name}_mean"] / grouped["obs_mean"]).log()
		rows.append(
			{
				"source": name,
				"slope": within_item_slope(grouped, f"{name}_mean", f"{name}_cv"),
				"median_log_cv_ratio": float(np.median(ratio_cv.to_numpy())),
				"median_abs_log_cv_ratio": float(np.median(ratio_cv.abs().to_numpy())),
				"median_log_mean_ratio": float(np.median(ratio_mean.to_numpy())),
			}
		)

	quartile = grouped["obs_mean"].qcut(4, labels=["closest", "2nd", "3rd", "farthest"])
	by = grouped.with_columns(distance=quartile)
	cols = [(pl.col(f"{n}_cv") / pl.col("obs_cv")).log().median().alias(n) for n in names]
	by_distance = (
		by.group_by("distance", maintain_order=False)
		.agg([*cols, pl.col("obs_mean").median(), pl.col("obs_cv").median()])
		.sort("obs_mean")
	)
	return pl.DataFrame(rows), by_distance
