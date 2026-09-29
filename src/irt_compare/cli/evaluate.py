"""Command-line entry point: compares finished runs on the same dataset and response."""

import argparse
from pathlib import Path

import jax
import numpy as np
import polars as pl

from irt_compare.dataset import RESPONSES, TrainData, load_split
from irt_compare.evaluation import shape
from irt_compare.evaluation.fitted import Fitted, load_run, parse_run
from irt_compare.evaluation.heldout import paired_difference, person_posterior
from irt_compare.evaluation.predictive import cell_predictive

THRESHOLDS = {
	"acc": (0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999),
	"score": (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999),
}
NEAR_CEILING = 0.003  # 1 - x below this counts as "just under the ceiling"


def main() -> None:
	"""Parses arguments, evaluates every run, writes the tables."""
	parser = argparse.ArgumentParser(description="Compare finished IRT runs.")
	parser.add_argument("--dataset", type=Path, required=True)
	parser.add_argument("--response", required=True, choices=RESPONSES)
	parser.add_argument("--run", action="append", required=True, help="name=path[#chains]")
	parser.add_argument("--reference", required=True, help="name of the run to compare against")
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--no-predictive", action="store_true", help="held-out only")
	args = parser.parse_args()

	jax.config.update("jax_enable_x64", True)
	runs = [load_run(*parse_run(spec)) for spec in args.run]
	names = [r.name for r in runs]
	if args.reference not in names:
		parser.error(f"--reference {args.reference!r} is not one of {names}")
	args.out.mkdir(parents=True, exist_ok=True)

	test = load_split(args.dataset, args.response, "test")
	keys = pl.read_parquet(args.dataset / "items.parquet").sort("item_idx")["keys"].to_numpy()
	tables = {"heldout": _heldout(runs, args.reference, test)}
	tables["heldout_breakdown"] = _breakdown(runs, args.reference, test, keys)

	if not args.no_predictive:
		train = load_split(args.dataset, args.response, "train")
		tables |= _predictive(runs, args.reference, train, THRESHOLDS[args.response])

	for name, table in tables.items():
		table.write_csv(args.out / f"{name}.csv")
	(args.out / "summary.md").write_text(_markdown(tables, args), encoding="utf-8")
	print((args.out / "summary.md").read_text(encoding="utf-8"))


def _heldout(runs: list[Fitted], reference: str, test: TrainData) -> pl.DataFrame:
	ll = {r.name: person_posterior(r, test).log_marginal for r in runs}
	rows = []
	for r in runs:
		diff, se = paired_difference(ll[reference], ll[r.name])
		rows.append(
			{
				"run": r.name,
				"estimator": r.kind,
				"heldout_loglik": float(ll[r.name].sum()),
				"per_cell": float(ll[r.name].sum() / test.x.size),
				f"diff_vs_{reference}": diff,
				"se": se,
			}
		)
	return pl.DataFrame(rows)


def _breakdown(
	runs: list[Fitted], reference: str, test: TrainData, keys: np.ndarray
) -> pl.DataFrame:
	"""Where the held-out difference comes from: key modes, and the cells just under 1."""
	u = 1.0 - test.x
	near = (u > 0.0) & (u < NEAR_CEILING)
	subsets = {"all": np.ones(test.x.size, dtype=bool), f"without 0<1-x<{NEAR_CEILING}": ~near}
	for k in np.unique(keys[test.item]):
		subsets[f"{k}K"] = keys[test.item] == k

	ref = next(r for r in runs if r.name == reference)
	rows = []
	for label, mask in subsets.items():
		sub = _subset(test, mask)
		base = person_posterior(ref, sub).log_marginal
		for r in runs:
			if r.name == reference:
				continue
			diff, se = paired_difference(base, person_posterior(r, sub).log_marginal)
			rows.append(
				{"cells": label, "n_cells": int(mask.sum()), "run": r.name, "diff": diff, "se": se}
			)
	return pl.DataFrame(rows)


def _predictive(
	runs: list[Fitted], reference: str, train: TrainData, thresholds: tuple[float, ...]
) -> dict[str, pl.DataFrame]:
	preds, posts = {}, {}
	for r in runs:
		posts[r.name] = person_posterior(r, train)
		preds[r.name] = cell_predictive(r, train, posts[r.name], thresholds)

	x = train.x
	observed = {"run": "observed", "x=1": float((x == 1.0).mean())}
	observed |= {f"x<={t}": float((x <= t).mean()) for t in thresholds}
	rows = [observed]
	for name, p in preds.items():
		row = {"run": name, "x=1": float(p["p_one"].mean())}
		row |= {f"x<={t}": float(p[f"cdf_{t}"].mean()) for t in thresholds}
		rows.append(row)

	pit_rows = []
	for name, p in preds.items():
		pit = p["pit"].drop_nans().to_numpy()
		hist = np.histogram(pit, bins=10, range=(0.0, 1.0))[0] / pit.size
		pit_rows.append({"run": name, **{f"d{i + 1}": float(h) for i, h in enumerate(hist)}})

	tagged = shape.groups(preds[reference].select("item", "person", "x"), posts[reference].mean)
	grouped = shape.compare(tagged, preds)
	law, by_distance = shape.summary(grouped, list(preds))
	return {
		"ppc": pl.DataFrame(rows),
		"pit": pl.DataFrame(pit_rows),
		"cv_law": law,
		"cv_by_distance": by_distance.with_columns(pl.col("distance").cast(pl.String)),
	}


def _subset(data: TrainData, mask: np.ndarray) -> TrainData:
	person_ids, person = np.unique(data.person[mask], return_inverse=True)
	return TrainData(
		person=person,
		item=data.item[mask],
		x=data.x[mask],
		n_persons=person_ids.size,
		n_items=data.n_items,
		person_idx=data.person_idx[person_ids],
		dataset_hash=data.dataset_hash,
		dataset_path=data.dataset_path,
	)


def _markdown(tables: dict[str, pl.DataFrame], args: argparse.Namespace) -> str:
	lines = [f"# {args.response}: {args.dataset.name}, reference {args.reference}", ""]
	for name, table in tables.items():
		lines += [f"## {name}", "", _md_table(table), ""]
	return "\n".join(lines)


def _md_table(df: pl.DataFrame) -> str:
	def fmt(v: object) -> str:
		if isinstance(v, float):
			return f"{v:.4g}" if abs(v) < 100 else f"{v:.1f}"
		return str(v)

	head = "| " + " | ".join(df.columns) + " |"
	rule = "|" + "|".join("---" for _ in df.columns) + "|"
	body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
	return "\n".join([head, rule, *body])


if __name__ == "__main__":
	main()
