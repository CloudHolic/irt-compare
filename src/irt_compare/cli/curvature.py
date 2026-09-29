"""Command-line entry point: the position-dependent curvature probe for NUTS runs."""

import argparse
from pathlib import Path

import jax
import polars as pl

from irt_compare.dataset import RESPONSES, load_train
from irt_compare.evaluation.curvature import person_and_item_curvature
from irt_compare.evaluation.fitted import load_run, parse_run


def main() -> None:
	"""Prints one table per run and writes them together to --out if given."""
	parser = argparse.ArgumentParser(description="Curvature probe for NUTS runs.")
	parser.add_argument("--dataset", type=Path, required=True)
	parser.add_argument("--response", required=True, choices=RESPONSES)
	parser.add_argument("--widths", type=Path, required=True)
	parser.add_argument("--run", action="append", required=True, help="name=path[#chains]")
	parser.add_argument("--out", type=Path)
	args = parser.parse_args()

	jax.config.update("jax_enable_x64", True)
	data = load_train(args.dataset, args.response)
	widths = pl.read_parquet(args.widths / "persons_summary.parquet").sort("person_idx")
	b_width = (
		pl.read_parquet(args.widths / "items_summary.parquet")
		.filter(pl.col("coord") == "b")
		.sort("item_idx")["sd"]
		.to_numpy()
	)

	tables = []
	for spec in args.run:
		name, path, chains = parse_run(spec)
		fitted = load_run(name, path, chains)
		means = pl.read_parquet(path / "persons_summary.parquet").sort("person_idx")
		table = person_and_item_curvature(
			fitted, data, means["mean"].to_numpy(), widths["sd"].to_numpy(), b_width
		).with_columns(pl.lit(name).alias("run"))
		print(name)
		print(table)
		tables.append(table)

	if args.out:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		pl.concat(tables).write_csv(args.out)


if __name__ == "__main__":
	main()
