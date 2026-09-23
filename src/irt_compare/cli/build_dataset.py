"""Command-line entry point for building a dataset."""

import argparse
import os
from pathlib import Path

from irt_compare.dataset import build, load_config


def main() -> None:
	"""Builds the dataset and prints its directory."""
	parser = argparse.ArgumentParser(description="Build a dataset from the osu! database.")
	parser.add_argument("config", type=Path, help="dataset config YAML")
	parser.add_argument("--out", type=Path, default=Path("datasets"), help="datasets root")
	args = parser.parse_args()

	dsn = os.environ.get("DSN")
	if not dsn:
		parser.error("DSN is not set; run with `uv run --env-file .env`")

	print(build(load_config(args.config), dsn, args.out))


if __name__ == "__main__":
	main()
