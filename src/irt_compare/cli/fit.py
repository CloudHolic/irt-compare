"""Command-line entry point: fits the CNRM by EM to one response variable and logs the run."""

import argparse
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import polars as pl
import yaml

from irt_compare import tracking
from irt_compare.dataset import RESPONSES, TrainData, load_train
from irt_compare.estimators.cnrm_em import EMConfig, EMResult, fit_em, person_loglik


def main() -> None:
	"""Parses argumetns, fits, and records everything in one MLflow run."""
	parser = argparse.ArgumentParser(description="Fit the CNRM by EM and log it to MLflow.")
	parser.add_argument("config", type=Path, help="fit config YAML")
	parser.add_argument("--response", required=True, choices=RESPONSES)
	args = parser.parse_args()

	jax.config.update("jax_enable_x64", True)
	raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	if raw["link"] != "identity":
		raise ValueError(f"only the linear link (identity) is implemented, got {raw['link']!r}")

	em = EMConfig(**raw["em"])
	data = load_train(Path(raw["dataset"]), args.response)
	resolved = {**raw, "response": args.response, "dataset_hash": data.dataset_hash}
	tags = {
		"family": "cnrm",
		"response": args.response,
		"estimator": "em-mml",
		"dataset_hash": data.dataset_hash,
		"spec_deviation": "none",
	}

	with tracking.fit_run(
		raw["experiment"], f"cnrm-{args.response}", tags, _flatten(resolved)
	) as run_id:
		start = time.perf_counter()
		result = fit_em(data.person, data.item, data.x, data.n_persons, data.n_items, em)
		runtime = time.perf_counter() - start

		args_ll = (data.person, data.item, data.x, data.n_persons)
		estimates = (result.alpha, result.beta, result.sigma)
		loglik = person_loglik(*args_ll, *estimates, em.n_nodes)
		loglik_ref = person_loglik(*args_ll, *estimates, raw["diagnostic"]["ref_nodes"])

		sigma_q = np.quantile(result.sigma, [0.05, 0.25, 0.5])

		tracking.log_metrics(
			run_id,
			{
				"runtime_s": runtime,
				"n_iter": result.n_iter,
				"converged": float(result.converged),
				"degenerate": float(result.degenerate),
				"n_degenerate_items": result.degenerate_items.size,
				"final_loglik": float(loglik.sum()),
				"sigma_min": float(result.sigma.min()),
				"sigma_q05": float(sigma_q[0]),
				"sigma_q25": float(sigma_q[1]),
				"sigma_q50": float(sigma_q[2]),
			},
			{"loglik": result.loglik.tolist()},
		)

		tracking.log_artifacts(
			run_id,
			lambda out: _write_artifacts(out, resolved, data, result, loglik, loglik_ref),
		)

	print(run_id)


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
	flat: dict[str, Any] = {}
	for key, value in tree.items():
		name = f"{prefix}{key}"
		if isinstance(value, dict):
			flat.update(_flatten(value, f"{name}."))
		else:
			flat[name] = value

	return flat


def _write_artifacts(
	out: Path,
	resolved: dict[str, Any],
	data: TrainData,
	result: EMResult,
	loglik: np.ndarray,
	loglik_ref: np.ndarray,
) -> None:
	(out / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

	pl.DataFrame(
		{
			"item_idx": np.arange(data.n_items),
			"alpha": result.alpha,
			"beta": result.beta,
			"sigma": result.sigma,
			"degenerate": np.isin(np.arange(data.n_items), result.degenerate_items),
		}
	).write_parquet(out / "items.parquet")

	pl.DataFrame(
		{"person_idx": data.person_idx, "eap": result.eap, "psd": result.psd}
	).write_parquet(out / "persons.parquet")

	pl.DataFrame(
		{
			"person_idx": data.person_idx,
			"n_responses": np.bincount(data.person, minlength=data.n_persons),
			"loglik": loglik,
			"loglik_ref": loglik_ref,
			"diff": loglik - loglik_ref,
		}
	).write_parquet(out / "quadrature.parquet")


if __name__ == "__main__":
	main()
