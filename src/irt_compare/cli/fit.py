"""Command-line entry point: fits one family to one response variable and logs the run."""

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
from irt_compare.estimators.nuts import NUTSConfig, NUTSResult, fit_nuts, summarize
from irt_compare.models.zoi import COORDS
from irt_compare.models.zoi import FAMILIES as ZOI_FAMILIES


def main() -> None:
	"""Parses arguments, fits the configured family, and records everything in one MLflow run."""
	parser = argparse.ArgumentParser(description="Fit one IRT family and log it to MLflow.")
	parser.add_argument("config", type=Path, help="fit config YAML")
	parser.add_argument("--response", required=True, choices=RESPONSES)
	args = parser.parse_args()

	jax.config.update("jax_enable_x64", True)
	raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	data = load_train(Path(raw["dataset"]), args.response)
	resolved = {**raw, "response": args.response, "dataset_hash": data.dataset_hash}

	family = raw["family"]
	if family == "cnrm":
		run_id = _fit_cnrm(resolved, data)
	elif family in ZOI_FAMILIES:
		run_id = _fit_zoi(resolved, data)
	else:
		raise ValueError(f"unknown family {family!r}")

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


def _tags(resolved: dict[str, Any], estimator: str, spec_deviation: str) -> dict[str, str]:
	return {
		"family": resolved["family"],
		"response": resolved["response"],
		"estimator": estimator,
		"dataset_hash": resolved["dataset_hash"],
		"spec_deviation": spec_deviation,
	}


def _write_config(out: Path, resolved: dict[str, Any]) -> None:
	(out / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")


def _fit_cnrm(resolved: dict[str, Any], data: TrainData) -> str:
	if resolved["link"] != "identity":
		raise ValueError(f"only the linear link (identity) is implemented, got {resolved['link']}")
	em = EMConfig(**resolved["em"])
	run_name = f"cnrm-{resolved['response']}"
	with tracking.fit_run(
		resolved["experiment"], run_name, _tags(resolved, "em-mml", "aghq"), _flatten(resolved)
	) as run_id:
		start = time.perf_counter()
		result = fit_em(data.person, data.item, data.x, data.n_persons, data.n_items, em)
		runtime = time.perf_counter() - start

		args_ll = (data.person, data.item, data.x, data.n_persons)
		estimates = (result.alpha, result.beta, result.sigma)
		loglik = person_loglik(*args_ll, *estimates, em.n_nodes)
		loglik_ref = person_loglik(*args_ll, *estimates, resolved["diagnostic"]["ref_nodes"])

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
			lambda out: _write_cnrm(out, resolved, data, result, loglik, loglik_ref),
		)
	return run_id


def _write_cnrm(
	out: Path,
	resolved: dict[str, Any],
	data: TrainData,
	result: EMResult,
	loglik: np.ndarray,
	loglik_ref: np.ndarray,
) -> None:
	_write_config(out, resolved)

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


def _fit_zoi(resolved: dict[str, Any], data: TrainData) -> str:
	nuts = NUTSConfig(**resolved["nuts"])
	run_name = f"{resolved['family']}-{resolved['response']}"
	with tracking.fit_run(
		resolved["experiment"], run_name, _tags(resolved, "nuts", "none"), _flatten(resolved)
	) as run_id:
		start = time.perf_counter()
		result = fit_nuts(
			resolved["family"],
			data.person,
			data.item,
			data.x,
			data.n_persons,
			data.n_items,
			resolved["prior"]["scale"],
			nuts,
		)
		runtime = time.perf_counter() - start

		items = summarize(result.item_draws)
		persons = summarize(result.theta_draws)
		tracking.log_metrics(
			run_id,
			{
				"runtime_s": runtime,
				"max_rhat": float(max(items["rhat"].max(), persons["rhat"].max())),
				"min_ess_bulk": float(min(items["ess_bulk"].min(), persons["ess_bulk"].min())),
				"min_ess_tail": float(min(items["ess_tail"].min(), persons["ess_tail"].min())),
				"n_divergent": result.n_divergent,
				"tree_depth_hit_frac": result.tree_depth_hit_frac,
			},
			{},
		)
		tracking.log_artifacts(
			run_id,
			lambda out: _write_zoi(out, resolved, data, items, persons, result),
		)
	return run_id


def _write_zoi(
	out: Path,
	resolved: dict[str, Any],
	data: TrainData,
	items: dict[str, np.ndarray],
	persons: dict[str, np.ndarray],
	result: NUTSResult,
) -> None:
	_write_config(out, resolved)
	coords = COORDS[resolved["family"]]

	pl.DataFrame(
		{
			"item_idx": np.repeat(np.arange(data.n_items), len(coords)),
			"coord": np.tile(coords, data.n_items),
			**{name: value.reshape(-1) for name, value in items.items()},
		}
	).write_parquet(out / "items_summary.parquet")

	pl.DataFrame({"person_idx": data.person_idx, **persons}).write_parquet(
		out / "persons_summary.parquet"
	)

	thin = resolved["draws"]["thin"]
	item_draws = result.item_draws[:, ::thin]
	chain, draw, idx, coord = np.indices(item_draws.shape).reshape(4, -1)
	pl.DataFrame(
		{
			"chain": chain,
			"draw": draw * thin,
			"item_idx": idx,
			"coord": np.asarray(coords)[coord],
			"value": item_draws.reshape(-1),
		}
	).write_parquet(out / "draws_items.parquet")

	theta_draws = result.theta_draws[:, ::thin]
	chain, draw, p = np.indices(theta_draws.shape).reshape(3, -1)
	pl.DataFrame(
		{
			"chain": chain,
			"draw": draw * thin,
			"person_idx": data.person_idx[p],
			"value": theta_draws.reshape(-1),
		}
	).write_parquet(out / "draws_persons.parquet")


if __name__ == "__main__":
	main()
