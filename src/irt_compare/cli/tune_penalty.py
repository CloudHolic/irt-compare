"""Command-line entry point: picks the P-spline penalty on a person hold-out inside train."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml

from irt_compare.dataset import RESPONSES, TrainData, load_train, split_persons
from irt_compare.estimators import map_aghq
from irt_compare.estimators.cnrm_em import init_person_mean_ols
from irt_compare.evaluation.fitted import Fitted
from irt_compare.evaluation.heldout import paired_difference, person_posterior
from irt_compare.models.transform import TransformSpec


def main() -> None:
	"""Parses arguments, fits every penalty, writes the tables."""
	parser = argparse.ArgumentParser(
		description="Choose the P-spline penalty on train data; the test split is never read.",
		epilog="Writes tune.csv (held-out log marginal likelihood per penalty), exponent.csv "
		"(q at the knots) and summary.md to --out; the chosen penalty is the largest one within "
		"one paired-bootstrap SE of the best. Finished penalties are skipped on a rerun.",
	)
	parser.add_argument("config", type=Path, help="a CNRM MAP config with transform kind pspline")
	parser.add_argument("--response", required=True, choices=RESPONSES)
	parser.add_argument("--penalty", type=float, nargs="+", required=True)
	parser.add_argument("--holdout-fraction", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--out", type=Path, required=True)
	args = parser.parse_args()

	jax.config.update("jax_enable_x64", True)
	raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	if raw.get("family") != "cnrm" or raw.get("transform", {}).get("kind") != "pspline":
		parser.error("the config must be a CNRM fit with transform kind pspline")
	base = TransformSpec.from_config({**raw["transform"], "penalty": 0.0})
	config = map_aghq.MAPConfig(**raw["map"], prior_scale=raw["prior"]["scale"])

	train = load_train(Path(raw["dataset"]), args.response)
	kept, held = split_persons(train, args.holdout_fraction, args.seed)
	args.out.mkdir(parents=True, exist_ok=True)
	(args.out / "setup.json").write_text(
		json.dumps(
			{
				"config": str(args.config),
				"response": args.response,
				"holdout_fraction": args.holdout_fraction,
				"seed": args.seed,
				"persons_kept": kept.n_persons,
				"persons_held": held.n_persons,
				"cells_kept": int(kept.x.size),
				"cells_held": int(held.x.size),
			},
			indent=2,
		),
		encoding="utf-8",
	)

	rows_path, exponent_path = args.out / "tune.csv", args.out / "exponent.csv"
	done = set(pl.read_csv(rows_path)["penalty"].to_list()) if rows_path.exists() else set()
	for penalty in args.penalty:
		if penalty in done:
			print(f"penalty {penalty:g}: already done", flush=True)
			continue
		row, exponent, per_person = _fit_and_score(base, penalty, config, kept, held)
		np.save(args.out / f"heldout_persons_{penalty:g}.npy", per_person)
		_append(exponent_path, exponent)
		_append(rows_path, pl.DataFrame([row]))  # last: a row marks the penalty as done
		print(f"penalty {penalty:g}: held-out {row['heldout_loglik']:.1f}", flush=True)

	_summarize(args.out, args.penalty)


def _fit_and_score(
	base: TransformSpec,
	penalty: float,
	config: map_aghq.MAPConfig,
	kept: TrainData,
	held: TrainData,
) -> tuple[dict, pl.DataFrame, np.ndarray]:
	spec = dataclasses.replace(base, penalty=penalty)
	model = map_aghq.Model("cnrm", spec)
	alpha, beta, sigma = init_person_mean_ols(
		kept.person, kept.item, kept.x, kept.n_persons, kept.n_items
	)
	init = map_aghq.cnrm_params(alpha, beta, sigma, spec)

	start = time.perf_counter()
	result = map_aghq.fit_map(
		model, kept.person, kept.item, kept.x, kept.n_persons, init, config, log_every=500
	)
	runtime = time.perf_counter() - start

	params = {k: jnp.asarray(v) for k, v in result.params.items()}
	fitted = Fitted(f"pspline-{penalty:g}", model, params, "map")
	per_person = person_posterior(fitted, held).log_marginal

	knots = np.asarray(spec.knots)
	q = np.asarray(spec.exponent(params["transform"], jnp.asarray(knots)))
	roughness = float(-2.0 * spec.log_penalty(params["transform"]) / max(penalty, 1e-300))
	row = {
		"penalty": penalty,
		"heldout_loglik": float(per_person.sum()),
		"heldout_per_cell": float(per_person.sum() / held.x.size),
		"roughness": roughness,  # sum of squared second differences of log q
		"best_loss": result.best_loss,
		"best_step": result.best_step,
		"drop_last": result.drop_last,
		"runtime_s": runtime,
	}
	exponent = pl.DataFrame(
		{
			"penalty": np.full(knots.size, penalty),
			"knot_s": knots,
			"one_minus_x": np.exp(knots),
			"q": q,
		}
	)
	return row, exponent, per_person


def _append(path: Path, frame: pl.DataFrame) -> None:
	if path.exists():
		# diagonal: tune.csv also carries the summary's columns once a run has finished
		frame = pl.concat([pl.read_csv(path), frame], how="diagonal_relaxed")
	frame.write_csv(path)


def _summarize(out: Path, penalties: list[float]) -> None:
	rows = pl.read_csv(out / "tune.csv").filter(pl.col("penalty").is_in(penalties))
	rows = rows.sort("penalty")
	best = rows.sort("heldout_loglik", descending=True)["penalty"][0]
	ref = np.load(out / f"heldout_persons_{best:g}.npy")
	diffs, ses = [], []
	for penalty in rows["penalty"]:
		d, se = paired_difference(ref, np.load(out / f"heldout_persons_{penalty:g}.npy"))
		diffs.append(d)
		ses.append(se)
	rows = rows.with_columns(diff_vs_best=pl.Series(diffs), se=pl.Series(ses))
	rows.write_csv(out / "tune.csv")
	chosen = _one_se(rows, best)

	exponent = pl.read_csv(out / "exponent.csv").filter(pl.col("penalty").is_in(penalties))
	wide = exponent.pivot(on="penalty", index=["knot_s", "one_minus_x"], values="q").sort("knot_s")
	lines = [
		f"# P-spline penalty ({out.name})",
		"",
		f"Held-out maximum: {best:g}. Chosen (one-standard-error rule): **{chosen:g}**",
		"",
		_markdown(rows),
		"",
		"## local exponent q at the knots",
		"",
		_markdown(wide),
		"",
	]
	(out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
	print(f"held-out maximum: {best:g}, chosen penalty: {chosen:g}", flush=True)


def _one_se(rows: pl.DataFrame, best: float) -> float:
	"""The one-standard-error rule: from the best penalty, step to larger ones while the held-out
	stays within one paired SE of the best, and keep the last. Near the optimum the held-out
	curve is flat and its maximum alone drifts to the smallest penalty tried; stepping only
	through neighbours keeps one badly converged fit further out from being picked."""
	chosen = best
	for penalty, diff, se in (
		rows.filter(pl.col("penalty") > best).select("penalty", "diff_vs_best", "se").iter_rows()
	):
		if diff < -se:
			break
		chosen = penalty
	return chosen


def _markdown(frame: pl.DataFrame) -> str:
	def fmt(v: object) -> str:
		return f"{v:.4g}" if isinstance(v, float) else str(v)

	header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
	rule = "|" + "---|" * frame.width
	body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.iter_rows()]
	return "\n".join([header, rule, *body])


if __name__ == "__main__":
	main()
