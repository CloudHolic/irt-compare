"""The only module that talks to MLflow: fail-fast checkcs, run creation, batched logging."""

import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param

ARTIFACT_SCHEME = "mlflow-artifacts:"


@contextmanager
def fit_run(
	experiment: str, run_name: str, tags: Mapping[str, str], params: Mapping[str, object]
) -> Iterator[str]:
	"""Checks the server, opens a run and yields its id. An exception marks the run FAILED."""
	if not os.environ.get("MLFLOW_TRACKING_URI"):
		raise RuntimeError("MLFLOW_TRACKING_URI is not set; run with `uv run --env-file .env`")

	client = MlflowClient()
	experiment_id = _experiment_id(client, experiment)
	all_tags = {**tags, "git_dirty": str(_git_dirty()).lower()}

	with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=all_tags) as run:
		client.log_batch(run.info.run_id, params=[Param(k, str(v)) for k, v in params.items()])
		yield run.info.run_id


def log_metrics(
	run_id: str, scalars: Mapping[str, float], series: Mapping[str, Sequence[float]]
) -> None:
	"""Sends every metric in one batch."""
	now = int(time.time() * 1000)
	metrics = [Metric(k, float(v), now, 0) for k, v in scalars.items()]
	metrics += [
		Metric(k, float(v), now, step)
		for k, values in series.items()
		for step, v in enumerate(values, start=1)
	]

	MlflowClient().log_batch(run_id, metrics=metrics)


def log_artifacts(run_id: str, write: Callable[[Path], None]) -> None:
	"""Let `write` fill a temporary directory, then uploads it to the run."""
	with tempfile.TemporaryDirectory() as tmp:
		write(Path(tmp))
		MlflowClient().log_artifacts(run_id, tmp)


def _experiment_id(client: MlflowClient, name: str) -> str:
	# The lookup doubles as the connectivity check: it fails before any fitting starts.
	experiment = client.get_experiment_by_name(name)
	if experiment is None:
		experiment = client.get_experiment(client.create_experiment(name))

	if not experiment.artifact_location.startswith(ARTIFACT_SCHEME):
		raise RuntimeError(
			f"experiment {name!r} stores artifacts at {experiment.artifact_location!r}, "
			"not on the server; use a new experiment name"
		)

	return experiment.experiment_id


def _git_dirty() -> bool:
	status = subprocess.run(
		["git", "status", "--porcelain"], capture_output=True, text=True, check=True
	)
	return bool(status.stdout.strip())
