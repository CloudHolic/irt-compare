"""Held-out log marginal likelihood of persons no fit has seen, and paired comparisons.

For each test person, log int prod_i k(x_pi | theta, tau_i) phi(theta) dtheta with the item
parameters fixed at the run's point estimate. theta is integrated on a dense global grid, plus a
fine local grid around the person's grid argmax that replaces the global points it covers: some
posteriors (the simplex's, near the ceiling) are narrower than any affordable global spacing.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats import norm

from ..dataset import TrainData
from ..estimators.map_aghq import Cells, Model
from .fitted import Fitted

GRID = np.linspace(-12.0, 12.0, 6001)
LOCAL_HALF_WIDTH = 0.01
LOCAL_POINTS = 2001
CHUNK = 512


@dataclass(frozen=True)
class Posterior:
	"""Per-person log marginal likelihood and the theta posterior's mean and sd."""

	log_marginal: np.ndarray
	mean: np.ndarray
	sd: np.ndarray


def person_posterior(fitted: Fitted, data: TrainData) -> Posterior:
	"""Log marginal likelihood (global + local grid) and posterior moments (global grid)."""
	grid = jnp.asarray(GRID)
	log_w = norm.logpdf(grid) + np.log(GRID[1] - GRID[0])

	acc = _accumulate(
		fitted, data, lambda person, _: jnp.broadcast_to(grid, (person.size, grid.size))
	)
	joint = acc + log_w[None, :]
	m0 = grid[jnp.argmax(acc + norm.logpdf(grid)[None, :], axis=1)]

	offsets = jnp.linspace(-LOCAL_HALF_WIDTH, LOCAL_HALF_WIDTH, LOCAL_POINTS)
	m0_pad = jnp.concatenate([m0, jnp.zeros(1)])
	local_acc = _accumulate(
		fitted, data, lambda person, _: m0_pad[person][:, None] + offsets[None, :]
	)
	step = 2.0 * LOCAL_HALF_WIDTH / (LOCAL_POINTS - 1)
	trap = jnp.full(LOCAL_POINTS, step).at[jnp.array([0, -1])].set(step / 2.0)
	local_theta = m0[:, None] + offsets[None, :]
	local_joint = local_acc + norm.logpdf(local_theta) + jnp.log(trap)[None, :]

	covered = jnp.abs(grid[None, :] - m0[:, None]) < LOCAL_HALF_WIDTH
	combined = jnp.concatenate([jnp.where(covered, -jnp.inf, joint), local_joint], axis=1)
	log_marginal = jax.scipy.special.logsumexp(combined, axis=1)

	w = jnp.exp(joint - jax.scipy.special.logsumexp(joint, axis=1, keepdims=True))
	mean = (w * grid).sum(axis=1)
	sd = jnp.sqrt((w * (grid - mean[:, None]) ** 2).sum(axis=1))
	return Posterior(np.asarray(log_marginal), np.asarray(mean), np.asarray(sd))


def paired_difference(
	a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0
) -> tuple[float, float]:
	"""Sum over persons of b - a, and its standard error by resampling persons."""
	d = np.asarray(b) - np.asarray(a)
	idx = np.random.default_rng(seed).integers(0, d.size, size=(n_boot, d.size))
	return float(d.sum()), float(d[idx].sum(axis=1).std())


def _accumulate(fitted: Fitted, data: TrainData, theta_rows) -> Array:
	"""sum over each person's cells of log k at per-cell theta rows, shape (persons, K).

	Cells go through in fixed-size chunks; the last one is padded with cells of a dummy person
	(index n_persons) that are dropped at the end, so one compiled kernel serves every chunk.
	"""
	n = data.x.size
	pad = (-n) % CHUNK
	person = np.concatenate([data.person, np.full(pad, data.n_persons)])
	item = np.concatenate([data.item, np.zeros(pad, dtype=data.item.dtype)])
	x = np.concatenate([data.x, np.full(pad, 0.5)])

	total = None
	for start in range(0, n + pad, CHUNK):
		sl = slice(start, start + CHUNK)
		p = jnp.asarray(person[sl])
		cells = Cells(p, jnp.asarray(item[sl]), jnp.asarray(x[sl]), _interior(jnp.asarray(x[sl])))
		part = _chunk(fitted.model, fitted.params, cells, theta_rows(p, cells), data.n_persons)
		total = part if total is None else total + part
	assert total is not None
	return total[: data.n_persons]


def _interior(x: Array) -> Array:
	return jnp.where((x == 0.0) | (x == 1.0), 0.5, x)


@partial(jax.jit, static_argnames=("model", "n_persons"))
def _chunk(
	model: Model, params: dict[str, Array], cells: Cells, theta: Array, n_persons: int
) -> Array:
	lk = model.cell_log_k(params, cells, theta)
	return jax.ops.segment_sum(lk, cells.person, num_segments=n_persons + 1)
