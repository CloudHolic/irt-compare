"""Why a NUTS run does not mix: how much the log-likelihood's curvature moves with position."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array

from ..dataset import TrainData
from ..estimators.map_aghq import Cells, Model
from .fitted import Fitted

OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def person_and_item_curvature(
	fitted: Fitted,
	data: TrainData,
	theta_mean: np.ndarray,
	theta_width: np.ndarray,
	b_width: np.ndarray,
) -> pl.DataFrame:
	"""Summary rows for theta (per person) and b (per item): level at the mean, spread, and how
	many coordinates are not concave somewhere in the window."""
	if fitted.model.family == "cnrm":
		raise ValueError("the curvature probe is written for the ZOI families")
	cells = Cells(
		jnp.asarray(data.person),
		jnp.asarray(data.item),
		jnp.asarray(data.x),
		jnp.where((data.x == 0.0) | (data.x == 1.0), 0.5, jnp.asarray(data.x)),
	)
	theta_mean_jax, theta_width_jax, b_width_jax = (
		jnp.asarray(v) for v in (theta_mean, theta_width, b_width)
	)

	h_theta = jnp.stack(
		[
			_theta_hessian(
				fitted.model,
				fitted.params,
				cells,
				theta_mean_jax + o * theta_width_jax,
				data.n_persons,
			)
			+ 1.0
			for o in OFFSETS
		]
	)
	h_b = jnp.stack(
		[
			_b_hessian(
				fitted.model, fitted.params, cells, theta_mean_jax, o * b_width_jax, data.n_items
			)
			for o in OFFSETS
		]
	)

	rows = []
	for name, h, width in (("theta", h_theta, theta_width_jax), ("b", h_b, b_width_jax)):
		h = np.asarray(h)
		level = h[OFFSETS.index(0.0)] * np.asarray(width) ** 2
		concave = h.min(axis=0) > 0.0
		spread = np.where(concave, h.max(axis=0) / np.where(concave, h.min(axis=0), 1.0), np.inf)
		finite = spread[np.isfinite(spread)]
		rows.append(
			{
				"coordinate": name,
				"level_median": float(np.median(level)),
				"level_max": float(level.max()),
				"spread_median": float(np.median(finite)),
				"spread_p99": float(np.quantile(finite, 0.99)),
				"spread_max": float(finite.max()),
				"not_concave": int((~concave).sum()),
			}
		)
	return pl.DataFrame(rows)


@partial(jax.jit, static_argnames=("model", "n_persons"))
def _theta_hessian(
	model: Model, params: dict[str, Array], cells: Cells, theta: Array, n_persons: int
) -> Array:
	def total(t: Array) -> Array:
		return model.cell_log_k(params, cells, t[cells.person]).sum()

	grad = jax.grad(total)
	return -jax.jvp(grad, (theta,), (jnp.ones_like(theta),))[1]


@partial(jax.jit, static_argnames=("model", "n_items"))
def _b_hessian(
	model: Model, params: dict[str, Array], cells: Cells, theta: Array, shift: Array, n_items: int
) -> Array:
	def total(b: Array) -> Array:
		return model.cell_log_k({**params, "b": b}, cells, theta[cells.person]).sum()

	b = params["b"] + shift
	grad = jax.grad(total)
	return -jax.jvp(grad, (b,), (jnp.ones_like(b),))[1]
