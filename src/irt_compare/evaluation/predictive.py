"""Posterior predictive checks near the ceiling."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array

from ..dataset import TrainData
from ..estimators.map_aghq import Cells, Model
from .fitted import Fitted
from .heldout import Posterior

T_GRID = np.linspace(-6.0, 16.0, 2200)  # logit(x): x from 0.0025 to 1 - 1e-7
N_THETA = 9
CHUNK = 256


def cell_predictive(
	fitted: Fitted, data: TrainData, posterior: Posterior, thresholds: tuple[float, ...]
) -> pl.DataFrame:
	"""One row per cell: observed x, P(x = 1), P(x <= t) per threshold, E/Var[u | in], PIT."""
	z, w = np.polynomial.hermite.hermgauss(N_THETA)
	nodes_z, nodes_w = jnp.asarray(np.sqrt(2.0) * z), jnp.asarray(w / np.sqrt(np.pi))
	thr = np.asarray(thresholds)
	t_thr = jnp.asarray(np.log(thr / (1.0 - thr)))

	n = data.x.size
	pad = (-n) % CHUNK
	item = np.concatenate([data.item, np.zeros(pad, dtype=data.item.dtype)])
	x = np.concatenate([data.x, np.full(pad, 0.5)])
	mean = np.concatenate([posterior.mean[data.person], np.zeros(pad)])
	sd = np.concatenate([posterior.sd[data.person], np.ones(pad)])

	parts = []
	for start in range(0, n + pad, CHUNK):
		sl = slice(start, start + CHUNK)
		theta = jnp.asarray(mean[sl])[:, None] + jnp.asarray(sd[sl])[:, None] * nodes_z[None, :]
		parts.append(
			_chunk(
				fitted.model,
				fitted.params,
				jnp.asarray(item[sl]),
				jnp.asarray(x[sl]),
				theta,
				nodes_w,
				t_thr,
			)
		)
	cols = [np.concatenate([np.asarray(p[i]) for p in parts])[:n] for i in range(5)]
	p_one, cdf, eu, vu, pit = cols

	out = {"x": data.x, "item": data.item, "person": data.person, "p_one": p_one}
	out |= {f"cdf_{t}": cdf[:, j] for j, t in enumerate(thresholds)}
	interior = (data.x > 0.0) & (data.x < 1.0)
	out |= {
		"e_u": np.where(interior, eu, np.nan),
		"var_u": np.where(interior, vu, np.nan),
		"pit": np.where(interior, pit, np.nan),
	}
	return pl.DataFrame(out)


@partial(jax.jit, static_argnames=("model",))
def _chunk(
	model: Model,
	params: dict[str, Array],
	item: Array,
	x: Array,
	theta: Array,
	w: Array,
	t_thr: Array,
) -> tuple[Array, Array, Array, Array, Array]:
	c = item.size
	dummy = jnp.zeros(c, dtype=jnp.int32)

	def at(value: Array | float) -> Array:
		xs = jnp.full(c, value)
		cells = Cells(dummy, item, xs, jnp.where((xs == 0.0) | (xs == 1.0), 0.5, xs))
		return model.cell_log_k(params, cells, theta)  # (c, nodes)

	p0, p1 = jnp.exp(at(0.0)), jnp.exp(at(1.0))
	t = jnp.asarray(T_GRID)
	xt = jax.nn.sigmoid(t)
	dt = t[1] - t[0]
	# density in t: f(x) dx/dt = f(x) x (1 - x); shape (T, c, nodes)
	dens = jnp.exp(jax.lax.map(at, xt)) * (xt * (1.0 - xt))[:, None, None]
	dens = jnp.nan_to_num(dens)
	p_in = jnp.clip(1.0 - p0 - p1, 0.0, 1.0)
	steps = (dens[1:] + dens[:-1]) / 2.0 * dt  # trapezoids between grid points
	below = jnp.clip(p_in - steps.sum(axis=0), 0.0, None)  # interior mass under the grid
	cum = below[None] + jnp.concatenate([jnp.zeros_like(dens[:1]), jnp.cumsum(steps, axis=0)])

	def cum_at(tq: Array) -> Array:
		"""P(0 < X <= logistic(tq)), linear between grid points; tq broadcasts over cum[0]."""
		pos = jnp.clip((tq - t[0]) / dt, 0.0, t.size - 1.0)
		j = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, t.size - 2)
		frac = (pos - j)[..., None]
		if tq.ndim == 0:
			return cum[j] + frac[..., 0] * (cum[j + 1] - cum[j])
		lo, hi = cum[j, jnp.arange(c)], cum[j + 1, jnp.arange(c)]
		return lo + frac * (hi - lo)

	p_one = (w * p1).sum(axis=1)
	cdf = jnp.stack([((p0 + cum_at(tq)) * w).sum(axis=1) for tq in t_thr], axis=1)

	mass = (w * p_in).sum(axis=1)
	u = jax.nn.sigmoid(-t)[:, None, None]
	eu = (dens * u * w).sum(axis=(0, 2)) * dt / mass
	eu2 = (dens * u**2 * w).sum(axis=(0, 2)) * dt / mass

	t_obs = jnp.log(x / (1.0 - x))
	pit = (cum_at(jnp.where(jnp.isfinite(t_obs), t_obs, t[0])) * w).sum(axis=1) / mass
	return p_one, cdf, eu, eu2 - eu**2, pit
