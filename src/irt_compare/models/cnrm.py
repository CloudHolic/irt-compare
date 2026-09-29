"""Censored Normal Response Model (Minamimoto, Wakai & Okada, 2026)."""

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from .partition import partition


def log_k(
	x: Array,
	theta: Array,
	alpha: Array,
	beta: Array,
	sigma: Array,
	scale: tuple[Array, Array] | None = None,
) -> Array:
	"""Log probability at the endpoints and log density inside.

	`scale` is (G(x), log G'(x)) from a response-scale transform (models.transform), evaluated
	at the interior responses; None is the paper's linear v, G(x) = x.
	"""
	mu = alpha * theta + beta
	part = partition(x)
	g, log_dg = (x, 0.0) if scale is None else scale
	interior = norm.logpdf((g - mu) / sigma) - jnp.log(sigma) + log_dg
	lower = norm.logcdf(-mu / sigma)
	upper = norm.logcdf((mu - 1.0) / sigma)

	return jnp.where(part.zero, lower, jnp.where(part.one, upper, interior))
