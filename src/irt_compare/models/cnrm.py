"""Censored Normal Response Model (Minamimoto, Wakai & Okada, 2026)."""

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from .partition import partition


def log_k(x: Array, theta: Array, alpha: Array, beta: Array, sigma: Array) -> Array:
	"""Log probability at the endpoints and log density inside."""
	mu = alpha * theta + beta
	part = partition(x)
	interior = norm.logpdf((x - mu) / sigma) - jnp.log(sigma)
	lower = norm.logcdf(-mu / sigma)
	upper = norm.logcdf((mu - 1.0) / sigma)

	return jnp.where(part.zero, lower, jnp.where(part.one, upper, interior))
