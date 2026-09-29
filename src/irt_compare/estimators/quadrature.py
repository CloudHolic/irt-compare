"""Adaptive Gauss-Hermite quadrature for integrals against the N(0, 1) ability prior."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats import norm


def hermite(n: int) -> tuple[Array, Array]:
	"""Gauss-Hermite nodes and log weights for the kernel exp(-x^2)."""
	x, w = np.polynomial.hermite.hermgauss(n)
	with np.errstate(divide="ignore"):
		log_w = np.log(w)

	return jnp.asarray(x), jnp.asarray(log_w)


def adaptive_nodes(mode: Array, scale: Array, x_h: Array, log_w_h: Array) -> tuple[Array, Array]:
	"""Per-person nodes theta_ph = mode_p + sqrt(2) scale_p x_h, shape (persons, nodes), and
	log weights such that sum_h w_ph g(theta_ph) approximates the integral of g(theta) phi(theta).
	"""
	theta = mode[:, None] + jnp.sqrt(2.0) * scale[:, None] * x_h[None, :]
	log_w = (
		log_w_h[None, :]
		+ x_h[None, :] ** 2
		+ jnp.log(jnp.sqrt(2.0) * scale)[:, None]
		+ norm.logpdf(theta)
	)

	return theta, log_w
