"""Zero and one inflated IRT models (Molenaar, Curi & Bazan, 2022): Beta, Simplex and S_B."""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.nn import log_sigmoid, sigmoid
from jax.scipy.special import gammaln, logit
from jax.scipy.stats import norm

from .partition import partition

FAMILIES = ("zoi-beta", "zoi-simplex", "zoi-sb")
COORDS = {
	"zoi-beta": ("log_a", "b", "omega", "gamma_0", "gamma_1"),
	"zoi-simplex": ("log_a", "b", "log_phi", "gamma_0", "gamma_1"),
	"zoi-sb": ("log_a", "b", "log_delta", "gamma_0", "gamma_1"),
}


class ItemParams(NamedTuple):
	"""Item perameters."""

	log_a: Array
	b: Array
	dispersion: Array
	gamma_0: Array
	gamma_1: Array


def log_k(family: str, x: Array, theta: Array, item: ItemParams) -> Array:
	"""Log probability at 0 and 1 and log density inside."""
	a_theta = jnp.exp(item.log_a) * theta
	u0 = item.gamma_0 - a_theta
	u1 = item.gamma_1 - a_theta
	part = partition(x)

	x_in = jnp.where(part.interior, x, 0.5)
	inside = (
		log_sigmoid(u1)
		+ log_sigmoid(-u0)
		+ jnp.log(-jnp.expm1(-(item.gamma_1 - item.gamma_0)))
		+ _INTERIOR[family](x_in, a_theta + item.b, item.dispersion)
	)

	return jnp.where(part.zero, log_sigmoid(u0), jnp.where(part.one, log_sigmoid(-u1), inside))


def log_prior(item: ItemParams, scale: float) -> Array:
	"""log h(tau): every coordinate N(0, scale^2), gamma_1 truncated to (gamma_0, inf)."""
	free = (
		norm.logpdf(item.log_a, scale=scale)
		+ norm.logpdf(item.b, scale=scale)
		+ norm.logpdf(item.dispersion, scale=scale)
		+ norm.logpdf(item.gamma_0, scale=scale)
	)

	truncated = norm.logpdf(item.gamma_1, scale=scale) - norm.logcdf(-item.gamma_0 / scale)
	truncated = jnp.where(item.gamma_1 > item.gamma_0, truncated, -jnp.inf)

	return (free + truncated).sum()


def _log_beta(x: Array, eta: Array, dispersion: Array) -> Array:
	a = jnp.exp((eta + dispersion) / 2.0)
	b = jnp.exp((dispersion - eta) / 2.0)
	log_norm = gammaln(a) + gammaln(b) - gammaln(a + b)

	return (a - 1.0) * jnp.log(x) + (b - 1.0) * jnp.log1p(-x) - log_norm


def _log_simplex(x: Array, eta: Array, dispersion: Array) -> Array:
	# Written with 1 - mu = sigmoid(-eta) and x - mu = (1 - mu) - (1 - x): the direct form rounds
	# mu to 1 once eta passes ~37, and the deviance and its gradient turn into 0/0.
	log_scale = (
		jnp.log(2.0)
		+ dispersion
		+ jnp.log(x)
		+ jnp.log1p(-x)
		+ 2.0 * log_sigmoid(eta)
		+ 2.0 * log_sigmoid(-eta)
	)
	diff = sigmoid(-eta) - (1.0 - x)

	return -0.5 * (
		jnp.log(2.0 * jnp.pi) + dispersion + 3.0 * jnp.log(x) + 3.0 * jnp.log1p(-x)
	) - diff**2 * jnp.exp(-log_scale)


def _log_sb(x: Array, eta: Array, dispersion: Array) -> Array:
	# 1 / (x (1 - x)) is the logit Jacobian: the density is on x, not on logit x.
	return (
		-0.5 * (jnp.log(2.0 * jnp.pi) + dispersion)
		- jnp.log(x)
		- jnp.log1p(-x)
		- (logit(x) - eta) ** 2 / (2.0 * jnp.exp(dispersion))
	)


_INTERIOR: dict[str, Callable[[Array, Array, Array], Array]] = {
	"zoi-beta": _log_beta,
	"zoi-simplex": _log_simplex,
	"zoi-sb": _log_sb,
}
