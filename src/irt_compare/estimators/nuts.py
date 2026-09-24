"""NUTS over the joint posterior of persons and items for the ZOI models."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from arviz_stats.base import array_stats
from jax import Array
from jax.scipy.stats import norm
from numpyro.infer import MCMC, NUTS

from ..models.zoi import ItemParams, log_k, log_prior

CHAIN_METHODS = ("vectorized", "sequential")
STATS = ("mean", "sd", "q05", "q25", "q50", "q75", "q95", "ess_bulk", "ess_tail", "rhat")


@dataclass(frozen=True)
class NUTSConfig:
	"""Sampler settings."""

	chains: int
	warmup: int
	samples: int
	target_accept: float
	max_tree_depth: int
	seed: int
	chain_method: str

	def __post_init__(self) -> None:
		if self.chain_method not in CHAIN_METHODS:
			msg = f"chain_method must be one of {CHAIN_METHODS}, got {self.chain_method!r}"
			raise ValueError(msg)


@dataclass(frozen=True)
class NUTSResult:
	"""Draws in tau coordinates and the sampler's diagnostics."""

	item_draws: np.ndarray
	theta_draws: np.ndarray
	n_divergent: int
	tree_depth_hit_frac: float


def fit_nuts(
	family: str,
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	n_items: int,
	scale: float,
	config: NUTSConfig,
) -> NUTSResult:
	"""Samples theta and tau jointly."""
	key_init, key_run = jax.random.split(jax.random.PRNGKey(config.seed))
	k = jax.random.split(key_init, 6)
	items = (config.chains, n_items)
	init = {
		"theta": jax.random.normal(k[0], (config.chains, n_persons)),
		"log_a": 0.2 * jax.random.normal(k[1], items),
		"b": 0.5 * jax.random.normal(k[2], items),
		"dispersion": 0.2 * jax.random.normal(k[3], items),
		"gamma_0": -jnp.abs(1.0 + 0.2 * jax.random.normal(k[4], items)),
		"gamma_1": jnp.abs(1.0 + 0.2 * jax.random.normal(k[5], items)),
	}

	kernel = NUTS(
		potential_fn=partial(
			_potential,
			family=family,
			person=jnp.asarray(person),
			item=jnp.asarray(item),
			x=jnp.asarray(x),
			scale=scale,
		),
		target_accept_prob=config.target_accept,
		max_tree_depth=config.max_tree_depth,
	)

	mcmc = MCMC(
		kernel,
		num_warmup=config.warmup,
		num_samples=config.samples,
		num_chains=config.chains,
		chain_method=config.chain_method,
		progress_bar=True,
	)
	mcmc.run(key_run, init_params=init, extra_fields=("diverging", "num_steps"))
	z = mcmc.get_samples(group_by_chain=True)
	extra = mcmc.get_extra_fields(group_by_chain=True)

	return NUTSResult(
		item_draws=np.stack([np.asarray(z[name]) for name in ItemParams._fields], axis=-1),
		theta_draws=np.asarray(z["theta"]),
		n_divergent=int(extra["diverging"].sum()),
		tree_depth_hit_frac=float((extra["num_steps"] >= 2**config.max_tree_depth - 1).mean()),
	)


def summarize(draws: np.ndarray) -> dict[str, np.ndarray]:
	"""Posterior summary of every entry of draws shaped."""
	flat = draws.reshape(-1, *draws.shape[2:])
	q = np.quantile(flat, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

	return {
		"mean": flat.mean(axis=0),
		"sd": flat.std(axis=0, ddof=1),
		"q05": q[0],
		"q25": q[1],
		"q50": q[2],
		"q75": q[3],
		"q95": q[4],
		"ess_bulk": array_stats.ess(draws, chain_axis=0, draw_axis=1, method="bulk"),
		"ess_tail": array_stats.ess(
			draws, chain_axis=0, draw_axis=1, method="tail", prob=(0.05, 0.95)
		),
		"rhat": array_stats.rhat(draws, chain_axis=0, draw_axis=1),
	}


def _potential(
	z: dict[str, Array], family: str, person: Array, item: Array, x: Array, scale: float
) -> Array:
	"""Negative log posterior of (theta, tau)."""
	tau = ItemParams(*(z[name] for name in ItemParams._fields))
	theta = z["theta"]
	cell = ItemParams(*(c[item] for c in tau))
	log_lik = log_k(family, x, theta[person], cell).sum()

	return -(log_lik + log_prior(tau, scale) + norm.logpdf(theta).sum())
