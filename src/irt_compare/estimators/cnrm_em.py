"""EM-based marginal maximum likelihood for the CNRM.

Only observed (person, item) cells enter the sums: missing responses are ignorable.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from ..models.cnrm import log_k
from ..models.partition import partition

INITS = ("person_mean_ols",)


@dataclass(frozen=True)
class EMConfig:
	"""Quadrature nodes, stopping rule and initial-value method."""

	n_nodes: int
	tol: float
	max_iter: int
	init: str

	def __post_init__(self) -> None:
		if self.init not in INITS:
			raise ValueError(f"init must be one of {INITS}, got {self.init!r}")


@dataclass(frozen=True)
class EMResult:
	"""Item estimates, the marginal log-likelihood per iteration, and train-person EAPs.

	When an update turns non-finite, EM stops there: the estimates are the last finite ones
	and `degenerate_items` lists the items whose update failed.
	"""

	alpha: np.ndarray
	beta: np.ndarray
	sigma: np.ndarray
	loglik: np.ndarray
	n_iter: int
	converged: bool
	degenerate_items: np.ndarray
	eap: np.ndarray
	psd: np.ndarray

	@property
	def degenerate(self) -> bool:
		"""True when EM stopped because an update turned non-finite."""
		return self.degenerate_items.size > 0


def gh_nodes(n: int) -> tuple[Array, Array]:
	"""Gauss-Hermite nodes and log weights for an expectation under N(0, 1)."""
	t, w = np.polynomial.hermite.hermgauss(n)
	with np.errstate(divide="ignore"):
		log_w = np.log(w) - 0.5 * np.log(np.pi)

	return jnp.asarray(np.sqrt(2.0) * t), jnp.asarray(log_w)


def init_person_mean_ols(
	person: np.ndarray, item: np.ndarray, x: np.ndarray, n_persons: int, n_items: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Item-wise OLS of responses on standardized persom means (slope, intercept, residual SE)."""
	mean_p = np.bincount(person, x, n_persons) / np.bincount(person, minlength=n_persons)
	t = ((mean_p - mean_p.mean()) / mean_p.std())[person]

	n = np.bincount(item, minlength=n_items).astype(float)
	t_bar = np.bincount(item, t, n_items) / n
	x_bar = np.bincount(item, x, n_items) / n
	dt = t - t_bar[item]
	sxx = np.bincount(item, dt**2, n_items)
	sxy = np.bincount(item, dt * (x - x_bar[item]), n_items)

	with np.errstate(divide="ignore", invalid="ignore"):
		slope = sxy / sxx
		intercept = x_bar - slope * t_bar
		sse = np.bincount(item, (x - intercept[item] - slope[item] * t) ** 2, n_items)
		rse = np.sqrt(sse / (n - 2.0))

	dt_all = t - t.mean()
	slope_all = (dt_all * (x - x.mean())).sum() / (dt_all**2).sum()
	intercept_all = x.mean() - slope_all * t.mean()
	rse_all = np.sqrt(((x - intercept_all - slope_all * t) ** 2).sum() / (len(x) - 2.0))

	valid = (sxx > 0.0) & (n > 2.0) & (rse > 0.0)
	return (
		np.where(valid, slope, slope_all),
		np.where(valid, intercept, intercept_all),
		np.where(valid, rse, rse_all),
	)


def fit_em(
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	n_items: int,
	config: EMConfig,
) -> EMResult:
	"""Runs EM until every parameter moves less than `tol` or `max_iter` is reahced."""
	nodes, log_w = gh_nodes(config.n_nodes)
	p, i, xs = jnp.asarray(person), jnp.asarray(item), jnp.asarray(x)
	alpha, beta, sigma = (
		jnp.asarray(v) for v in init_person_mean_ols(person, item, x, n_persons, n_items)
	)

	loglik = []
	converged = False
	degenerate_items = np.array([], dtype=np.int64)

	for _ in range(config.max_iter):
		new_alpha, new_beta, new_sigma, ll = _em_step(
			p, i, xs, n_persons, n_items, alpha, beta, sigma, nodes, log_w
		)
		loglik.append(float(ll))

		finite = jnp.isfinite(new_alpha) & jnp.isfinite(new_beta) & jnp.isfinite(new_sigma)
		if not bool(finite.all()):
			# Nothing in MML bounds sigma away from 0: an item with too few responses drives it
			# there until the posterior over the nodes collapses and the update is 0/0.
			degenerate_items = np.flatnonzero(~np.asarray(finite))
			break

		change = max(
			float(jnp.abs(new_alpha - alpha).max()),
			float(jnp.abs(new_beta - beta).max()),
			float(jnp.abs(new_sigma - sigma).max()),
		)
		alpha, beta, sigma = new_alpha, new_beta, new_sigma
		if change < config.tol:
			converged = True
			break

	post, _ = _posterior(p, i, xs, n_persons, alpha, beta, sigma, nodes, log_w)
	eap = post @ nodes
	psd = jnp.sqrt((post * (nodes[None, :] - eap[:, None]) ** 2).sum(axis=1))

	return EMResult(
		alpha=np.asarray(alpha),
		beta=np.asarray(beta),
		sigma=np.asarray(sigma),
		loglik=np.asarray(loglik),
		n_iter=len(loglik),
		converged=converged,
		degenerate_items=degenerate_items,
		eap=np.asarray(eap),
		psd=np.asarray(psd),
	)


def person_loglik(
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	alpha: np.ndarray,
	beta: np.ndarray,
	sigma: np.ndarray,
	n_nodes: int,
) -> np.ndarray:
	"""Each person's log marginal likelihood by Gauss-Hermite with `n_nodes` nodes."""
	nodes, log_w = gh_nodes(n_nodes)
	_, log_marginal = _posterior(
		jnp.asarray(person),
		jnp.asarray(item),
		jnp.asarray(x),
		n_persons,
		jnp.asarray(alpha),
		jnp.asarray(beta),
		jnp.asarray(sigma),
		nodes,
		log_w,
	)

	return np.asarray(log_marginal)


def _node_loglik(
	person: Array,
	item: Array,
	x: Array,
	n_persons: int,
	alpha: Array,
	beta: Array,
	sigma: Array,
	nodes: Array,
) -> Array:
	"""log L_p(t_h): each person's log-likelihood at every node, shape (persons, nodes)."""
	lk = log_k(x[:, None], nodes[None, :], alpha[item, None], beta[item, None], sigma[item, None])

	return jax.ops.segment_sum(lk, person, num_segments=n_persons)


@partial(jax.jit, static_argnames=("n_persons",))
def _posterior(
	person: Array,
	item: Array,
	x: Array,
	n_persons: int,
	alpha: Array,
	beta: Array,
	sigma: Array,
	nodes: Array,
	log_w: Array,
) -> tuple[Array, Array]:
	"""Posterior weights over nodes  and each person's log marginal likelihood."""
	joint = _node_loglik(person, item, x, n_persons, alpha, beta, sigma, nodes) + log_w
	log_marginal = logsumexp(joint, axis=1)

	return jnp.exp(joint - log_marginal[:, None]), log_marginal


@partial(jax.jit, static_argnames=("n_persons", "n_items"))
def _em_step(
	person: Array,
	item: Array,
	x: Array,
	n_persons: int,
	n_items: int,
	alpha: Array,
	beta: Array,
	sigma: Array,
	nodes: Array,
	log_w: Array,
) -> tuple[Array, Array, Array, Array]:
	"""One E-step and M-step."""
	post, log_marginal = _posterior(person, item, x, n_persons, alpha, beta, sigma, nodes, log_w)

	a, b, s = alpha[item, None], beta[item, None], sigma[item, None]
	mu = a * nodes[None, :] + b

	z0 = -mu / s
	z1 = (1.0 - mu) / s

	mills0 = jnp.exp(norm.logpdf(z0) - norm.logcdf(z0))
	mills1 = jnp.exp(norm.logpdf(z1) - norm.logcdf(-z1))

	part = partition(x)
	zero, one, xc = part.zero[:, None], part.one[:, None], x[:, None]

	ey = jnp.where(zero, mu - s * mills0, jnp.where(one, mu + s * mills1, xc))
	ey2 = jnp.where(
		zero,
		mu**2 - s * mu * mills0 + s**2,
		jnp.where(one, mu**2 + s * (1.0 + mu) * mills1 + s**2, xc**2),
	)

	w = post[person]
	t = nodes[None, :]
	n_i = jax.ops.segment_sum(jnp.ones_like(x), item, num_segments=n_items)

	def mean(v: Array) -> Array:
		return jax.ops.segment_sum(v.sum(axis=1), item, num_segments=n_items) / n_i

	m_t, m_t2 = mean(w * t), mean(w * t**2)
	m_y, m_y2, m_ty = mean(w * ey), mean(w * ey2), mean(w * t * ey)

	alpha_new = (m_ty - m_t * m_y) / (m_t2 - m_t**2)
	beta_new = m_y - alpha_new * m_t
	sigma_new = jnp.sqrt(
		m_y2
		- 2.0 * alpha_new * m_ty
		- 2.0 * beta_new * m_y
		+ alpha_new**2 * m_t2
		+ 2.0 * alpha_new * beta_new * m_t
		+ beta_new**2
	)

	return alpha_new, beta_new, sigma_new, log_marginal.sum()
