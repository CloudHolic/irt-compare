"""MAP of the item parameters with theta marginalized by adaptive Gauss-Hermite quadrature."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from ..models import cnrm, zoi
from ..models.transform import TransformSpec
from .quadrature import adaptive_nodes, hermite

_GRID = np.linspace(-8.0, 8.0, 321)
_NEWTON_STEPS = 8
_NEWTON_CLIP = 0.5
_FLAT_SCALE = 0.3  # node scale where the log posterior is not concave at the mode


@dataclass(frozen=True)
class MAPConfig:
	"""Quadrature, optimizer (Adam with cosine decay) and node re-centring schedule."""

	n_nodes: int
	steps: int
	learning_rate: float
	final_lr_fraction: float
	scan_every: int
	prior_scale: float


class Cells(NamedTuple):
	"""Observed train cells. x_in is x with the endpoints replaced by 0.5 (transform input)."""

	person: Array
	item: Array
	x: Array
	x_in: Array


@dataclass(frozen=True)
class Model:
	"""A ZOI family, or the CNRM with a shared response-scale transform. Hashable (jit-static)."""

	family: str
	transform: TransformSpec | None = None

	def __post_init__(self) -> None:
		if self.family == "cnrm":
			if self.transform is None:
				raise ValueError("cnrm needs a transform (kind: linear for the paper's v)")
		elif self.family not in zoi.FAMILIES:
			raise ValueError(f"unknown family {self.family!r}")
		elif self.transform is not None:
			raise ValueError("only the cnrm takes a response-scale transform")

	def cell_log_k(self, params: dict[str, Array], cells: Cells, theta: Array) -> Array:
		"""log k at every cell; theta is (cells,) or (cells, nodes)."""
		col = (lambda v: v[cells.item]) if theta.ndim == 1 else (lambda v: v[cells.item][:, None])
		x = cells.x if theta.ndim == 1 else cells.x[:, None]

		if self.family == "cnrm":
			assert self.transform is not None
			g, log_dg = self.transform.apply(params["transform"], cells.x_in)
			scale = (g, log_dg) if theta.ndim == 1 else (g[:, None], log_dg[:, None])
			# The optimizer works in (log alpha, log sigma) as the paper does; an EM estimate read
			# back for evaluation carries alpha itself, which MML does not keep positive.
			alpha = params["alpha"] if "alpha" in params else jnp.exp(params["log_alpha"])
			return cnrm.log_k(
				x,
				theta,
				col(alpha),
				col(params["beta"]),
				jnp.exp(col(params["log_sigma"])),
				scale,
			)

		tau = zoi_tau(params)
		return zoi.log_k(self.family, x, theta, zoi.ItemParams(*(col(t) for t in tau)))

	def log_prior(self, params: dict[str, Array], scale: float) -> Array:
		"""ZOI: the paper's h(tau) at tau(z), no Jacobian (MAP). CNRM: N(0, scale) everywhere,
		plus the transform's smoothness penalty (P-spline only)."""
		if self.family == "cnrm":
			assert self.transform is not None
			flat = sum(
				(norm.logpdf(v, scale=scale).sum() for v in jax.tree_util.tree_leaves(params)),
				jnp.array(0.0),
			)
			return flat + self.transform.log_penalty(params["transform"])
		return zoi.log_prior(zoi.ItemParams(*zoi_tau(params)), scale)


@dataclass(frozen=True)
class MAPResult:
	"""Point estimates, the loss trace, and train-person EAPs at the estimate."""

	params: dict[str, np.ndarray]
	loss: np.ndarray
	best_loss: float
	best_step: int
	log_marginal: float
	eap: np.ndarray
	psd: np.ndarray

	@property
	def drop_last(self) -> float:
		"""How much the loss fell over the last 200 steps: the convergence check."""
		k = min(200, len(self.loss) - 1)
		return float(self.loss[-k - 1] - self.loss[-1]) if k > 0 else float("nan")


def zoi_tau(params: dict[str, Array]) -> tuple[Array, ...]:
	"""(log_a, b, dispersion, gamma_0, gamma_1) with gamma_1 = gamma_0 + exp(lambda) > gamma_0."""
	return (
		params["log_a"],
		params["b"],
		params["dispersion"],
		params["gamma_0"],
		params["gamma_0"] + jnp.exp(params["lambda"]),
	)


def zoi_params(
	log_a: np.ndarray,
	b: np.ndarray,
	dispersion: np.ndarray,
	gamma_0: np.ndarray,
	gamma_1: np.ndarray,
) -> dict[str, Array]:
	"""The optimizer's coordinates from tau (inverse of zoi_tau)."""
	gap = np.maximum(np.asarray(gamma_1) - np.asarray(gamma_0), 1e-3)
	return {
		"log_a": jnp.asarray(log_a),
		"b": jnp.asarray(b),
		"dispersion": jnp.asarray(dispersion),
		"gamma_0": jnp.asarray(gamma_0),
		"lambda": jnp.asarray(np.log(gap)),
	}


def cnrm_params(
	alpha: np.ndarray, beta: np.ndarray, sigma: np.ndarray, spec: TransformSpec
) -> dict[str, Array]:
	"""The optimizer's coordinates: (log alpha, beta, log sigma) as in the paper, plus G's."""
	return {
		"log_alpha": jnp.asarray(np.log(np.maximum(np.asarray(alpha), 1e-3))),
		"beta": jnp.asarray(beta),
		"log_sigma": jnp.asarray(np.log(sigma)),
		"transform": spec.init(),
	}


def fit_map(
	model: Model,
	person: np.ndarray,
	item: np.ndarray,
	x: np.ndarray,
	n_persons: int,
	init: dict[str, Array],
	config: MAPConfig,
	log_every: int = 100,
) -> MAPResult:
	"""Adam on the AGHQ-marginalized MAP objective, keeping the best iterate."""
	cells = _cells(person, item, x)
	x_h, log_w_h = hermite(config.n_nodes)
	params = init
	m1 = jax.tree_util.tree_map(jnp.zeros_like, params)
	m2 = jax.tree_util.tree_map(jnp.zeros_like, params)
	mode = jnp.zeros(n_persons)

	trace: list[float] = []
	best = (np.inf, params, 0)
	for step in range(config.steps):
		mode, scale = _modes(model, params, cells, n_persons, mode, step % config.scan_every == 0)
		nodes, log_w = adaptive_nodes(mode, scale, x_h, log_w_h)
		value, grads = _value_and_grad(
			model, params, cells, n_persons, nodes, log_w, config.prior_scale
		)
		value = float(value)
		if not np.isfinite(value) or not all(
			bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads)
		):
			raise FloatingPointError(f"non-finite loss or gradient at step {step}")

		trace.append(value)
		if value < best[0]:
			best = (value, params, step)
		if log_every and step % log_every == 0:
			print(f"step {step} loss {value:.3f}", flush=True)

		lr = _cosine(step, config)
		params, m1, m2 = _adam(params, grads, m1, m2, step + 1, lr)

	params = best[1]
	mode, scale = _modes(model, params, cells, n_persons, mode, True)
	nodes, log_w = adaptive_nodes(mode, scale, x_h, log_w_h)
	joint = _joint(model, params, cells, n_persons, nodes, log_w)
	log_marginal = logsumexp(joint, axis=1)
	post = jnp.exp(joint - log_marginal[:, None])
	eap = (post * nodes).sum(axis=1)
	psd = jnp.sqrt((post * (nodes - eap[:, None]) ** 2).sum(axis=1))

	return MAPResult(
		params={k: np.asarray(v) for k, v in params.items()},
		loss=np.asarray(trace),
		best_loss=float(best[0]),
		best_step=int(best[2]),
		log_marginal=float(log_marginal.sum()),
		eap=np.asarray(eap),
		psd=np.asarray(psd),
	)


def _cells(person: np.ndarray, item: np.ndarray, x: np.ndarray) -> Cells:
	xs = jnp.asarray(x)
	return Cells(
		jnp.asarray(person), jnp.asarray(item), xs, jnp.where((xs == 0.0) | (xs == 1.0), 0.5, xs)
	)


def _cosine(step: int, config: MAPConfig) -> float:
	frac = config.final_lr_fraction
	return config.learning_rate * (
		frac + (1.0 - frac) * 0.5 * (1.0 + np.cos(np.pi * step / config.steps))
	)


Params = dict[str, Array]


@jax.jit
def _adam(
	params: Params, grads: Params, m1: Params, m2: Params, t: int, lr: float
) -> tuple[Params, Params, Params]:
	b1, b2, eps = 0.9, 0.999, 1e-8
	m1 = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m1, grads)
	m2 = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, m2, grads)
	c1, c2 = 1 - b1**t, 1 - b2**t
	params = jax.tree_util.tree_map(
		lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), params, m1, m2
	)
	return params, m1, m2


def _joint(
	model: Model, params: dict[str, Array], cells: Cells, n_persons: int, nodes: Array, log_w: Array
) -> Array:
	"""log L_p(theta_ph) + log w_ph, shape (persons, nodes)."""
	lk = model.cell_log_k(params, cells, nodes[cells.person])
	return jax.ops.segment_sum(lk, cells.person, num_segments=n_persons) + log_w


@partial(jax.jit, static_argnames=("model", "n_persons", "prior_scale"))
def _value_and_grad(
	model: Model,
	params: dict[str, Array],
	cells: Cells,
	n_persons: int,
	nodes: Array,
	log_w: Array,
	prior_scale: float,
) -> tuple[Array, dict[str, Array]]:
	def loss(p: dict[str, Array]) -> Array:
		lm = logsumexp(_joint(model, p, cells, n_persons, nodes, log_w), axis=1).sum()
		return -(lm + model.log_prior(p, prior_scale))

	return jax.value_and_grad(loss)(params)


@partial(jax.jit, static_argnames=("model", "n_persons", "scan"))
def _modes(
	model: Model, params: dict[str, Array], cells: Cells, n_persons: int, start: Array, scan: bool
) -> tuple[Array, Array]:
	"""Each person's posterior mode of theta and the node scale (-f'')^(-1/2) there."""

	def log_post(theta: Array) -> Array:
		lk = model.cell_log_k(params, cells, theta[cells.person])
		return jax.ops.segment_sum(lk, cells.person, num_segments=n_persons) + norm.logpdf(theta)

	def total(theta: Array) -> Array:
		return log_post(theta).sum()

	grad = jax.grad(total)

	def curvature(theta: Array) -> Array:
		# The log posterior is a sum of one-person terms: its Hessian is diagonal, and one
		# Hessian-vector product with ones returns that diagonal.
		return jax.jvp(grad, (theta,), (jnp.ones_like(theta),))[1]

	mode = start
	if scan:
		# A non-concave posterior (the simplex) can hold Newton in a poor local mode.
		grid = jnp.asarray(_GRID)
		on_grid = jax.lax.map(lambda t: log_post(jnp.full(n_persons, t)), grid)  # (G, persons)
		best = grid[jnp.argmax(on_grid, axis=0)]
		mode = jnp.where(log_post(start) >= on_grid.max(axis=0), start, best)

	def step(_: int, theta: Array) -> Array:
		h = curvature(theta)
		move = jnp.where(h < 0.0, -grad(theta) / h, 0.0)
		return theta + jnp.clip(move, -_NEWTON_CLIP, _NEWTON_CLIP)

	mode = jax.lax.fori_loop(0, _NEWTON_STEPS, step, mode)
	h = curvature(mode)
	return mode, jnp.where(h < -1e-8, 1.0 / jnp.sqrt(-h), _FLAT_SCALE)


def zoi_moment_init(family: str, item: np.ndarray, x: np.ndarray, n_items: int) -> dict[str, Array]:
	"""A data-only starting point: each item's interior mean and spread, and endpoint rates.

	b puts the interior location at the item's mean for theta = 0; the dispersion is solved from
	the interior variance for each family's variance law; gamma_0/gamma_1 match the endpoint
	rates at theta = 0 (clipped, so an item without zeros starts far down, not at -inf).
	"""
	n = np.bincount(item, minlength=n_items).astype(float)
	inside = (x > 0.0) & (x < 1.0)
	n_in = np.maximum(np.bincount(item, inside, n_items), 1.0)
	x_in = np.where(inside, x, 0.0)
	mean = np.clip(np.bincount(item, x_in, n_items) / n_in, 0.02, 0.98)
	var = np.bincount(item, np.where(inside, (x - mean[item]) ** 2, 0.0), n_items) / n_in
	var = np.clip(var, 1e-6, mean * (1.0 - mean) * 0.99)
	b = np.log(mean / (1.0 - mean))

	if family == "zoi-beta":
		precision = np.maximum(mean * (1.0 - mean) / var - 1.0, 1.0)
		dispersion = 2.0 * np.log(precision / (2.0 * np.cosh(b / 2.0)))
	elif family == "zoi-simplex":
		# Var ~ phi mu^3 (1 - mu)^3 for small phi.
		dispersion = np.log(var / (mean * (1.0 - mean)) ** 3)
	else:
		logit = np.log(np.clip(x, 1e-6, 1 - 1e-6) / np.clip(1.0 - x, 1e-6, None))
		lm = np.bincount(item, np.where(inside, logit, 0.0), n_items) / n_in
		lv = np.bincount(item, np.where(inside, (logit - lm[item]) ** 2, 0.0), n_items) / n_in
		dispersion = np.log(np.maximum(lv, 1e-3))

	p0 = np.clip(np.bincount(item, x == 0.0, n_items) / n, 1e-4, 0.5)
	p1 = np.clip(np.bincount(item, x == 1.0, n_items) / n, 1e-4, 0.5)
	gamma_0 = np.log(p0 / (1.0 - p0))
	gamma_1 = np.log((1.0 - p1) / p1)
	return zoi_params(np.full(n_items, np.log(0.5)), b, dispersion, gamma_0, gamma_1)
