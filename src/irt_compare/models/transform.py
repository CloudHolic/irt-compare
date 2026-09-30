"""Monotone response-scale transforms for the CNRM.

The CNRM censors Y* ~ N(alpha theta + beta, sigma^2) to [0, 1] and maps it to the response with a
strictly increasing v. The paper takes v linear, which after normalization is the identity on
every item. Here v stays shared by all items and only its shape is freed: G = v^{-1} takes a
response x to Y*'s scale, defined on s = log(1 - x) through a local exponent q(s) > 0,

    log(1 - G(x)) = h(s) = -int_s^0 q(r) dr,    log G'(x) = h(s) + log q(s) - s.

- linear: q = 1, G(x) = x (the paper).
- boxcox: q constant, 1 - G(x) = (1 - x)^q (Box-Cox on the distance to the ceiling).
- spline: q piecewise linear over knots in s, held at its first value below the first knot,
  so the ceiling keeps a power-law tail. q = softplus(raw) at the knots.
- pspline: the same interpolation with q = exp(raw) at many knots, and a second-difference
  penalty on log q (Eilers & Marx, 1996) instead of a hand-picked knot count. The penalty's null
  space is log q linear in s, which contains every constant q (Box-Cox); `penalty` is its weight.

G(0) = 0 and G(1) = 1 for every q, so no part of G can trade off against an item's
(alpha, beta, sigma) and the censoring points stay at 0 and 1.
"""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

KINDS = ("linear", "boxcox", "spline", "pspline")
_KNOTTED = ("spline", "pspline")
_SOFTPLUS_INV_1 = float(np.log(np.e - 1.0))


@dataclass(frozen=True)
class TransformSpec:
	"""Which transform, the spline's knots in s = log(1 - x) (increasing, last one 0), and the
	P-spline's penalty weight."""

	kind: str
	knots: tuple[float, ...] = ()
	penalty: float = 0.0

	def __post_init__(self) -> None:
		if self.kind not in KINDS:
			raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
		if self.kind in _KNOTTED:
			k = np.asarray(self.knots, dtype=float)
			if k.size < 2 or np.any(np.diff(k) <= 0.0) or k[-1] != 0.0:
				raise ValueError("spline knots must be increasing, at least two, and end at 0")
		elif self.knots:
			raise ValueError(f"{self.kind} takes no knots")
		if self.kind == "pspline":
			if self.penalty < 0.0 or len(self.knots) < 3:
				raise ValueError("pspline needs a penalty >= 0 and at least three knots")
		elif self.penalty != 0.0:
			raise ValueError(f"{self.kind} takes no penalty")

	@classmethod
	def from_config(cls, raw: dict | None) -> Self:
		"""{kind: spline, knots: {start, stop, num}}, the same with `penalty` for pspline, or
		{kind: boxcox}; None means linear."""
		if raw is None:
			return cls("linear")
		knots = raw.get("knots")
		penalty = float(raw.get("penalty", 0.0))
		if knots is None:
			return cls(raw["kind"], penalty=penalty)
		grid = np.linspace(knots["start"], knots["stop"], knots["num"])
		return cls(raw["kind"], tuple(float(v) for v in grid), penalty)

	def init(self) -> Array:
		"""Raw parameters at which G is the identity (the paper's linear v)."""
		if self.kind == "linear":
			return jnp.zeros((0,))
		if self.kind == "boxcox":
			return jnp.zeros(())  # log q
		if self.kind == "pspline":
			return jnp.zeros(len(self.knots))  # log q = 0 at every knot
		return jnp.full(len(self.knots), _SOFTPLUS_INV_1)  # softplus^{-1}(1) at every knot

	def exponent(self, raw: Array, s: Array) -> Array:
		"""The local exponent q(s)."""
		if self.kind == "linear":
			return jnp.ones_like(s)
		if self.kind == "boxcox":
			return jnp.exp(raw) * jnp.ones_like(s)
		return jnp.interp(s, jnp.asarray(self.knots), self._knot_exponent(raw))

	def log_penalty(self, raw: Array) -> Array:
		"""-penalty / 2 * sum of squared second differences of log q over the knots (pspline);
		zero for the other kinds. Read as a second-order random-walk prior on log q."""
		if self.kind != "pspline":
			return jnp.zeros(())
		d2 = raw[2:] - 2.0 * raw[1:-1] + raw[:-2]
		return -0.5 * self.penalty * jnp.sum(d2**2)

	def apply(self, raw: Array, x: Array) -> tuple[Array, Array]:
		"""G(x) and log G'(x) for interior responses 0 < x < 1."""
		s = jnp.log1p(-x)
		h = self._h(raw, s)
		log_dg = h + jnp.log(self.exponent(raw, s)) - s
		return -jnp.expm1(h), log_dg

	def _h(self, raw: Array, s: Array) -> Array:
		if self.kind == "linear":
			return s
		if self.kind == "boxcox":
			return jnp.exp(raw) * s

		knots = jnp.asarray(self.knots)
		q = self._knot_exponent(raw)
		area = (q[1:] + q[:-1]) / 2.0 * jnp.diff(knots)
		tail_area = jnp.concatenate([jnp.cumsum(area[::-1])[::-1], jnp.zeros(1)])  # int_{k_j}^0 q

		j = jnp.clip(jnp.searchsorted(knots, s) - 1, 0, knots.size - 2)
		q_s = jnp.interp(s, knots, q)
		inside = (q_s + q[j + 1]) / 2.0 * (knots[j + 1] - s) + tail_area[j + 1]
		below = tail_area[0] + q[0] * (knots[0] - s)

		return -jnp.where(s < knots[0], below, inside)

	def _knot_exponent(self, raw: Array) -> Array:
		return jnp.exp(raw) if self.kind == "pspline" else jax.nn.softplus(raw)
