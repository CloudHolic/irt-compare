import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irt_compare.models.transform import TransformSpec

X = jnp.linspace(1e-4, 1.0 - 1e-6, 4001)
SPLINE = {"kind": "spline", "knots": {"start": -8.0, "stop": 0.0, "num": 12}}


@pytest.mark.parametrize("raw", [None, {"kind": "boxcox"}, SPLINE])
def test_init_is_the_papers_linear_v(raw: dict | None) -> None:
	spec = TransformSpec.from_config(raw)
	g, log_dg = spec.apply(spec.init(), X)
	np.testing.assert_allclose(g, X, atol=1e-12)
	np.testing.assert_allclose(log_dg, 0.0, atol=1e-12)


def test_boxcox_is_a_power_of_the_distance_to_the_ceiling() -> None:
	spec = TransformSpec("boxcox")
	g, _ = spec.apply(jnp.log(0.4), X)
	np.testing.assert_allclose(1.0 - g, (1.0 - X) ** 0.4, rtol=1e-10)


def test_spline_is_monotone_with_fixed_endpoints_and_exact_derivative() -> None:
	spec = TransformSpec.from_config(SPLINE)
	raw = jnp.asarray(np.random.default_rng(1).normal(size=len(spec.knots)))
	g, log_dg = spec.apply(raw, X)
	assert bool(jnp.all(jnp.diff(g) > 0.0))
	ends, _ = spec.apply(raw, jnp.array([1e-12, 1.0 - 1e-15]))
	np.testing.assert_allclose(ends, [0.0, 1.0], atol=1e-9)
	autodiff = jax.vmap(jax.grad(lambda x: spec.apply(raw, x)[0]))(X)
	np.testing.assert_allclose(autodiff, jnp.exp(log_dg), rtol=1e-9)


def test_spline_with_constant_exponent_equals_boxcox() -> None:
	spec = TransformSpec.from_config(SPLINE)
	q = 0.4
	raw = jnp.full(len(spec.knots), jnp.log(jnp.expm1(q)))  # softplus^{-1}(q)
	g_spline, _ = spec.apply(raw, X)
	g_boxcox, _ = TransformSpec("boxcox").apply(jnp.log(q), X)
	np.testing.assert_allclose(g_spline, g_boxcox, atol=1e-12)


@pytest.mark.parametrize("knots", [(-1.0,), (-2.0, -3.0, 0.0), (-2.0, -1.0)])
def test_rejects_bad_knots(knots: tuple[float, ...]) -> None:
	with pytest.raises(ValueError):
		TransformSpec("spline", knots)
