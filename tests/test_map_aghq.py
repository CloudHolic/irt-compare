import jax.numpy as jnp
import numpy as np
import pytest

from irt_compare.estimators import cnrm_em
from irt_compare.estimators.map_aghq import (
	MAPConfig,
	Model,
	cnrm_params,
	fit_map,
	zoi_moment_init,
)
from irt_compare.models.transform import TransformSpec

CONFIG = MAPConfig(
	n_nodes=15, steps=60, learning_rate=0.01, final_lr_fraction=0.1, scan_every=20, prior_scale=10.0
)


def test_linear_cnrm_log_marginal_matches_em_quadrature(cnrm_cells) -> None:
	"""At the EM estimate, the MAP code's AGHQ log marginal is the EM code's."""
	person, item, x, n_persons, n_items = cnrm_cells
	em = cnrm_em.fit_em(
		person,
		item,
		x,
		n_persons,
		n_items,
		cnrm_em.EMConfig(
			n_nodes=31, tol=1e-6, max_iter=300, init="person_mean_ols", quadrature="adaptive"
		),
	)
	reference = cnrm_em.person_loglik(
		person, item, x, n_persons, em.alpha, em.beta, em.sigma, 31
	).sum()
	spec = TransformSpec("linear")
	config = MAPConfig(
		n_nodes=31,
		steps=1,
		learning_rate=0.0,
		final_lr_fraction=1.0,
		scan_every=1,
		prior_scale=10.0,
	)
	result = fit_map(
		Model("cnrm", spec),
		person,
		item,
		x,
		n_persons,
		cnrm_params(em.alpha, em.beta, em.sigma, spec),
		config,
		log_every=0,
	)
	np.testing.assert_allclose(result.log_marginal, reference, rtol=1e-8)


@pytest.mark.parametrize("kind", ["linear", "boxcox", "spline"])
def test_cnrm_map_decreases_the_loss(cnrm_cells, kind: str) -> None:
	person, item, x, n_persons, n_items = cnrm_cells
	raw = {
		"kind": kind,
		**({"knots": {"start": -6.0, "stop": 0.0, "num": 5}} if kind == "spline" else {}),
	}
	spec = TransformSpec.from_config(raw)
	init = cnrm_params(*cnrm_em.init_person_mean_ols(person, item, x, n_persons, n_items), spec)
	result = fit_map(Model("cnrm", spec), person, item, x, n_persons, init, CONFIG, log_every=0)
	assert np.isfinite(result.best_loss)
	assert result.best_loss < result.loss[0]


@pytest.mark.parametrize("family", ["zoi-beta", "zoi-sb", "zoi-simplex"])
def test_zoi_map_decreases_the_loss(cnrm_cells, family: str) -> None:
	person, item, x, n_persons, n_items = cnrm_cells
	init = zoi_moment_init(family, item, x, n_items)
	result = fit_map(Model(family), person, item, x, n_persons, init, CONFIG, log_every=0)
	assert np.isfinite(result.best_loss)
	assert result.best_loss < result.loss[0]
	assert np.all(np.isfinite(result.eap))


def test_model_rejects_mismatched_transform() -> None:
	with pytest.raises(ValueError):
		Model("zoi-beta", TransformSpec("linear"))
	with pytest.raises(ValueError):
		Model("cnrm")
	assert jnp.isfinite(0.0)
