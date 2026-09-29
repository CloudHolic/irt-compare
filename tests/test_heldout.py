from pathlib import Path

import numpy as np

from irt_compare.dataset import TrainData
from irt_compare.estimators import cnrm_em
from irt_compare.estimators.map_aghq import Model, cnrm_params
from irt_compare.evaluation.fitted import Fitted
from irt_compare.evaluation.heldout import paired_difference, person_posterior
from irt_compare.models.transform import TransformSpec


def test_grid_marginal_matches_adaptive_quadrature(cnrm_cells) -> None:
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
	aghq = cnrm_em.person_loglik(person, item, x, n_persons, em.alpha, em.beta, em.sigma, 61)
	spec = TransformSpec("linear")
	fitted = Fitted("em", Model("cnrm", spec), cnrm_params(em.alpha, em.beta, em.sigma, spec), "em")
	data = TrainData(person, item, x, n_persons, n_items, np.arange(n_persons), "", Path())
	np.testing.assert_allclose(person_posterior(fitted, data).log_marginal, aghq, atol=1e-6)


def test_paired_difference_is_the_sum_and_a_positive_se() -> None:
	a = np.zeros(100)
	b = np.random.default_rng(0).normal(1.0, 1.0, 100)
	diff, se = paired_difference(a, b)
	assert diff == b.sum()
	assert 5.0 < se < 15.0
