import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def cnrm_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
	"""A small censored-normal data set: 60 persons x 8 items, every cell observed."""
	rng = np.random.default_rng(0)
	n_persons, n_items = 60, 8
	theta = rng.normal(size=n_persons)
	alpha, beta, sigma = rng.uniform(0.05, 0.15, n_items), rng.uniform(0.8, 0.95, n_items), 0.05
	person, item = (
		a.ravel() for a in np.meshgrid(np.arange(n_persons), np.arange(n_items), indexing="ij")
	)
	y = alpha[item] * theta[person] + beta[item] + sigma * rng.normal(size=person.size)
	return person, item, np.clip(y, 0.0, 1.0), n_persons, n_items
