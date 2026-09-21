import jax
import jax.numpy as jnp
from numpyro.infer import CustomGibbs

from fast_polya_gamma_jax.numpyro import make_custom_gibbs, make_pg1_gibbs_fn


def test_gibbs_adapter_updates_named_site():
    def linear_predictor_fn(*, gibbs_sites, hmc_sites):
        del gibbs_sites
        return hmc_sites["eta"]

    update = make_pg1_gibbs_fn(
        linear_predictor_fn,
        site_name="omega",
        num_terms=8,
    )
    result = update(
        rng_key=jax.random.key(0),
        gibbs_sites={"omega": jnp.ones(3)},
        hmc_sites={"eta": jnp.array([0.0, 1.0, 2.0])},
    )
    assert set(result) == {"omega"}
    assert result["omega"].shape == (3,)
    assert bool(jnp.all(result["omega"] > 0.0))


def test_custom_gibbs_factory_returns_numpyro_kernel():
    def linear_predictor_fn(*, gibbs_sites, hmc_sites):
        del gibbs_sites
        return hmc_sites["eta"]

    kernel = make_custom_gibbs(linear_predictor_fn, num_terms=8)
    assert isinstance(kernel, CustomGibbs)
