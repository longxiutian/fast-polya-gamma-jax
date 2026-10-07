"""Numerical and model-contract tests for the non-mixture HB sampler."""

import jax
import jax.numpy as jnp
import numpy as np

from fast_polya_gamma_jax.classic_hb_gibbs import (
    _interweave_location,
    _interweave_scales,
    classic_sweep,
    initialize_classic_state,
    original_classic_priors,
    prepare_classic_data,
    run_classic_collect,
)

jax.config.update("jax_enable_x64", True)


def small_panel():
    return prepare_classic_data(
        y=np.array([1, 0, 1, 0, 0, 1], dtype=float),
        person_index=np.array([0, 0, 1, 1, 2, 2]),
        campaign_index=np.array([0, 0, 1, 1, 0, 1]),
        item_index=np.array([0, 1, 2, 2, 1, 2]),
        creative_features=np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=float),
        controls=np.array([[1, 0], [1, 1], [1, 0],
                           [1, 1], [1, 0], [1, 1]], dtype=float),
        respondent_features=np.array([[0.2, -0.1], [0.5, 0.1],
                                      [-0.3, 0.4]], dtype=float),
    )


def test_original_prior_constants():
    prior = original_classic_priors(covariates=73)
    assert np.isclose(prior.mu_a_sd, 1.5)
    assert np.isclose(prior.sigma_a_scale, 0.25)
    assert np.isclose(prior.mu_b_sd, 1.0)
    assert np.isclose(prior.gamma_global_scale, 0.1 / np.sqrt(73))
    assert np.isclose(prior.gamma_slab_sd, 1.0)
    assert np.isclose(prior.sigma_b_df, 4.0)
    assert np.isclose(prior.sigma_b_scale, 0.5)
    assert np.isclose(prior.lkj_concentration, 2.0)
    assert np.isclose(prior.delta_sd, 0.5)


def test_off_centered_location_conditional():
    data = small_panel()
    priors = original_classic_priors(covariates=2)
    person = jnp.array([[0.2, 0.5, -0.1, 0.3],
                        [-0.4, 0.1, 0.2, -0.2],
                        [0.1, -0.2, 0.4, 0.1]])
    mu_a = jnp.array(0.15)
    mu_b = jnp.array([0.1, -0.2, 0.05])
    omega = jnp.array([0.2, 0.3, 0.15, 0.27, 0.22, 0.18])
    kappa = data.panel.y - 0.5
    campaign = jnp.array([0.2, -0.1])
    item = jnp.array([0.0, 0.1, -0.2])
    delta = jnp.array([0.1, -0.1])
    residual = person - jnp.concatenate([mu_a[None], mu_b])[None, :]
    x = data.panel.person_design
    offset = (
        jnp.sum(x * residual[data.panel.person_index], axis=1)
        + campaign[data.panel.campaign_index]
        + item[data.panel.item_index]
        + data.panel.controls @ delta
    )
    prior_precision = jnp.diag(jnp.array([1 / 1.5**2, 1, 1, 1]))
    precision = prior_precision + x.T @ (omega[:, None] * x)
    expected = jnp.linalg.solve(precision, x.T @ (kappa - omega * offset))
    keys = jax.random.split(jax.random.key(12), 4000)
    draw = jax.jit(jax.vmap(lambda key: _interweave_location(
        key, person=person, mu_a=mu_a, mu_b=mu_b, campaign=campaign,
        item=item, delta=delta, data=data, omega=omega, kappa=kappa,
        priors=priors,
    )))
    new_person, new_mu_a, new_mu_b = draw(keys)
    location = jnp.concatenate([new_mu_a[:, None], new_mu_b], axis=1)
    np.testing.assert_allclose(np.asarray(location.mean(axis=0)), expected, atol=0.04)
    np.testing.assert_allclose(
        np.asarray(new_person - location[:, None, :]),
        np.broadcast_to(np.asarray(residual), new_person.shape),
        atol=1e-12,
    )


def test_jit_sweep_and_archives():
    data = small_panel()
    priors = original_classic_priors(covariates=2)
    state = initialize_classic_state(
        jax.random.key(4), people=3, campaigns=2, items=3,
        controls=2, covariates=2,
    )
    next_state, metrics = jax.jit(classic_sweep)(state, data, priors)
    assert bool(metrics.all_finite)
    assert float(metrics.omega_min) > 0
    assert float(metrics.person_cholesky_min) > 0
    assert metrics.off_centered_scale_acceptance.shape == (4,)
    assert next_state.person.shape == (3, 4)
    collect = jax.jit(lambda initial: run_classic_collect(
        initial, data, priors, num_steps=10, archive_stride=5,
    ))
    final, metrics, globals_, summed, squared, archive = collect(next_state)
    assert final.person.shape == (3, 4)
    assert metrics.mu_b.shape == (10, 3)
    assert globals_["Gamma"].shape == (10, 3, 2)
    assert archive.shape == (2, 3, 4)
    assert summed.shape == squared.shape == (3, 4)
    assert bool(metrics.all_finite.all())
    assert np.isfinite(np.asarray(archive)).all()


def test_off_centered_scales_preserve_standardized_residuals_and_correlation():
    data = small_panel()
    priors = original_classic_priors(covariates=2)
    mu_a = jnp.array(0.2)
    mu_b = jnp.array([0.1, -0.2, 0.05])
    gamma = jnp.zeros((3, 2))
    person = jnp.array([[0.5, 0.5, -0.1, 0.3],
                        [-0.1, 0.1, 0.2, -0.2],
                        [0.2, -0.2, 0.4, 0.1]])
    sigma_a2 = jnp.array(0.09)
    sigma_b = jnp.array([[0.25, 0.04, 0.0],
                         [0.04, 0.36, 0.03],
                         [0.0, 0.03, 0.16]])
    old_tau = jnp.sqrt(jnp.diag(sigma_b))
    old_a = (person[:, 0] - mu_a) / jnp.sqrt(sigma_a2)
    old_b = (person[:, 1:] - mu_b) / old_tau
    keys = jax.random.split(jax.random.key(23), 30)
    update = jax.jit(jax.vmap(lambda key: _interweave_scales(
        key, person=person, mu_a=mu_a, mu_b=mu_b, gamma=gamma,
        sigma_a2=sigma_a2, sigma_b=sigma_b, data=data,
        omega=jnp.full((6,), 0.2), kappa=data.panel.y - 0.5,
        campaign=jnp.zeros((2,)), item=jnp.zeros((3,)),
        delta=jnp.zeros((2,)), priors=priors,
    )))
    new_person, new_sigma_a2, new_sigma_b, accepted = update(keys)
    new_tau = jnp.sqrt(jnp.diagonal(new_sigma_b, axis1=-2, axis2=-1))
    np.testing.assert_allclose(
        np.asarray((new_person[:, :, 0] - mu_a) / jnp.sqrt(new_sigma_a2)[:, None]),
        np.broadcast_to(np.asarray(old_a), (len(keys), len(person))),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray((new_person[:, :, 1:] - mu_b) / new_tau[:, None, :]),
        np.broadcast_to(np.asarray(old_b), (len(keys), len(person), 3)),
        atol=1e-12,
    )
    old_corr = sigma_b / old_tau[:, None] / old_tau[None, :]
    new_corr = new_sigma_b / new_tau[:, :, None] / new_tau[:, None, :]
    np.testing.assert_allclose(
        np.asarray(new_corr), np.broadcast_to(np.asarray(old_corr), new_corr.shape),
        atol=1e-12,
    )
    assert np.asarray(accepted).any()
