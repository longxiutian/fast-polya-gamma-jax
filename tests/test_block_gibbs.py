from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from fast_polya_gamma_jax import (
    default_block_gibbs_priors,
    initialize_block_gibbs_state,
    prepare_block_gibbs_data,
    run_block_gibbs,
)

jax.config.update("jax_enable_x64", True)


def _small_problem():
    rng = np.random.default_rng(7)
    people = 24
    observations_per_person = 3
    observations = people * observations_per_person
    person_index = np.repeat(np.arange(people, dtype=np.int32), 3)
    creative = rng.binomial(1, 0.5, size=(observations, 3))
    controls = rng.normal(size=(observations, 4))
    respondent = rng.normal(size=(people, 5))
    campaign_index = rng.integers(0, 4, size=observations, dtype=np.int32)
    item_index = rng.integers(0, 8, size=observations, dtype=np.int32)
    y = rng.binomial(1, 0.4, size=observations)
    data = prepare_block_gibbs_data(
        y=y,
        person_index=person_index,
        campaign_index=campaign_index,
        item_index=item_index,
        creative_features=creative,
        controls=controls,
        respondent_features=respondent,
    )
    state = initialize_block_gibbs_state(
        jax.random.key(9),
        people=people,
        campaigns=4,
        items=8,
        controls=4,
        hierarchy_columns=6,
    )
    return data, state, default_block_gibbs_priors()


def test_block_gibbs_jits_and_stays_finite():
    data, state, priors = _small_problem()
    runner = jax.jit(partial(run_block_gibbs, num_steps=4, num_terms=8))
    final_state, metrics = runner(state, data, priors)
    assert final_state.person.shape == (24, 4)
    assert metrics.mu_b.shape == (4, 3)
    assert bool(jnp.all(metrics.all_finite))
    assert bool(jnp.all(metrics.omega_min > 0.0))
    assert bool(jnp.all(metrics.min_person_cholesky_diagonal > 0.0))


def test_block_gibbs_is_reproducible_from_same_state():
    data, state, priors = _small_problem()
    run = partial(run_block_gibbs, num_steps=2, num_terms=8)
    first_state, first_metrics = run(state, data, priors)
    second_state, second_metrics = run(state, data, priors)
    np.testing.assert_array_equal(first_state.person, second_state.person)
    np.testing.assert_array_equal(first_state.sigma_b, second_state.sigma_b)
    np.testing.assert_array_equal(first_metrics.mu_a, second_metrics.mu_a)


def test_prepare_rejects_unsorted_people():
    data, _, _ = _small_problem()
    reversed_people = np.asarray(data.person_index)[::-1]
    try:
        prepare_block_gibbs_data(
            y=data.y,
            person_index=reversed_people,
            campaign_index=data.campaign_index,
            item_index=data.item_index,
            creative_features=data.person_design[:, 1:],
            controls=data.controls,
            respondent_features=data.hierarchy_design[:, 1:],
        )
    except ValueError as exc:
        assert "sorted" in str(exc)
    else:
        raise AssertionError("unsorted person_index was accepted")
