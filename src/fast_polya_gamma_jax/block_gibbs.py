"""JAX block Gibbs kernel for a hierarchical Bernoulli-logit panel.

This module is a computational harness, not a reproduction of any particular
applied model.  It keeps the large blocks that matter for scalability:

* one intercept and three coefficients per person;
* a respondent-covariate hierarchy for the three coefficients;
* campaign and item intercepts;
* a dense global-control block; and
* a Polya-Gamma update for every observed binary outcome.

All observation arrays are flat.  ``person_index`` must be sorted so the
person sufficient statistics can use the efficient sorted segment reduction.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .sampler import sample_pg1


class BlockGibbsData(NamedTuple):
    """Immutable arrays used by the block Gibbs transition."""

    y: jax.Array
    person_index: jax.Array
    campaign_index: jax.Array
    item_index: jax.Array
    person_design: jax.Array
    controls: jax.Array
    hierarchy_design: jax.Array
    hierarchy_precision_cholesky: jax.Array


class BlockGibbsPriors(NamedTuple):
    """Conjugate priors used by the computational benchmark."""

    delta_variance: jax.Array
    mu_a_variance: jax.Array
    hierarchy_row_precision: jax.Array
    sigma_a_shape: jax.Array
    sigma_a_scale: jax.Array
    sigma_campaign_shape: jax.Array
    sigma_campaign_scale: jax.Array
    sigma_item_shape: jax.Array
    sigma_item_scale: jax.Array
    sigma_b_df: jax.Array
    sigma_b_scale: jax.Array


class BlockGibbsState(NamedTuple):
    """Markov state for the hierarchical panel benchmark."""

    key: jax.Array
    person: jax.Array
    campaign: jax.Array
    item: jax.Array
    delta: jax.Array
    hierarchy_coef: jax.Array
    sigma_b: jax.Array
    mu_a: jax.Array
    sigma_a2: jax.Array
    sigma_campaign2: jax.Array
    sigma_item2: jax.Array


class BlockGibbsMetrics(NamedTuple):
    """Small per-sweep diagnostics retained by ``run_block_gibbs``."""

    all_finite: jax.Array
    log_likelihood_per_observation: jax.Array
    omega_mean: jax.Array
    omega_min: jax.Array
    omega_max: jax.Array
    max_abs_eta: jax.Array
    min_person_cholesky_diagonal: jax.Array
    mu_a: jax.Array
    mu_b: jax.Array
    sigma_a: jax.Array
    tau_b: jax.Array
    sigma_campaign: jax.Array
    sigma_item: jax.Array


def prepare_block_gibbs_data(
    *,
    y,
    person_index,
    campaign_index,
    item_index,
    creative_features,
    controls,
    respondent_features,
    hierarchy_row_precision: float = 1.0,
) -> BlockGibbsData:
    """Validate and convert flat panel arrays for the Gibbs kernel.

    ``creative_features`` must have three columns.  The person block prepends
    an intercept, while the hierarchy prepends an intercept to the respondent
    features.  The hierarchy precision factor is precomputed once because it
    does not change across sweeps.
    """
    y = jnp.asarray(y)
    person_index = jnp.asarray(person_index, dtype=jnp.int32)
    campaign_index = jnp.asarray(campaign_index, dtype=jnp.int32)
    item_index = jnp.asarray(item_index, dtype=jnp.int32)
    creative_features = jnp.asarray(creative_features)
    controls = jnp.asarray(controls)
    respondent_features = jnp.asarray(respondent_features)

    observations = y.shape[0]
    if y.ndim != 1 or observations < 1:
        raise ValueError("y must be a nonempty vector")
    if creative_features.shape != (observations, 3):
        raise ValueError("creative_features must have shape (observations, 3)")
    if controls.ndim != 2 or controls.shape[0] != observations:
        raise ValueError("controls must have one row per observation")
    if respondent_features.ndim != 2 or respondent_features.shape[0] < 1:
        raise ValueError("respondent_features must be a nonempty matrix")
    for name, index in (
        ("person_index", person_index),
        ("campaign_index", campaign_index),
        ("item_index", item_index),
    ):
        if index.shape != (observations,):
            raise ValueError(f"{name} must have one entry per observation")
    person_host = jax.device_get(person_index)
    if int(person_host.min()) < 0 or int(person_host.max()) >= len(respondent_features):
        raise ValueError("person_index is out of range")
    if bool(jnp.any(person_index[1:] < person_index[:-1])):
        raise ValueError("person_index must be sorted")
    if hierarchy_row_precision <= 0.0:
        raise ValueError("hierarchy_row_precision must be positive")

    dtype = jnp.result_type(
        y.dtype,
        creative_features.dtype,
        controls.dtype,
        respondent_features.dtype,
        float,
    )
    y = y.astype(dtype)
    creative_features = creative_features.astype(dtype)
    controls = controls.astype(dtype)
    respondent_features = respondent_features.astype(dtype)
    person_design = jnp.concatenate(
        [jnp.ones((observations, 1), dtype=dtype), creative_features], axis=1
    )
    hierarchy_design = jnp.concatenate(
        [
            jnp.ones((len(respondent_features), 1), dtype=dtype),
            respondent_features,
        ],
        axis=1,
    )
    hierarchy_precision = (
        hierarchy_design.T @ hierarchy_design
        + hierarchy_row_precision * jnp.eye(hierarchy_design.shape[1], dtype=dtype)
    )
    hierarchy_precision_cholesky = jnp.linalg.cholesky(hierarchy_precision)
    return BlockGibbsData(
        y=y,
        person_index=person_index,
        campaign_index=campaign_index,
        item_index=item_index,
        person_design=person_design,
        controls=controls,
        hierarchy_design=hierarchy_design,
        hierarchy_precision_cholesky=hierarchy_precision_cholesky,
    )


def default_block_gibbs_priors(
    *, dtype=jnp.float64, hierarchy_row_precision: float = 1.0
) -> BlockGibbsPriors:
    """Return weakly regularizing conjugate priors for the benchmark."""
    dtype = jnp.dtype(dtype)

    def scalar(value):
        return jnp.asarray(value, dtype=dtype)

    return BlockGibbsPriors(
        delta_variance=scalar(0.25),
        mu_a_variance=scalar(2.25),
        hierarchy_row_precision=scalar(hierarchy_row_precision),
        sigma_a_shape=scalar(2.5),
        sigma_a_scale=scalar(0.125),
        sigma_campaign_shape=scalar(2.5),
        sigma_campaign_scale=scalar(0.25),
        sigma_item_shape=scalar(2.5),
        sigma_item_scale=scalar(0.25),
        sigma_b_df=scalar(6.0),
        sigma_b_scale=scalar(0.5) * jnp.eye(3, dtype=dtype),
    )


def initialize_block_gibbs_state(
    key: jax.Array,
    *,
    people: int,
    campaigns: int,
    items: int,
    controls: int,
    hierarchy_columns: int,
    dtype=jnp.float64,
) -> BlockGibbsState:
    """Create a finite, neutral initial state."""
    if min(people, campaigns, items, controls, hierarchy_columns) < 1:
        raise ValueError("all block dimensions must be positive")
    dtype = jnp.dtype(dtype)
    return BlockGibbsState(
        key=key,
        person=jnp.zeros((people, 4), dtype=dtype),
        campaign=jnp.zeros(campaigns, dtype=dtype),
        item=jnp.zeros(items, dtype=dtype),
        delta=jnp.zeros(controls, dtype=dtype),
        hierarchy_coef=jnp.zeros((hierarchy_columns, 3), dtype=dtype),
        sigma_b=jnp.eye(3, dtype=dtype) * jnp.asarray(0.25, dtype=dtype),
        mu_a=jnp.asarray(0.0, dtype=dtype),
        sigma_a2=jnp.asarray(0.25, dtype=dtype),
        sigma_campaign2=jnp.asarray(0.25, dtype=dtype),
        sigma_item2=jnp.asarray(0.25, dtype=dtype),
    )


def linear_predictor(data: BlockGibbsData, state: BlockGibbsState):
    """Evaluate the current observation-level linear predictor."""
    person_part = jnp.sum(data.person_design * state.person[data.person_index], axis=1)
    return (
        person_part
        + state.campaign[data.campaign_index]
        + state.item[data.item_index]
        + data.controls @ state.delta
    )


def _sample_precision_normal(key, precision, information):
    cholesky = jnp.linalg.cholesky(precision)
    rhs = jsp.linalg.solve_triangular(cholesky, information[..., None], lower=True)
    mean = jsp.linalg.solve_triangular(cholesky, rhs, trans="T", lower=True)[..., 0]
    standard_normal = jax.random.normal(key, information.shape, information.dtype)
    noise = jsp.linalg.solve_triangular(
        cholesky,
        standard_normal[..., None],
        trans="T",
        lower=True,
    )[..., 0]
    return mean + noise, jnp.min(jnp.diagonal(cholesky, axis1=-2, axis2=-1))


def _sample_inverse_gamma(key, shape, scale):
    return scale / jax.random.gamma(key, shape)


def _sample_inverse_wishart(key, degrees_of_freedom, scale):
    dimension = scale.shape[0]
    diagonal_rows = jnp.arange(dimension)
    diagonal_df = degrees_of_freedom - diagonal_rows
    key_diagonal, key_lower = jax.random.split(key)
    diagonal = jnp.sqrt(jax.random.chisquare(key_diagonal, diagonal_df))
    lower_rows, lower_columns = jnp.tril_indices(dimension, -1)
    lower = jax.random.normal(
        key_lower, (dimension * (dimension - 1) // 2,), dtype=scale.dtype
    )
    bartlett = jnp.zeros_like(scale)
    bartlett = bartlett.at[diagonal_rows, diagonal_rows].set(diagonal)
    bartlett = bartlett.at[lower_rows, lower_columns].set(lower)
    inverse_scale = jnp.linalg.solve(scale, jnp.eye(dimension, dtype=scale.dtype))
    scale_cholesky = jnp.linalg.cholesky(inverse_scale)
    factor = scale_cholesky @ bartlett
    wishart = factor @ factor.T
    covariance = jnp.linalg.solve(wishart, jnp.eye(dimension, dtype=scale.dtype))
    return 0.5 * (covariance + covariance.T)


def _all_state_finite(state: BlockGibbsState):
    leaves = jax.tree.leaves(state)[1:]
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _standardize_person_effects(
    person,
    mu_a,
    hierarchy_coef,
    sigma_a2,
    sigma_b,
    hierarchy_design,
):
    """Map centered person effects to unit-scale residual coordinates."""
    sigma_a = jnp.sqrt(sigma_a2)
    sigma_b_cholesky = jnp.linalg.cholesky(sigma_b)
    z_a = (person[:, 0] - mu_a) / sigma_a
    hierarchy_residual = person[:, 1:] - hierarchy_design @ hierarchy_coef
    z_b = jsp.linalg.solve_triangular(
        sigma_b_cholesky,
        hierarchy_residual.T,
        lower=True,
    ).T
    return z_a, z_b, sigma_a, sigma_b_cholesky


def _interweave_person_location(
    key,
    *,
    person,
    campaign,
    item,
    delta,
    hierarchy_coef,
    sigma_b,
    mu_a,
    sigma_a2,
    omega,
    kappa,
    data: BlockGibbsData,
    priors: BlockGibbsPriors,
):
    """Redraw the four global person locations in noncentered coordinates.

    The centered sweep remains responsible for the conjugate scale updates.
    This ancillary-sufficiency interweaving step holds the standardized person
    residuals fixed and jointly updates ``mu_a`` and the intercept row of the
    three-coefficient hierarchy.
    """
    dtype = person.dtype
    z_a, z_b, sigma_a, sigma_b_cholesky = _standardize_person_effects(
        person,
        mu_a,
        hierarchy_coef,
        sigma_a2,
        sigma_b,
        data.hierarchy_design,
    )

    hierarchy_without_intercept = (
        data.hierarchy_design[:, 1:] @ hierarchy_coef[1:]
    )
    person_without_location = jnp.concatenate(
        [
            (sigma_a * z_a)[:, None],
            hierarchy_without_intercept + z_b @ sigma_b_cholesky.T,
        ],
        axis=1,
    )
    observation_offset = (
        jnp.sum(
            data.person_design * person_without_location[data.person_index],
            axis=1,
        )
        + campaign[data.campaign_index]
        + item[data.item_index]
        + data.controls @ delta
    )

    sigma_b_inverse = jnp.linalg.solve(sigma_b, jnp.eye(3, dtype=dtype))
    location_prior_precision = jnp.zeros((4, 4), dtype=dtype)
    location_prior_precision = location_prior_precision.at[0, 0].set(
        1.0 / priors.mu_a_variance
    )
    location_prior_precision = location_prior_precision.at[1:, 1:].set(
        priors.hierarchy_row_precision * sigma_b_inverse
    )
    location_precision = location_prior_precision + data.person_design.T @ (
        omega[:, None] * data.person_design
    )
    location_information = data.person_design.T @ (
        kappa - omega * observation_offset
    )
    location, _ = _sample_precision_normal(
        key,
        location_precision,
        location_information,
    )

    mu_a = location[0]
    hierarchy_coef = hierarchy_coef.at[0].set(location[1:])
    person = person_without_location + location[None, :]
    return person, mu_a, hierarchy_coef


def block_gibbs_sweep(
    state: BlockGibbsState,
    data: BlockGibbsData,
    priors: BlockGibbsPriors,
    *,
    num_terms: int = 16,
    tail_correction: bool = True,
    interweave_person_location: bool = False,
) -> tuple[BlockGibbsState, BlockGibbsMetrics]:
    """Run one systematic-scan block Gibbs transition."""
    keys = jax.random.split(state.key, 13 if interweave_person_location else 12)
    next_key = keys[0]
    people = state.person.shape[0]
    campaigns = state.campaign.shape[0]
    items = state.item.shape[0]
    coefficient_columns = state.hierarchy_coef.shape[0]
    dtype = state.person.dtype

    eta = linear_predictor(data, state)
    omega = sample_pg1(
        keys[1],
        eta,
        num_terms=num_terms,
        tail_correction=tail_correction,
    )
    kappa = data.y - 0.5

    sigma_b_inverse = jnp.linalg.solve(state.sigma_b, jnp.eye(3, dtype=dtype))
    prior_precision = jnp.zeros((4, 4), dtype=dtype)
    prior_precision = prior_precision.at[0, 0].set(1.0 / state.sigma_a2)
    prior_precision = prior_precision.at[1:, 1:].set(sigma_b_inverse)
    prior_b_mean = data.hierarchy_design @ state.hierarchy_coef
    prior_information = jnp.concatenate(
        [
            jnp.full((people, 1), state.mu_a / state.sigma_a2, dtype=dtype),
            prior_b_mean @ sigma_b_inverse,
        ],
        axis=1,
    )
    offset = (
        state.campaign[data.campaign_index]
        + state.item[data.item_index]
        + data.controls @ state.delta
    )
    weighted_outer = (
        omega[:, None, None]
        * data.person_design[:, :, None]
        * data.person_design[:, None, :]
    )
    person_precision = prior_precision + jax.ops.segment_sum(
        weighted_outer,
        data.person_index,
        num_segments=people,
        indices_are_sorted=True,
    )
    person_information = prior_information + jax.ops.segment_sum(
        data.person_design * (kappa - omega * offset)[:, None],
        data.person_index,
        num_segments=people,
        indices_are_sorted=True,
    )
    person, min_person_cholesky = _sample_precision_normal(
        keys[2], person_precision, person_information
    )

    person_part = jnp.sum(data.person_design * person[data.person_index], axis=1)
    offset_no_campaign = (
        person_part + state.item[data.item_index] + data.controls @ state.delta
    )
    campaign_precision = 1.0 / state.sigma_campaign2 + jax.ops.segment_sum(
        omega, data.campaign_index, num_segments=campaigns
    )
    campaign_information = jax.ops.segment_sum(
        kappa - omega * offset_no_campaign,
        data.campaign_index,
        num_segments=campaigns,
    )
    campaign = campaign_information / campaign_precision + jax.random.normal(
        keys[3], (campaigns,), dtype=dtype
    ) / jnp.sqrt(campaign_precision)

    offset_no_item = (
        person_part + campaign[data.campaign_index] + data.controls @ state.delta
    )
    item_precision = 1.0 / state.sigma_item2 + jax.ops.segment_sum(
        omega, data.item_index, num_segments=items
    )
    item_information = jax.ops.segment_sum(
        kappa - omega * offset_no_item,
        data.item_index,
        num_segments=items,
    )
    item = item_information / item_precision + jax.random.normal(
        keys[4], (items,), dtype=dtype
    ) / jnp.sqrt(item_precision)

    offset_no_delta = (
        person_part + campaign[data.campaign_index] + item[data.item_index]
    )
    delta_precision = jnp.eye(
        state.delta.shape[0], dtype=dtype
    ) / priors.delta_variance + data.controls.T @ (omega[:, None] * data.controls)
    delta_information = data.controls.T @ (kappa - omega * offset_no_delta)
    delta, _ = _sample_precision_normal(keys[5], delta_precision, delta_information)

    mu_a_precision = 1.0 / priors.mu_a_variance + people / state.sigma_a2
    mu_a_mean = jnp.sum(person[:, 0]) / state.sigma_a2 / mu_a_precision
    mu_a = mu_a_mean + jax.random.normal(keys[6], (), dtype=dtype) / jnp.sqrt(
        mu_a_precision
    )

    hierarchy_cholesky = data.hierarchy_precision_cholesky
    hierarchy_rhs = data.hierarchy_design.T @ person[:, 1:]
    hierarchy_intermediate = jsp.linalg.solve_triangular(
        hierarchy_cholesky, hierarchy_rhs, lower=True
    )
    hierarchy_mean = jsp.linalg.solve_triangular(
        hierarchy_cholesky,
        hierarchy_intermediate,
        trans="T",
        lower=True,
    )
    hierarchy_noise = jsp.linalg.solve_triangular(
        hierarchy_cholesky,
        jax.random.normal(keys[7], (coefficient_columns, 3), dtype=dtype),
        trans="T",
        lower=True,
    )
    hierarchy_coef = (
        hierarchy_mean + hierarchy_noise @ jnp.linalg.cholesky(state.sigma_b).T
    )

    sigma_a2 = _sample_inverse_gamma(
        keys[8],
        priors.sigma_a_shape + 0.5 * people,
        priors.sigma_a_scale + 0.5 * jnp.sum(jnp.square(person[:, 0] - mu_a)),
    )
    sigma_campaign2 = _sample_inverse_gamma(
        keys[9],
        priors.sigma_campaign_shape + 0.5 * campaigns,
        priors.sigma_campaign_scale + 0.5 * jnp.sum(jnp.square(campaign)),
    )
    sigma_item2 = _sample_inverse_gamma(
        keys[10],
        priors.sigma_item_shape + 0.5 * items,
        priors.sigma_item_scale + 0.5 * jnp.sum(jnp.square(item)),
    )

    hierarchy_residual = person[:, 1:] - data.hierarchy_design @ hierarchy_coef
    sigma_b_scale = (
        priors.sigma_b_scale
        + hierarchy_residual.T @ hierarchy_residual
        + priors.hierarchy_row_precision * hierarchy_coef.T @ hierarchy_coef
    )
    sigma_b = _sample_inverse_wishart(
        keys[11],
        priors.sigma_b_df + people + coefficient_columns,
        sigma_b_scale,
    )

    if interweave_person_location:
        person, mu_a, hierarchy_coef = _interweave_person_location(
            keys[12],
            person=person,
            campaign=campaign,
            item=item,
            delta=delta,
            hierarchy_coef=hierarchy_coef,
            sigma_b=sigma_b,
            mu_a=mu_a,
            sigma_a2=sigma_a2,
            omega=omega,
            kappa=kappa,
            data=data,
            priors=priors,
        )

    new_state = BlockGibbsState(
        key=next_key,
        person=person,
        campaign=campaign,
        item=item,
        delta=delta,
        hierarchy_coef=hierarchy_coef,
        sigma_b=sigma_b,
        mu_a=mu_a,
        sigma_a2=sigma_a2,
        sigma_campaign2=sigma_campaign2,
        sigma_item2=sigma_item2,
    )
    new_eta = linear_predictor(data, new_state)
    log_likelihood = jnp.mean(data.y * new_eta - jax.nn.softplus(new_eta))
    metrics = BlockGibbsMetrics(
        all_finite=_all_state_finite(new_state),
        log_likelihood_per_observation=log_likelihood,
        omega_mean=jnp.mean(omega),
        omega_min=jnp.min(omega),
        omega_max=jnp.max(omega),
        max_abs_eta=jnp.max(jnp.abs(new_eta)),
        min_person_cholesky_diagonal=min_person_cholesky,
        mu_a=mu_a,
        mu_b=hierarchy_coef[0],
        sigma_a=jnp.sqrt(sigma_a2),
        tau_b=jnp.sqrt(jnp.diag(sigma_b)),
        sigma_campaign=jnp.sqrt(sigma_campaign2),
        sigma_item=jnp.sqrt(sigma_item2),
    )
    return new_state, metrics


def run_block_gibbs(
    state: BlockGibbsState,
    data: BlockGibbsData,
    priors: BlockGibbsPriors,
    *,
    num_steps: int,
    num_terms: int = 16,
    tail_correction: bool = True,
    interweave_person_location: bool = False,
) -> tuple[BlockGibbsState, BlockGibbsMetrics]:
    """Run ``num_steps`` transitions while retaining only scalar diagnostics."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    def transition(current_state, _):
        return block_gibbs_sweep(
            current_state,
            data,
            priors,
            num_terms=num_terms,
            tail_correction=tail_correction,
            interweave_person_location=interweave_person_location,
        )

    return jax.lax.scan(transition, state, xs=None, length=num_steps)
