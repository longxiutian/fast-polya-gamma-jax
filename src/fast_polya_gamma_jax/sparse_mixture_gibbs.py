"""PG-augmented sparse finite-mixture micro-panel sampler.

The likelihood and non-mixture priors match ``micro_panel_model`` in the NBE
project.  The Gaussian respondent-coefficient prior is replaced by a sparse
finite Gaussian mixture with a common covariance.  Original nonconjugate scale,
LKJ, and regularized-horseshoe priors are retained through MH-within-Gibbs.

This kernel inherits the approximate PG generator from ``sampler.py``.  Its
output must not be described as exact MCMC for the stated logistic model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .block_gibbs import (
    BlockGibbsData,
    _sample_inverse_gamma,
    _sample_inverse_wishart,
    prepare_block_gibbs_data,
)
from .sampler import sample_pg1


class MixtureData(NamedTuple):
    panel: BlockGibbsData
    audience: jax.Array
    audience_gram: jax.Array


class MixturePriors(NamedTuple):
    mu_a_sd: jax.Array
    sigma_a_scale: jax.Array
    mu_b_sd: jax.Array
    component_sd: jax.Array
    dirichlet_concentration: jax.Array
    gamma_global_scale: jax.Array
    gamma_slab_sd: jax.Array
    sigma_b_df: jax.Array
    sigma_b_scale: jax.Array
    lkj_concentration: jax.Array
    delta_sd: jax.Array
    sigma_campaign_scale: jax.Array
    sigma_item_scale: jax.Array


class MixtureState(NamedTuple):
    key: jax.Array
    person: jax.Array
    allocation: jax.Array
    weights: jax.Array
    component_mean: jax.Array
    mu_b: jax.Array
    gamma: jax.Array
    hs_global: jax.Array
    hs_local: jax.Array
    sigma_b: jax.Array
    campaign: jax.Array
    item: jax.Array
    delta: jax.Array
    mu_a: jax.Array
    sigma_a2: jax.Array
    sigma_campaign2: jax.Array
    sigma_item2: jax.Array


class MixtureMetrics(NamedTuple):
    all_finite: jax.Array
    log_likelihood_per_observation: jax.Array
    occupied_components: jax.Array
    component_counts: jax.Array
    component_mean: jax.Array
    weights: jax.Array
    mu_a: jax.Array
    mu_b: jax.Array
    sigma_b_sd: jax.Array
    omega_min: jax.Array
    person_cholesky_min: jax.Array
    sigma_b_accepted: jax.Array
    sigma_a_accepted: jax.Array
    sigma_campaign_accepted: jax.Array
    sigma_item_accepted: jax.Array
    hs_global_acceptance: jax.Array
    hs_local_acceptance: jax.Array


def prepare_mixture_data(
    *,
    y,
    person_index,
    campaign_index,
    item_index,
    creative_features,
    controls,
    respondent_features,
) -> MixtureData:
    """Keep the source model's flat row design and respondent covariates."""
    panel = prepare_block_gibbs_data(
        y=y,
        person_index=person_index,
        campaign_index=campaign_index,
        item_index=item_index,
        creative_features=creative_features,
        controls=controls,
        respondent_features=respondent_features,
    )
    audience = panel.hierarchy_design[:, 1:]
    if audience.shape[1] < 1:
        raise ValueError("At least one respondent covariate is required")
    return MixtureData(panel, audience, audience.T @ audience)


def original_micro_priors(
    *, dtype=jnp.float64, covariates: int, component_sd: float = 1.0,
    dirichlet_concentration: float = 0.1,
) -> MixturePriors:
    """Original micro HB priors plus explicit mixture-only hyperparameters."""
    if covariates < 1 or component_sd <= 0 or dirichlet_concentration <= 0:
        raise ValueError("Invalid mixture or covariate dimensions")
    def cast(value):
        return jnp.asarray(value, dtype=dtype)
    return MixturePriors(
        mu_a_sd=cast(1.5),
        sigma_a_scale=cast(0.25),
        mu_b_sd=cast(1.0),
        component_sd=cast(component_sd),
        dirichlet_concentration=cast(dirichlet_concentration),
        gamma_global_scale=cast(0.1 / jnp.sqrt(covariates)),
        gamma_slab_sd=cast(1.0),
        sigma_b_df=cast(4.0),
        sigma_b_scale=cast(0.5),
        lkj_concentration=cast(2.0),
        delta_sd=cast(0.5),
        sigma_campaign_scale=cast(0.5),
        sigma_item_scale=cast(0.5),
    )


def initialize_mixture_state(
    key: jax.Array,
    *, people: int, campaigns: int, items: int, controls: int,
    covariates: int, components: int, dtype=jnp.float64,
) -> MixtureState:
    if min(people, campaigns, items, controls, covariates) < 1 or components < 2:
        raise ValueError("Invalid model dimensions")
    centers = jnp.zeros((components, 3), dtype=dtype)
    centers = centers.at[:, 0].set(jnp.linspace(-0.8, 0.8, components, dtype=dtype))
    return MixtureState(
        key=key,
        person=jnp.zeros((people, 4), dtype=dtype),
        allocation=jnp.arange(people, dtype=jnp.int32) % components,
        weights=jnp.ones((components,), dtype=dtype) / components,
        component_mean=centers,
        mu_b=jnp.zeros((3,), dtype=dtype),
        gamma=jnp.zeros((3, covariates), dtype=dtype),
        hs_global=jnp.full((3,), 0.1 / jnp.sqrt(covariates), dtype=dtype),
        hs_local=jnp.ones((3, covariates), dtype=dtype),
        sigma_b=0.25 * jnp.eye(3, dtype=dtype),
        campaign=jnp.zeros((campaigns,), dtype=dtype),
        item=jnp.zeros((items,), dtype=dtype),
        delta=jnp.zeros((controls,), dtype=dtype),
        mu_a=jnp.asarray(0.0, dtype=dtype),
        sigma_a2=jnp.asarray(0.0625, dtype=dtype),
        sigma_campaign2=jnp.asarray(0.0625, dtype=dtype),
        sigma_item2=jnp.asarray(0.0625, dtype=dtype),
    )


def linear_predictor(data: MixtureData, state: MixtureState) -> jax.Array:
    panel = data.panel
    return (
        jnp.sum(panel.person_design * state.person[panel.person_index], axis=1)
        + state.campaign[panel.campaign_index]
        + state.item[panel.item_index]
        + panel.controls @ state.delta
    )


def _draw_precision_normal(key, cholesky, information):
    rhs = jsp.linalg.solve_triangular(cholesky, information[..., None], lower=True)
    mean = jsp.linalg.solve_triangular(cholesky, rhs, lower=True, trans="T")[..., 0]
    normal = jax.random.normal(key, information.shape, information.dtype)
    noise = jsp.linalg.solve_triangular(
        cholesky, normal[..., None], lower=True, trans="T"
    )[..., 0]
    return mean + noise


def collapsed_allocation_logits(
    data: MixtureData, state: MixtureState, omega: jax.Array,
    kappa: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrate each 4D person block under the PG quadratic likelihood.

    The shared component covariance makes the conditional precision independent
    of class.  Thus one 4x4 Cholesky per respondent serves every candidate class.
    """
    panel = data.panel
    people = len(state.person)
    dtype = state.person.dtype
    sigma_inv = jnp.linalg.solve(state.sigma_b, jnp.eye(3, dtype=dtype))
    prior_precision = jnp.zeros((4, 4), dtype=dtype)
    prior_precision = prior_precision.at[0, 0].set(1.0 / state.sigma_a2)
    prior_precision = prior_precision.at[1:, 1:].set(sigma_inv)
    offset = (
        state.campaign[panel.campaign_index]
        + state.item[panel.item_index]
        + panel.controls @ state.delta
    )
    x = panel.person_design
    sufficient_precision = jax.ops.segment_sum(
        omega[:, None, None] * x[:, :, None] * x[:, None, :],
        panel.person_index, num_segments=people, indices_are_sorted=True,
    )
    sufficient_information = jax.ops.segment_sum(
        x * (kappa - omega * offset)[:, None],
        panel.person_index, num_segments=people, indices_are_sorted=True,
    )
    cholesky = jnp.linalg.cholesky(prior_precision + sufficient_precision)
    mean_b = data.audience @ state.gamma.T
    candidate = jnp.concatenate(
        [
            jnp.broadcast_to(state.mu_a, (people, len(state.weights), 1)),
            mean_b[:, None, :] + state.component_mean[None, :, :],
        ],
        axis=-1,
    )
    prior_info = jnp.einsum("ij,nkj->nki", prior_precision, candidate)
    information = sufficient_information[:, None, :] + prior_info
    solved = jsp.linalg.solve_triangular(
        cholesky, jnp.swapaxes(information, 1, 2), lower=True
    )
    log_weights = (
        jnp.log(state.weights)[None, :]
        - 0.5 * jnp.einsum("nki,nki->nk", candidate, prior_info)
        + 0.5 * jnp.sum(jnp.square(solved), axis=1)
    )
    return log_weights, cholesky, sufficient_information, prior_precision


def _update_sparse_weights(key, allocation, components, concentration, dtype):
    counts = jnp.bincount(allocation, length=components)
    gamma = jax.random.gamma(
        key, counts.astype(dtype) + concentration, dtype=dtype
    )
    return gamma / jnp.sum(gamma), counts


def _update_component_means(key, state, data, prior, allocation, person, sigma_inv):
    components = len(state.weights)
    residual = person[:, 1:] - data.audience @ state.gamma.T
    counts = jnp.bincount(allocation, length=components)
    sums = jax.ops.segment_sum(residual, allocation, num_segments=components)
    prior_precision = jnp.eye(3, dtype=person.dtype) / jnp.square(prior.component_sd)
    precision = counts[:, None, None] * sigma_inv + prior_precision
    information = sums @ sigma_inv + state.mu_b / jnp.square(prior.component_sd)
    means = _draw_precision_normal(key, jnp.linalg.cholesky(precision), information)
    return means


def _update_gamma(key, state, data, person, allocation, sigma_inv, prior):
    residual = person[:, 1:] - state.component_mean[allocation]
    scale = state.hs_global[:, None] * state.hs_local
    prior_inverse_variance = jnp.reciprocal(jnp.square(scale)) + jnp.reciprocal(
        jnp.square(prior.gamma_slab_sd)
    )
    precision = jnp.kron(sigma_inv, data.audience_gram)
    precision = precision + jnp.diag(prior_inverse_variance.reshape(-1))
    information = (sigma_inv @ (residual.T @ data.audience)).reshape(-1)
    draw = _draw_precision_normal(key, jnp.linalg.cholesky(precision), information)
    return draw.reshape(state.gamma.shape)


def _horseshoe_normal_log_density(gamma, global_scale, local_scale, slab_sd):
    product = global_scale[:, None] * local_scale
    variance = jnp.square(product) / (
        1.0 + jnp.square(product / slab_sd)
    )
    return -0.5 * (jnp.log(variance) + jnp.square(gamma) / variance)


def _update_horseshoe(key, state, gamma, priors):
    local_key, local_accept_key, global_key, global_accept_key = jax.random.split(
        key, 4
    )
    log_local = jnp.log(state.hs_local)
    proposed_log_local = log_local + 0.25 * jax.random.normal(
        local_key, log_local.shape, log_local.dtype
    )
    proposed_local = jnp.exp(proposed_log_local)
    current_normal = _horseshoe_normal_log_density(
        gamma, state.hs_global, state.hs_local, priors.gamma_slab_sd
    )
    proposed_normal = _horseshoe_normal_log_density(
        gamma, state.hs_global, proposed_local, priors.gamma_slab_sd
    )
    local_log_ratio = (
        proposed_normal - current_normal
        - jnp.log1p(jnp.square(proposed_local))
        + jnp.log1p(jnp.square(state.hs_local))
        + proposed_log_local - log_local
    )
    local_accept = jnp.log(
        jax.random.uniform(local_accept_key, log_local.shape, dtype=log_local.dtype)
    ) < local_log_ratio
    local = jnp.where(local_accept, proposed_local, state.hs_local)

    log_global = jnp.log(state.hs_global)
    proposed_log_global = log_global + 0.12 * jax.random.normal(
        global_key, log_global.shape, log_global.dtype
    )
    proposed_global = jnp.exp(proposed_log_global)
    current_normal = _horseshoe_normal_log_density(
        gamma, state.hs_global, local, priors.gamma_slab_sd
    ).sum(axis=1)
    proposed_normal = _horseshoe_normal_log_density(
        gamma, proposed_global, local, priors.gamma_slab_sd
    ).sum(axis=1)
    global_log_ratio = (
        proposed_normal - current_normal
        - jnp.log1p(jnp.square(proposed_global / priors.gamma_global_scale))
        + jnp.log1p(jnp.square(state.hs_global / priors.gamma_global_scale))
        + proposed_log_global - log_global
    )
    global_accept = jnp.log(
        jax.random.uniform(global_accept_key, log_global.shape, dtype=log_global.dtype)
    ) < global_log_ratio
    global_scale = jnp.where(global_accept, proposed_global, state.hs_global)
    return local, global_scale, jnp.mean(local_accept), jnp.mean(global_accept)


def _half_normal_variance_correction(variance, scale, reference_shape, reference_scale):
    # Target half-normal density on sigma, transformed to sigma squared,
    # divided by the reference inverse-gamma prior density.
    return (
        -variance / (2.0 * jnp.square(scale))
        + (reference_shape + 0.5) * jnp.log(variance)
        + reference_scale / variance
    )


def _update_half_normal_variance(key, current, residual_squares, count, scale):
    proposal_key, acceptance_key = jax.random.split(key)
    reference_shape = 2.5
    reference_scale = jnp.square(scale)
    proposed = _sample_inverse_gamma(
        proposal_key,
        reference_shape + 0.5 * count,
        reference_scale + 0.5 * residual_squares,
    )
    log_ratio = _half_normal_variance_correction(
        proposed, scale, reference_shape, reference_scale
    ) - _half_normal_variance_correction(
        current, scale, reference_shape, reference_scale
    )
    accepted = jnp.log(
        jax.random.uniform(acceptance_key, (), dtype=current.dtype)
    ) < log_ratio
    return jnp.where(accepted, proposed, current), accepted


def _covariance_target_prior(sigma, priors):
    tau = jnp.sqrt(jnp.diag(sigma))
    correlation = sigma / (tau[:, None] * tau[None, :])
    _, log_det_correlation = jnp.linalg.slogdet(correlation)
    half_student = -0.5 * (priors.sigma_b_df + 1.0) * jnp.sum(
        jnp.log1p(jnp.square(tau / priors.sigma_b_scale) / priors.sigma_b_df)
    )
    dimension = len(tau)
    return (
        half_student
        + (priors.lkj_concentration - 1.0) * log_det_correlation
        - dimension * jnp.sum(jnp.log(tau))
    )


def _covariance_reference_prior(sigma, df, scale):
    dimension = sigma.shape[0]
    _, log_det = jnp.linalg.slogdet(sigma)
    return -0.5 * (df + dimension + 1.0) * log_det - 0.5 * jnp.trace(
        jnp.linalg.solve(sigma, scale)
    )


def _update_covariance(key, current, residual, priors):
    proposal_key, acceptance_key = jax.random.split(key)
    reference_df = 5.0
    reference_scale = 0.25 * jnp.eye(3, dtype=current.dtype)
    scatter = residual.T @ residual
    proposed = _sample_inverse_wishart(
        proposal_key, reference_df + len(residual), reference_scale + scatter
    )
    current_correction = _covariance_target_prior(
        current, priors
    ) - _covariance_reference_prior(current, reference_df, reference_scale)
    proposed_correction = _covariance_target_prior(
        proposed, priors
    ) - _covariance_reference_prior(proposed, reference_df, reference_scale)
    log_ratio = proposed_correction - current_correction
    accepted = jnp.log(
        jax.random.uniform(acceptance_key, (), dtype=current.dtype)
    ) < log_ratio
    return jnp.where(accepted, proposed, current), accepted


def _interweave_mu_a(key, person, mu_a, state, data, omega, kappa, priors, *,
                     campaign, item, delta):
    panel = data.panel
    centered_a = person[:, 0] - mu_a
    offset = (
        centered_a[panel.person_index]
        + jnp.sum(panel.person_design[:, 1:] * person[panel.person_index, 1:], axis=1)
        + campaign[panel.campaign_index]
        + item[panel.item_index]
        + panel.controls @ delta
    )
    precision = jnp.reciprocal(jnp.square(priors.mu_a_sd)) + jnp.sum(omega)
    information = jnp.sum(kappa - omega * offset)
    new_mu_a = information / precision + jax.random.normal(
        key, (), dtype=person.dtype
    ) / jnp.sqrt(precision)
    return person.at[:, 0].set(centered_a + new_mu_a), new_mu_a


def sparse_mixture_sweep(
    state: MixtureState,
    data: MixtureData,
    priors: MixturePriors,
    *, num_terms: int = 8, interweave_mu_a: bool = True,
) -> tuple[MixtureState, MixtureMetrics]:
    """One PG, collapsed-label, Gaussian-block, MH-hyperparameter sweep."""
    keys = jax.random.split(state.key, 18)
    panel = data.panel
    dtype = state.person.dtype
    people = len(state.person)
    eta = linear_predictor(data, state)
    omega = sample_pg1(keys[1], eta, num_terms=num_terms, tail_correction=True)
    kappa = panel.y - 0.5

    logits, person_cholesky, person_info, prior_precision = (
        collapsed_allocation_logits(data, state, omega, kappa)
    )
    allocation = jax.random.categorical(keys[2], logits, axis=-1).astype(jnp.int32)
    prior_mean_b = data.audience @ state.gamma.T + state.component_mean[allocation]
    prior_mean = jnp.concatenate(
        [jnp.full((people, 1), state.mu_a, dtype=dtype), prior_mean_b], axis=1
    )
    person = _draw_precision_normal(
        keys[3], person_cholesky,
        person_info + prior_mean @ prior_precision.T,
    )
    person_part = jnp.sum(panel.person_design * person[panel.person_index], axis=1)

    offset_no_campaign = (
        person_part + state.item[panel.item_index] + panel.controls @ state.delta
    )
    campaign_precision = jnp.reciprocal(state.sigma_campaign2) + jax.ops.segment_sum(
        omega, panel.campaign_index, num_segments=len(state.campaign)
    )
    campaign_information = jax.ops.segment_sum(
        kappa - omega * offset_no_campaign,
        panel.campaign_index, num_segments=len(state.campaign),
    )
    campaign = campaign_information / campaign_precision + jax.random.normal(
        keys[4], state.campaign.shape, dtype=dtype
    ) / jnp.sqrt(campaign_precision)

    offset_no_item = person_part + campaign[panel.campaign_index] + (
        panel.controls @ state.delta
    )
    item_precision = jnp.reciprocal(state.sigma_item2) + jax.ops.segment_sum(
        omega, panel.item_index, num_segments=len(state.item)
    )
    item_information = jax.ops.segment_sum(
        kappa - omega * offset_no_item,
        panel.item_index, num_segments=len(state.item),
    )
    item = item_information / item_precision + jax.random.normal(
        keys[5], state.item.shape, dtype=dtype
    ) / jnp.sqrt(item_precision)

    offset_no_delta = (
        person_part + campaign[panel.campaign_index] + item[panel.item_index]
    )
    delta_precision = (
        jnp.eye(len(state.delta), dtype=dtype) / jnp.square(priors.delta_sd)
        + panel.controls.T @ (omega[:, None] * panel.controls)
    )
    delta_information = panel.controls.T @ (
        kappa - omega * offset_no_delta
    )
    delta = _draw_precision_normal(
        keys[6], jnp.linalg.cholesky(delta_precision), delta_information
    )

    mu_a_precision = jnp.reciprocal(jnp.square(priors.mu_a_sd)) + (
        people / state.sigma_a2
    )
    mu_a = (
        jnp.sum(person[:, 0]) / state.sigma_a2 / mu_a_precision
        + jax.random.normal(keys[7], (), dtype=dtype) / jnp.sqrt(mu_a_precision)
    )
    sigma_inv = jnp.linalg.solve(state.sigma_b, jnp.eye(3, dtype=dtype))
    component_mean = _update_component_means(
        keys[8], state, data, priors, allocation, person, sigma_inv
    )
    mu_b_precision = jnp.reciprocal(jnp.square(priors.mu_b_sd)) + (
        len(state.weights) / jnp.square(priors.component_sd)
    )
    mu_b = (
        jnp.sum(component_mean, axis=0) / jnp.square(priors.component_sd)
        / mu_b_precision
        + jax.random.normal(keys[9], (3,), dtype=dtype) / jnp.sqrt(mu_b_precision)
    )
    component_state = state._replace(component_mean=component_mean)
    gamma = _update_gamma(
        keys[10], component_state, data, person, allocation, sigma_inv, priors
    )
    hs_local, hs_global, hs_local_accept, hs_global_accept = _update_horseshoe(
        keys[11], state, gamma, priors
    )

    sigma_a2, sigma_a_accept = _update_half_normal_variance(
        keys[12], state.sigma_a2,
        jnp.sum(jnp.square(person[:, 0] - mu_a)), people, priors.sigma_a_scale,
    )
    sigma_campaign2, sigma_campaign_accept = _update_half_normal_variance(
        keys[13], state.sigma_campaign2, jnp.sum(jnp.square(campaign)),
        len(campaign), priors.sigma_campaign_scale,
    )
    sigma_item2, sigma_item_accept = _update_half_normal_variance(
        keys[14], state.sigma_item2, jnp.sum(jnp.square(item)),
        len(item), priors.sigma_item_scale,
    )
    b_residual = (
        person[:, 1:] - component_mean[allocation] - data.audience @ gamma.T
    )
    sigma_b, sigma_b_accept = _update_covariance(
        keys[15], state.sigma_b, b_residual, priors
    )
    weights, counts = _update_sparse_weights(
        keys[16], allocation, len(state.weights),
        priors.dirichlet_concentration, dtype,
    )
    if interweave_mu_a:
        person, mu_a = _interweave_mu_a(
            keys[17], person, mu_a, state, data, omega, kappa, priors,
            campaign=campaign, item=item, delta=delta,
        )

    updated = MixtureState(
        key=keys[0], person=person, allocation=allocation,
        weights=weights, component_mean=component_mean, mu_b=mu_b,
        gamma=gamma, hs_global=hs_global, hs_local=hs_local,
        sigma_b=sigma_b, campaign=campaign, item=item, delta=delta,
        mu_a=mu_a, sigma_a2=sigma_a2, sigma_campaign2=sigma_campaign2,
        sigma_item2=sigma_item2,
    )
    final_eta = linear_predictor(data, updated)
    leaves = jax.tree.leaves(updated)[1:]
    all_finite = jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(leaf)) for leaf in leaves
    ]))
    metrics = MixtureMetrics(
        all_finite=all_finite,
        log_likelihood_per_observation=jnp.mean(
            panel.y * final_eta - jax.nn.softplus(final_eta)
        ),
        occupied_components=jnp.sum(counts > 0),
        component_counts=counts,
        component_mean=component_mean,
        weights=weights,
        mu_a=mu_a, mu_b=mu_b,
        sigma_b_sd=jnp.sqrt(jnp.diag(sigma_b)),
        omega_min=jnp.min(omega),
        person_cholesky_min=jnp.min(
            jnp.diagonal(person_cholesky, axis1=-2, axis2=-1)
        ),
        sigma_b_accepted=sigma_b_accept,
        sigma_a_accepted=sigma_a_accept,
        sigma_campaign_accepted=sigma_campaign_accept,
        sigma_item_accepted=sigma_item_accept,
        hs_global_acceptance=hs_global_accept,
        hs_local_acceptance=hs_local_accept,
    )
    return updated, metrics


def run_sparse_mixture(
    state: MixtureState, data: MixtureData, priors: MixturePriors, *,
    num_steps: int, num_terms: int = 8, interweave_mu_a: bool = True,
) -> tuple[MixtureState, MixtureMetrics]:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    def step(current, _):
        return sparse_mixture_sweep(
            current, data, priors, num_terms=num_terms,
            interweave_mu_a=interweave_mu_a,
        )

    return jax.lax.scan(step, state, xs=None, length=num_steps)


def run_sparse_mixture_collect(
    state: MixtureState, data: MixtureData, priors: MixturePriors, *,
    num_steps: int, archive_stride: int = 10, num_terms: int = 8,
) -> tuple[MixtureState, MixtureMetrics, dict, jax.Array, jax.Array, jax.Array]:
    """Keep global traces and bounded person reductions for one draw chunk."""
    if num_steps < 1 or archive_stride < 1 or num_steps % archive_stride:
        raise ValueError("Collector chunk must align with archive stride")
    people = len(state.person)
    dtype = state.person.dtype
    initial = (
        state,
        jnp.zeros((people, 4), dtype=dtype),
        jnp.zeros((people, 4), dtype=dtype),
        jnp.zeros((num_steps // archive_stride, people, 4), dtype=jnp.float32),
    )

    def step(carry, index):
        current, person_sum, person_square_sum, archive = carry
        current, metrics = sparse_mixture_sweep(
            current, data, priors, num_terms=num_terms
        )
        person_sum = person_sum + current.person
        person_square_sum = person_square_sum + jnp.square(current.person)
        archive = jax.lax.cond(
            (index + 1) % archive_stride == 0,
            lambda values: values.at[index // archive_stride].set(
                current.person.astype(jnp.float32)
            ),
            lambda values: values,
            archive,
        )
        global_draw = {
            "mu_a": current.mu_a,
            "sigma_a": jnp.sqrt(current.sigma_a2),
            "mu_B": current.mu_b,
            "component_mean": current.component_mean,
            "weights": current.weights,
            "Gamma": current.gamma,
            "Sigma_B": current.sigma_b,
            "delta": current.delta,
            "sigma_campaign": jnp.sqrt(current.sigma_campaign2),
            "sigma_ad": jnp.sqrt(current.sigma_item2),
            "campaign_effect": current.campaign,
            "ad_effect": current.item,
        }
        return (current, person_sum, person_square_sum, archive), (
            metrics, global_draw
        )

    (final_state, person_sum, person_square_sum, archive), (
        metrics, global_trace
    ) = jax.lax.scan(step, initial, jnp.arange(num_steps))
    return (
        final_state, metrics, global_trace, person_sum, person_square_sum, archive
    )
