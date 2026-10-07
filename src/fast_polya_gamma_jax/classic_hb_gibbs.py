"""Approximate PG block Gibbs sampler for the non-mixture micro-panel HB.

The likelihood and priors match ``micro_panel_model`` in mengyao-nbe. The
fixed-cost PG draw is approximate, so these are not exact logistic-posterior
draws. Hyperparameters with nonconjugate priors use MH corrections.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .block_gibbs import BlockGibbsData, prepare_block_gibbs_data
from .sampler import sample_pg1
from .sparse_mixture_gibbs import (
    _draw_precision_normal,
    _update_covariance,
    _update_half_normal_variance,
    _update_horseshoe,
)


class ClassicData(NamedTuple):
    panel: BlockGibbsData
    audience: jax.Array
    audience_gram: jax.Array


class ClassicPriors(NamedTuple):
    mu_a_sd: jax.Array
    sigma_a_scale: jax.Array
    mu_b_sd: jax.Array
    gamma_global_scale: jax.Array
    gamma_slab_sd: jax.Array
    sigma_b_df: jax.Array
    sigma_b_scale: jax.Array
    lkj_concentration: jax.Array
    delta_sd: jax.Array
    sigma_campaign_scale: jax.Array
    sigma_item_scale: jax.Array


class ClassicState(NamedTuple):
    key: jax.Array
    person: jax.Array
    mu_a: jax.Array
    sigma_a2: jax.Array
    mu_b: jax.Array
    gamma: jax.Array
    hs_global: jax.Array
    hs_local: jax.Array
    sigma_b: jax.Array
    campaign: jax.Array
    item: jax.Array
    delta: jax.Array
    sigma_campaign2: jax.Array
    sigma_item2: jax.Array


class ClassicMetrics(NamedTuple):
    all_finite: jax.Array
    log_likelihood_per_observation: jax.Array
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
    off_centered_scale_acceptance: jax.Array


def prepare_classic_data(
    *, y, person_index, campaign_index, item_index,
    creative_features, controls, respondent_features,
) -> ClassicData:
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
        raise ValueError("At least one audience covariate is required")
    return ClassicData(panel, audience, audience.T @ audience)


def original_classic_priors(*, covariates: int, dtype=jnp.float64) -> ClassicPriors:
    if covariates < 1:
        raise ValueError("covariates must be positive")

    def cast(value):
        return jnp.asarray(value, dtype=dtype)

    return ClassicPriors(
        mu_a_sd=cast(1.5), sigma_a_scale=cast(0.25),
        mu_b_sd=cast(1.0),
        gamma_global_scale=cast(0.1 / jnp.sqrt(covariates)),
        gamma_slab_sd=cast(1.0), sigma_b_df=cast(4.0),
        sigma_b_scale=cast(0.5), lkj_concentration=cast(2.0),
        delta_sd=cast(0.5), sigma_campaign_scale=cast(0.5),
        sigma_item_scale=cast(0.5),
    )


def initialize_classic_state(
    key: jax.Array, *, people: int, campaigns: int, items: int,
    controls: int, covariates: int, dtype=jnp.float64,
) -> ClassicState:
    if min(people, campaigns, items, controls, covariates) < 1:
        raise ValueError("All model dimensions must be positive")
    return ClassicState(
        key=key, person=jnp.zeros((people, 4), dtype=dtype),
        mu_a=jnp.asarray(0.0, dtype=dtype),
        sigma_a2=jnp.asarray(0.0625, dtype=dtype),
        mu_b=jnp.zeros((3,), dtype=dtype),
        gamma=jnp.zeros((3, covariates), dtype=dtype),
        hs_global=jnp.full((3,), 0.1 / jnp.sqrt(covariates), dtype=dtype),
        hs_local=jnp.ones((3, covariates), dtype=dtype),
        sigma_b=0.25 * jnp.eye(3, dtype=dtype),
        campaign=jnp.zeros((campaigns,), dtype=dtype),
        item=jnp.zeros((items,), dtype=dtype),
        delta=jnp.zeros((controls,), dtype=dtype),
        sigma_campaign2=jnp.asarray(0.0625, dtype=dtype),
        sigma_item2=jnp.asarray(0.0625, dtype=dtype),
    )


def linear_predictor(data: ClassicData, state: ClassicState) -> jax.Array:
    panel = data.panel
    return (
        jnp.sum(panel.person_design * state.person[panel.person_index], axis=1)
        + state.campaign[panel.campaign_index]
        + state.item[panel.item_index]
        + panel.controls @ state.delta
    )


def _update_gamma(key, state, data, person, mu_b, sigma_inv, priors):
    residual = person[:, 1:] - mu_b
    scale = state.hs_global[:, None] * state.hs_local
    prior_inverse_variance = (
        jnp.reciprocal(jnp.square(scale))
        + jnp.reciprocal(jnp.square(priors.gamma_slab_sd))
    )
    precision = jnp.kron(sigma_inv, data.audience_gram)
    precision += jnp.diag(prior_inverse_variance.reshape(-1))
    information = (sigma_inv @ (residual.T @ data.audience)).reshape(-1)
    draw = _draw_precision_normal(key, jnp.linalg.cholesky(precision), information)
    return draw.reshape(state.gamma.shape)


def _interweave_location(key, *, person, mu_a, mu_b, campaign, item,
                         delta, data, omega, kappa, priors):
    """Update population locations conditional on off-centered person residuals."""
    panel = data.panel
    location = jnp.concatenate([mu_a[None], mu_b])
    residual = person - location[None, :]
    offset = (
        jnp.sum(panel.person_design * residual[panel.person_index], axis=1)
        + campaign[panel.campaign_index] + item[panel.item_index]
        + panel.controls @ delta
    )
    prior_precision = jnp.diag(jnp.concatenate([
        jnp.reciprocal(jnp.square(priors.mu_a_sd))[None],
        jnp.full((3,), jnp.reciprocal(jnp.square(priors.mu_b_sd))),
    ]))
    precision = prior_precision + panel.person_design.T @ (
        omega[:, None] * panel.person_design
    )
    information = panel.person_design.T @ (kappa - omega * offset)
    new_location = _draw_precision_normal(
        key, jnp.linalg.cholesky(precision), information
    )
    return residual + new_location[None, :], new_location[0], new_location[1:]


def _interweave_scales(key, *, person, mu_a, mu_b, gamma, sigma_a2,
                       sigma_b, data, omega, kappa, campaign, item,
                       delta, priors):
    """MH-update four scales with standardized respondent residuals fixed."""
    panel = data.panel
    scales = jnp.concatenate([
        jnp.sqrt(sigma_a2)[None], jnp.sqrt(jnp.diag(sigma_b)),
    ])
    mean_b = mu_b + data.audience @ gamma.T
    residual = jnp.concatenate([
        ((person[:, 0] - mu_a) / scales[0])[:, None],
        (person[:, 1:] - mean_b) / scales[None, 1:],
    ], axis=1)
    basis = panel.person_design * residual[panel.person_index]
    eta = (
        jnp.sum(panel.person_design * person[panel.person_index], axis=1)
        + campaign[panel.campaign_index] + item[panel.item_index]
        + panel.controls @ delta
    )
    gradient = basis.T @ (kappa - omega * eta)
    curvature = basis.T @ (omega[:, None] * basis)
    proposal_keys = jax.random.split(key, 4)
    proposal_sd = jnp.asarray([0.01, 0.01, 0.01, 0.01], dtype=scales.dtype)

    def log_scale_prior(index, value):
        half_normal = -0.5 * jnp.square(value / priors.sigma_a_scale)
        half_student = -0.5 * (priors.sigma_b_df + 1.0) * jnp.log1p(
            jnp.square(value / priors.sigma_b_scale) / priors.sigma_b_df
        )
        return jnp.where(index == 0, half_normal, half_student)

    def step(carry, index):
        current_scales, applied = carry
        normal_key, accept_key = jax.random.split(proposal_keys[index])
        current = current_scales[index]
        proposed = current * jnp.exp(
            proposal_sd[index] * jax.random.normal(
                normal_key, (), dtype=scales.dtype
            )
        )
        change = proposed - current
        conditional_gradient = gradient[index] - curvature[index] @ applied
        log_ratio = (
            change * conditional_gradient
            - 0.5 * jnp.square(change) * curvature[index, index]
            + log_scale_prior(index, proposed)
            - log_scale_prior(index, current)
            + jnp.log(proposed / current)
        )
        accepted = jnp.log(jax.random.uniform(
            accept_key, (), dtype=scales.dtype
        )) < log_ratio
        current_scales = current_scales.at[index].set(
            jnp.where(accepted, proposed, current)
        )
        applied = applied.at[index].set(jnp.where(accepted, change, 0.0))
        return (current_scales, applied), accepted

    (updated_scales, _), accepted = jax.lax.scan(
        step, (scales, jnp.zeros_like(scales)), jnp.arange(4)
    )
    updated_person = jnp.concatenate([
        (mu_a + updated_scales[0] * residual[:, 0])[:, None],
        mean_b + updated_scales[None, 1:] * residual[:, 1:],
    ], axis=1)
    ratios = updated_scales[1:] / scales[1:]
    updated_sigma_b = ratios[:, None] * sigma_b * ratios[None, :]
    return (
        updated_person, jnp.square(updated_scales[0]),
        updated_sigma_b, accepted,
    )


def classic_sweep(
    state: ClassicState, data: ClassicData, priors: ClassicPriors,
    *, num_terms: int = 8, interweave_location: bool = True,
) -> tuple[ClassicState, ClassicMetrics]:
    keys = jax.random.split(state.key, 17)
    panel = data.panel
    dtype = state.person.dtype
    people = len(state.person)
    omega = sample_pg1(
        keys[1], linear_predictor(data, state),
        num_terms=num_terms, tail_correction=True,
    )
    kappa = panel.y - 0.5
    sigma_inv = jnp.linalg.solve(state.sigma_b, jnp.eye(3, dtype=dtype))
    prior_precision = jnp.zeros((4, 4), dtype=dtype)
    prior_precision = prior_precision.at[0, 0].set(1.0 / state.sigma_a2)
    prior_precision = prior_precision.at[1:, 1:].set(sigma_inv)
    prior_b_mean = state.mu_b + data.audience @ state.gamma.T
    prior_information = jnp.concatenate([
        jnp.full((people, 1), state.mu_a / state.sigma_a2, dtype=dtype),
        prior_b_mean @ sigma_inv,
    ], axis=1)
    offset = (
        state.campaign[panel.campaign_index]
        + state.item[panel.item_index] + panel.controls @ state.delta
    )
    x = panel.person_design
    precision = prior_precision + jax.ops.segment_sum(
        omega[:, None, None] * x[:, :, None] * x[:, None, :],
        panel.person_index, num_segments=people, indices_are_sorted=True,
    )
    information = prior_information + jax.ops.segment_sum(
        x * (kappa - omega * offset)[:, None],
        panel.person_index, num_segments=people, indices_are_sorted=True,
    )
    person_cholesky = jnp.linalg.cholesky(precision)
    person = _draw_precision_normal(keys[2], person_cholesky, information)
    person_part = jnp.sum(x * person[panel.person_index], axis=1)

    campaign_precision = jnp.reciprocal(state.sigma_campaign2) + jax.ops.segment_sum(
        omega, panel.campaign_index, num_segments=len(state.campaign),
    )
    campaign_information = jax.ops.segment_sum(
        kappa - omega * (
            person_part + state.item[panel.item_index] + panel.controls @ state.delta
        ), panel.campaign_index, num_segments=len(state.campaign),
    )
    campaign = campaign_information / campaign_precision + jax.random.normal(
        keys[3], state.campaign.shape, dtype=dtype
    ) / jnp.sqrt(campaign_precision)
    item_precision = jnp.reciprocal(state.sigma_item2) + jax.ops.segment_sum(
        omega, panel.item_index, num_segments=len(state.item),
    )
    item_information = jax.ops.segment_sum(
        kappa - omega * (
            person_part + campaign[panel.campaign_index] + panel.controls @ state.delta
        ), panel.item_index, num_segments=len(state.item),
    )
    item = item_information / item_precision + jax.random.normal(
        keys[4], state.item.shape, dtype=dtype
    ) / jnp.sqrt(item_precision)
    delta_precision = (
        jnp.eye(len(state.delta), dtype=dtype) / jnp.square(priors.delta_sd)
        + panel.controls.T @ (omega[:, None] * panel.controls)
    )
    delta_information = panel.controls.T @ (
        kappa - omega * (
            person_part + campaign[panel.campaign_index] + item[panel.item_index]
        )
    )
    delta = _draw_precision_normal(
        keys[5], jnp.linalg.cholesky(delta_precision), delta_information
    )

    mu_a_precision = (
        jnp.reciprocal(jnp.square(priors.mu_a_sd)) + people / state.sigma_a2
    )
    mu_a = (
        jnp.sum(person[:, 0]) / state.sigma_a2 / mu_a_precision
        + jax.random.normal(keys[6], (), dtype=dtype) / jnp.sqrt(mu_a_precision)
    )
    b_residual = person[:, 1:] - data.audience @ state.gamma.T
    mu_b_precision = (
        jnp.eye(3, dtype=dtype) / jnp.square(priors.mu_b_sd) + people * sigma_inv
    )
    mu_b_information = sigma_inv @ jnp.sum(b_residual, axis=0)
    mu_b = _draw_precision_normal(
        keys[7], jnp.linalg.cholesky(mu_b_precision), mu_b_information
    )
    gamma = _update_gamma(keys[8], state, data, person, mu_b, sigma_inv, priors)
    hs_local, hs_global, hs_local_accept, hs_global_accept = _update_horseshoe(
        keys[9], state, gamma, priors
    )
    sigma_a2, sigma_a_accept = _update_half_normal_variance(
        keys[10], state.sigma_a2,
        jnp.sum(jnp.square(person[:, 0] - mu_a)), people, priors.sigma_a_scale,
    )
    sigma_campaign2, sigma_campaign_accept = _update_half_normal_variance(
        keys[11], state.sigma_campaign2, jnp.sum(jnp.square(campaign)),
        len(campaign), priors.sigma_campaign_scale,
    )
    sigma_item2, sigma_item_accept = _update_half_normal_variance(
        keys[12], state.sigma_item2, jnp.sum(jnp.square(item)),
        len(item), priors.sigma_item_scale,
    )
    residual_b = person[:, 1:] - mu_b - data.audience @ gamma.T
    sigma_b, sigma_b_accept = _update_covariance(
        keys[13], state.sigma_b, residual_b, priors
    )
    if interweave_location:
        person, mu_a, mu_b = _interweave_location(
            keys[14], person=person, mu_a=mu_a, mu_b=mu_b,
            campaign=campaign, item=item, delta=delta, data=data,
            omega=omega, kappa=kappa, priors=priors,
        )
    person, sigma_a2, sigma_b, scale_accepted = _interweave_scales(
        keys[15], person=person, mu_a=mu_a, mu_b=mu_b, gamma=gamma,
        sigma_a2=sigma_a2, sigma_b=sigma_b, data=data,
        omega=omega, kappa=kappa, campaign=campaign, item=item,
        delta=delta, priors=priors,
    )
    updated = ClassicState(
        key=keys[0], person=person, mu_a=mu_a, sigma_a2=sigma_a2,
        mu_b=mu_b, gamma=gamma, hs_global=hs_global, hs_local=hs_local,
        sigma_b=sigma_b, campaign=campaign, item=item, delta=delta,
        sigma_campaign2=sigma_campaign2, sigma_item2=sigma_item2,
    )
    final_eta = linear_predictor(data, updated)
    all_finite = jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(updated)[1:]
    ]))
    metrics = ClassicMetrics(
        all_finite=all_finite,
        log_likelihood_per_observation=jnp.mean(
            panel.y * final_eta - jax.nn.softplus(final_eta)
        ), mu_a=mu_a, mu_b=mu_b, sigma_b_sd=jnp.sqrt(jnp.diag(sigma_b)),
        omega_min=jnp.min(omega),
        person_cholesky_min=jnp.min(jnp.diagonal(
            person_cholesky, axis1=-2, axis2=-1,
        )), sigma_b_accepted=sigma_b_accept,
        sigma_a_accepted=sigma_a_accept,
        sigma_campaign_accepted=sigma_campaign_accept,
        sigma_item_accepted=sigma_item_accept,
        hs_global_acceptance=hs_global_accept,
        hs_local_acceptance=hs_local_accept,
        off_centered_scale_acceptance=scale_accepted,
    )
    return updated, metrics


def run_classic(
    state: ClassicState, data: ClassicData, priors: ClassicPriors,
    *, num_steps: int, num_terms: int = 8,
) -> tuple[ClassicState, ClassicMetrics]:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    def step(current, _):
        return classic_sweep(current, data, priors, num_terms=num_terms)

    return jax.lax.scan(step, state, xs=None, length=num_steps)


def run_classic_collect(
    state: ClassicState, data: ClassicData, priors: ClassicPriors,
    *, num_steps: int, archive_stride: int, num_terms: int = 8,
) -> tuple[ClassicState, ClassicMetrics, dict, jax.Array, jax.Array, jax.Array]:
    if num_steps < 1 or archive_stride < 1 or num_steps % archive_stride:
        raise ValueError("Collector chunk must align with archive stride")
    people = len(state.person)
    initial = (
        state,
        jnp.zeros((people, 4), dtype=state.person.dtype),
        jnp.zeros((people, 4), dtype=state.person.dtype),
        jnp.zeros((num_steps // archive_stride, people, 4), dtype=jnp.float32),
    )

    def step(carry, index):
        current, person_sum, person_square_sum, archive = carry
        current, metrics = classic_sweep(
            current, data, priors, num_terms=num_terms
        )
        person_sum += current.person
        person_square_sum += jnp.square(current.person)
        archive = jax.lax.cond(
            (index + 1) % archive_stride == 0,
            lambda values: values.at[index // archive_stride].set(
                current.person.astype(jnp.float32)
            ), lambda values: values, archive,
        )
        global_draw = {
            "mu_a": current.mu_a,
            "sigma_a": jnp.sqrt(current.sigma_a2),
            "mu_B": current.mu_b,
            "Gamma": current.gamma,
            "Gamma_global": current.hs_global,
            "Gamma_local": current.hs_local,
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

    (final, person_sum, person_square_sum, archive), (metrics, globals_) = (
        jax.lax.scan(step, initial, jnp.arange(num_steps))
    )
    return final, metrics, globals_, person_sum, person_square_sum, archive
