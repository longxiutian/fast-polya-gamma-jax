# Classical HB PG Sampler

This is a separate inference implementation of `micro_panel_model` in the NBE
project. The original HMC code remains unchanged. It uses the same frozen
training packs as the other teachers and never reads heldout outcomes during
fitting.

## Statistical contract

For respondent `i`, exposure `r`, campaign `c`, and ad `j`:

```text
y_ir | a_i, B_i, u_c, v_j, delta ~ Bernoulli(sigmoid(eta_ir))
eta_ir = a_i + Z_ir' B_i + u_c + v_j + D_ir' delta
a_i | mu_a, sigma_a ~ N(mu_a, sigma_a^2)
B_i | mu_B, Gamma, Sigma_B ~ N(mu_B + Gamma A_i, Sigma_B)
```

The independent hyperpriors match `micro_panel_teacher.PRIORS`:
`mu_a ~ N(0,1.5^2)`, `sigma_a ~ HalfNormal(0.25)`,
`mu_B[k] ~ N(0,1)`, `delta[d] ~ N(0,0.5^2)`, and both campaign and ad
standard deviations are `HalfNormal(0.5)`. Each `Gamma[k,d]` has the
regularized horseshoe induced by a `HalfCauchy(0.1/sqrt(p))` row-global scale,
`HalfCauchy(1)` local scale, and slab standard deviation 1. The three
`Sigma_B` scales are folded Student-t with 4 degrees of freedom and scale
0.5; its correlation has an `LKJ(2)` prior.

The `PG(1, eta)` augmentation makes respondent, campaign, ad, and global
control blocks conditionally Gaussian. The nonconjugate half-normal, folded-t,
LKJ, and horseshoe scale priors receive Metropolis corrections. After each
centered sweep, an ancillary-sufficiency interweaving step holds
`a_i-mu_a` and `B_i-mu_B` fixed and redraws the four population locations
jointly under the augmented likelihood. This is an off-centered move, not a
different hierarchical model.

A second interweaving block holds `(a_i-mu_a)/sigma_a` and each
`(B_ik-mu_Bk-Gamma_k A_i)/tau_Bk` fixed while updating `sigma_a` and the
three `tau_B` scales by Metropolis steps in log-scale coordinates. The
correlation matrix is unchanged in this block. Its augmented log-likelihood
ratio uses one four-dimensional gradient and curvature matrix for all four
scalar proposals, avoiding four extra passes through the exposures. The
log-scale proposal standard deviation is 0.01 and its acceptance rate is
archived for tuning review. Neither interweaving block replaces the centered
hyperparameter updates; both preserve their conditional target.

The PG generator truncates the infinite gamma series after `K` terms and
replaces the omitted tail by its exact mean. It is intentionally approximate;
the chain is **not exact MCMC for the Bernoulli-logit posterior**. Compare
`K=8`, `K=16`, and a small exact-PG or HMC reference before using substantive
results. A million sweeps do not by themselves establish adequate ESS or
convergence.

## Run contract

The `longleaf/` submission examples require an explicit Slurm allocation via
`sbatch --account=YOUR_ACCOUNT` and a `PG_PYTHON` environment variable naming
the cluster's Python executable. Project and data roots remain caller-supplied
environment variables; no workstation-specific interpreter or dataset is bundled.

`scripts/run_classic_hb_teacher.py` checks the four-split registry and all
training-pack hashes before compiling the float64 GPU kernel. A production
chain uses 20,000 warmup sweeps and 1,000,000 retained sweeps. Every retained
sweep contributes to respondent posterior means and variances. Global traces
are archived every 20 sweeps; full respondent states every 1,000 sweeps. Two
chains per split are expected. Every chunk is atomically written and hashed;
`scripts/verify_classic_hb_teacher.py` checks the completed chain.

Before reporting, check split R-hat and ESS for global locations, scales,
horseshoe parameters, covariance, campaign/ad effects, and selected
respondent coefficients. Compare posterior predictive distributions with the
existing HMC implementation where its diagnostics are credible. Report the PG
approximation as a separate computational sensitivity if truncation affects
the comparison. Never label a chain certified merely because it completed.

Reference: Polson, Scott, and Windle (2013), *Journal of the American
Statistical Association* 108(504): 1339--1349,
<https://doi.org/10.1080/01621459.2013.829001>.
