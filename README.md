# fast-polya-gamma-jax

Fast, fixed-cost approximations to `PG(1, eta)` implemented in pure JAX.

This project targets the common Bernoulli-logit Pólya-Gamma update

```text
omega_n | eta_n ~ PG(1, eta_n)
```

when accelerator throughput and predictable runtime matter more than exact
random-variate generation. It provides a standalone JAX kernel and a thin
adapter for NumPyro's composable `CustomGibbs` API.

> **Status:** experimental alpha. The sampler is deliberately approximate.
> It is not a drop-in replacement when exact Pólya-Gamma draws are required.

## Why another sampler?

The defining Pólya-Gamma representation is an infinite weighted sum of gamma
variates. Exact Devroye samplers avoid a fixed truncation through
alternating-series rejection, but their variable control flow is awkward on
accelerators. NumPyro's `TruncatedPolyaGamma` is accelerator compatible, but it
only represents a clipped approximation to `PG(1, 0)` and uses general gamma
draws.

For shape one, every series variate is

```text
Gamma(1, 1) = Exponential(1).
```

That identity gives us a simple, fixed-shape JAX kernel built from uniform
random bits, logarithms, elementwise arithmetic, and one reduction.

## Installation

From a checkout:

```bash
python -m pip install -e .
```

For the NumPyro adapter:

```bash
python -m pip install -e ".[numpyro]"
```

Python 3.11 or newer is required so the optional integration can use NumPyro's
current composable Gibbs API.

## Quick start

```python
import jax
import jax.numpy as jnp

from fast_polya_gamma_jax import sample_pg1

key = jax.random.key(0)
eta = jnp.array([0.0, 1.0, 4.0])

omega = sample_pg1(
    key,
    eta,
    num_terms=16,
    tail_correction=True,
)
```

The function is JIT compatible. `num_terms` and `tail_correction` should be
static when compiled:

```python
from functools import partial

sample_fast = jax.jit(
    partial(sample_pg1, num_terms=16, tail_correction=True)
)
omega = sample_fast(key, eta)
```

## Approximation

For `E_k ~ Exponential(1)`, the explicit head is

```text
                  1       K             E_k
omega_K(eta) = ------- * sum  --------------------------- .
                2 pi^2   k=1  (k - 1/2)^2 + eta^2/(4 pi^2)
```

The default correction adds the exact expectation of the omitted tail:

```text
r_K(eta) = E[PG(1, eta)] - E[omega_K(eta)],

E[PG(1, eta)] = tanh(eta / 2) / (2 eta),
E[PG(1, 0)]   = 1/4.
```

Consequently, the corrected sampler has the exact analytical mean. Because the
tail is replaced by its expectation, its variance is understated by exactly
the omitted tail variance.

Across a dense grid with `|eta| <= 12`:

| Terms | Maximum omitted variance | Maximum uncorrected tail mean |
| ---: | ---: | ---: |
| 8 | 0.537% | 14.90% |
| 16 | 0.071% | 7.56% |

The second column explains why the mean correction matters. The first shows why
that deterministic correction is useful: the tail can carry appreciable mean
while carrying very little variance.

## Reference validation

We compared `K=8` and `K=16` corrected draws against the compiled exact Devroye
sampler in the `polyagamma` package at
`eta = {0, 1, 2, 4, 8, 12}`. Each configuration used 200,000 independent draws.
The largest observed two-sample Kolmogorov-Smirnov distance was `0.00422`.

This is a Monte Carlo comparison, not an error bound. Reproduce it with:

```bash
python -m pip install -e ".[test,reference]"
python benchmarks/validate_reference.py --samples 200000
```

The checked-in reference receipt is
[`benchmarks/baselines/reference_devroye_200k.json`](benchmarks/baselines/reference_devroye_200k.json).

## Preliminary performance

Warm-call throughput for 60,000 float64 draws, excluding compilation and using
the median of 30 repetitions:

| Kernel | Draws/second | Relative to NumPyro |
| --- | ---: | ---: |
| Custom `K=8` + tail mean | 14.84 million | 80.6x |
| Custom `K=16` + tail mean | 10.54 million | 57.3x |
| NumPyro `TruncatedPolyaGamma` | 0.184 million | 1.0x |

Environment: Windows CPU backend, Python 3.12, JAX 0.11.2, NumPyro 0.22.0.
These are preliminary machine-specific measurements. GPU results will be
reported separately rather than inferred from CPU behavior.

Run the benchmark with:

```bash
python benchmarks/benchmark_pg.py --size 60000 --repeats 30
```

The checked-in CPU receipt is
[`benchmarks/baselines/windows_cpu_jax_0_11_2.json`](benchmarks/baselines/windows_cpu_jax_0_11_2.json).

## NumPyro integration

The numerical sampler is pure JAX. NumPyro integration is an adapter rather
than a `Distribution` subclass because the fast tail-corrected sampler does not
have a comparably simple coherent `log_prob` implementation.

```python
from fast_polya_gamma_jax.numpyro import make_custom_gibbs


def linear_predictor(*, gibbs_sites, hmc_sites):
    del gibbs_sites
    beta = hmc_sites["beta"]
    return X @ beta


omega_kernel = make_custom_gibbs(
    linear_predictor,
    site_name="omega",
    num_terms=16,
    tail_correction=True,
)
```

The returned object is a NumPyro `CustomGibbs` block. A complete composite
Gibbs model must still declare the owned site and provide the other conditional
updates. For a fully conjugate augmentation, a hand-written JAX sweep may be
simpler and faster than forcing every block through a probabilistic-program
trace.

## Choosing `num_terms`

- Start with `16` when accuracy matters and PG sampling is not the bottleneck.
- Try `8` when fixed-cost throughput is the priority.
- Validate over the empirical distribution of `eta`, not only at zero.
- Increase the term count if large absolute logits are common or tail fidelity
  matters for the downstream Markov chain.

## Limitations

- Only `PG(1, eta)` is currently implemented.
- Draws are approximate even though the analytical mean correction is exact.
- The deterministic tail correction slightly understates variance.
- No `Distribution` subclass or general-purpose `log_prob` is provided.
- CPU performance does not establish GPU performance.
- Approximation quality for an MCMC application must be assessed at the
  posterior logits actually visited by that chain.

## Development

```bash
python -m pip install -e ".[test,reference]"
ruff check .
pytest -q
```

## References

- Polson, N. G., Scott, J. G., and Windle, J. (2013), "Bayesian Inference for
  Logistic Models Using Pólya-Gamma Latent Variables," *JASA*.
- Windle, J., Polson, N. G., and Scott, J. G. (2014), "Sampling Pólya-Gamma
  Random Variates: Alternate and Approximate Techniques."

## License

MIT. See [LICENSE](LICENSE).
