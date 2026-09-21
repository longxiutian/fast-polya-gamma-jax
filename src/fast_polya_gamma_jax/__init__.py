"""Fast Pólya-Gamma approximations implemented in pure JAX."""

from .moments import (
    pg1_mean,
    pg1_series_mean,
    pg1_series_variance,
    pg1_tail_mean,
    pg1_tail_variance,
    pg1_variance,
)
from .sampler import sample_pg1

__all__ = [
    "pg1_mean",
    "pg1_series_mean",
    "pg1_series_variance",
    "pg1_tail_mean",
    "pg1_tail_variance",
    "pg1_variance",
    "sample_pg1",
]

__version__ = "0.1.0"
