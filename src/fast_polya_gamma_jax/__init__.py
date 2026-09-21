"""Fast Pólya-Gamma approximations implemented in pure JAX."""

from .block_gibbs import (
    BlockGibbsData,
    BlockGibbsMetrics,
    BlockGibbsPriors,
    BlockGibbsState,
    block_gibbs_sweep,
    default_block_gibbs_priors,
    initialize_block_gibbs_state,
    linear_predictor,
    prepare_block_gibbs_data,
    run_block_gibbs,
)
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
    "BlockGibbsData",
    "BlockGibbsMetrics",
    "BlockGibbsPriors",
    "BlockGibbsState",
    "block_gibbs_sweep",
    "default_block_gibbs_priors",
    "initialize_block_gibbs_state",
    "linear_predictor",
    "pg1_mean",
    "pg1_series_mean",
    "pg1_series_variance",
    "pg1_tail_mean",
    "pg1_tail_variance",
    "pg1_variance",
    "prepare_block_gibbs_data",
    "run_block_gibbs",
    "sample_pg1",
]

__version__ = "0.1.0"
