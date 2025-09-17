from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pandas as pd
from jax.scipy.special import logsumexp

Array = Any


def waic(log_prob_samples: Array) -> pd.DataFrame:
    """
    Summarises the widely applicable information criterion for an array of log
    likelihood samples.
    """
    assert len(log_prob_samples.shape) == 3

    nsamples = log_prob_samples.shape[0] * log_prob_samples.shape[1]
    nobs = log_prob_samples.shape[-1]

    waic_lppd_i = logsumexp(log_prob_samples, axis=(0, 1)) - jnp.log(nsamples)
    waic_p_i = jnp.var(log_prob_samples, axis=(0, 1))
    waic_elpd_i = waic_lppd_i - waic_p_i

    waic_se = jnp.std(waic_elpd_i) * jnp.sqrt(nobs)
    waic_p = waic_p_i.sum()
    waic_lppd = waic_lppd_i.sum()
    waic_elpd = waic_lppd - waic_p
    waic_deviance = -2 * waic_elpd
    n_var_greater_4 = jnp.sum(jnp.var(waic_lppd_i) > 4)

    waic_df = pd.DataFrame(
        {
            "waic_lppd": waic_lppd,
            "waic_elpd": waic_elpd,
            "waic_se": waic_se,
            "waic_p": waic_p,
            "waic_deviance": waic_deviance,
            "n_warning": n_var_greater_4,
        },
        index=[0],
    )
    return waic_df
