import logging
from typing import Any

import jax.numpy as jnp
import pandas as pd
from scipy.integrate import trapezoid

import liesel_ptm as ptm
from liesel_ptm.model import NewPTMLocScale
from liesel_ptm.ptm_ls import waic
from liesel_ptm.util import standardize

KeyArray = Any
Array = Any
SamplingResults = Any
Position = Any

logger = logging.getLogger("sim")


def kwargs_helper(x: Array, scale_terms: bool = False) -> dict[str, Array]:
    kwargs_loc = {f"x{i}_loc": x[:, i] for i in range(x.shape[-1])}

    if not scale_terms:
        return kwargs_loc

    kwargs_scale = {f"x{i}_scale": x[:, i] for i in range(x.shape[-1])}

    kwargs = kwargs_loc | kwargs_scale
    return kwargs


def kwargs_helper_fixed(
    value: float, index: int, x: Array, scale_terms: bool = False
) -> dict[str, Array]:
    """
    Sets *one* covariate value to the fixed value, fixing the others at zero.
    """
    x = jnp.zeros((1, x.shape[-1]))
    x = x.at[0, index].set(value)

    return kwargs_helper(x, scale_terms=scale_terms)


def model_mse(
    samples: dict[str, Array],
    model: NewPTMLocScale,
    test_df: pd.DataFrame,
    scale_terms: bool = False,
):
    x = test_df.loc[:, test_df.columns.str.startswith("x")].to_numpy()

    fx_summaries = []

    for i in range(x.shape[-1]):
        term = model.loc_model[f"x{i}_loc"]
        fx = term.predict(samples, x[:, i])
        fx = standardize(fx, center=True, scale=False, axis=-1)

        fx_summary = ptm.summarise_by_quantiles(fx, names_prefix="fx")
        fx_summary["x_index"] = i
        fx_summary["x_label"] = f"x{i}"
        fx_summary["x_value"] = x[:, i]
        true = test_df[f"fx{i}_loc"].to_numpy()
        fx_summary["mse"] = jnp.square(true - fx).mean(axis=(0, 1))
        fx_summary["bias"] = (fx - true).mean(axis=(0, 1))
        fx_summary["variance"] = fx.var(axis=(0, 1))
        fx_summary["fx_true"] = true
        fx_summary["predictor"] = "loc"

        fx_summaries.append(fx_summary)

    if not scale_terms:
        fx_summaries = pd.concat(fx_summaries)
        logger.info("Returning MSE dataframe.")
        return fx_summaries

    fx_summaries_scale = []

    for i in range(x.shape[-1]):
        term = model.log_scale_model[f"x{i}_scale"]
        fx = term.predict(samples, x[:, i])
        fx = standardize(fx, center=True, scale=True, axis=-1)

        fx_summary = ptm.summarise_by_quantiles(fx, names_prefix="fx")
        fx_summary["x_index"] = i
        fx_summary["x_label"] = f"x{i}"
        fx_summary["x_value"] = x[:, i]
        true = test_df[f"fx{i}_scale"].to_numpy()
        fx_summary["mse"] = jnp.square(true - fx).mean(axis=(0, 1))
        fx_summary["bias"] = (fx - true).mean(axis=(0, 1))
        fx_summary["variance"] = fx.var(axis=(0, 1))
        fx_summary["fx_true"] = true
        fx_summary["predictor"] = "log_scale"

        fx_summaries_scale.append(fx_summary)

    fx_summaries = pd.concat(fx_summaries + fx_summaries_scale)

    logger.info("Returning MSE dataframe.")
    return fx_summaries


def model_mse_summary(mse_df: pd.DataFrame) -> pd.DataFrame:
    selected = mse_df[["mse", "bias", "variance", "x_label", "x_index", "predictor"]]
    grouped = selected.groupby(["x_label", "predictor"])
    mse = grouped.mean().reset_index()
    return mse


def model_mad(
    samples: dict[str, Array],
    model: NewPTMLocScale,
    test_df: pd.DataFrame,
    scale_terms: bool = False,
):
    y = test_df["y"].to_numpy()
    x = test_df.loc[:, test_df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(x, scale_terms=scale_terms)

    cdf = model.init_dist(samples, **kwargs).cdf(y)

    true_cdf = test_df["cdf"].to_numpy()
    absdiff = jnp.abs(true_cdf - cdf).mean(axis=-1, keepdims=True)
    mad_df = ptm.summarise_by_quantiles(absdiff, names_prefix="mad_cdf")
    logger.info("Returning mean absolute difference dataframe.")
    return mad_df


def model_kld(
    samples: dict[str, Array],
    model: NewPTMLocScale,
    test_df: pd.DataFrame,
    scale_terms: bool = False,
):
    y = test_df["y"].to_numpy()
    x = test_df.loc[:, test_df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(x, scale_terms=scale_terms)

    dist = model.init_dist(samples, **kwargs)
    log_pdf = dist.log_prob(y)

    true_log_pdf = test_df["log_prob"].to_numpy()
    kld = jnp.mean(true_log_pdf - log_pdf, axis=-1, keepdims=True)
    kld_df = ptm.summarise_by_quantiles(kld, names_prefix="kld")
    logger.info("Returning KLD dataframe.")
    return kld_df


def model_waic(
    samples: dict[str, Array],
    model: NewPTMLocScale,
    df: pd.DataFrame,
    scale_terms: bool = False,
):
    y = df["y"].to_numpy()
    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(x, scale_terms=scale_terms)

    dist = model.init_dist(samples, **kwargs)
    log_prob = dist.log_prob(y)

    logger.info("Returning WAIC dataframe.")
    return waic(log_prob_samples=log_prob)


def model_deviance(
    samples: dict[str, Array],
    model: NewPTMLocScale,
    df: pd.DataFrame,
    scale_terms: bool = False,
):
    y = df["y"].to_numpy()
    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(x, scale_terms=scale_terms)

    dist = model.init_dist(samples, **kwargs)
    log_prob = dist.log_prob(y).sum(axis=-1)

    deviance = -2 * log_prob
    mean_deviance = deviance.mean()

    sample_means = ptm.sample_means(samples, keepdims=True)
    dist = model.init_dist(sample_means, **kwargs)
    deviance_at_mean = jnp.squeeze(-2 * dist.log_prob(y).sum(axis=-1))

    dic_p = mean_deviance - deviance_at_mean
    dic = mean_deviance + dic_p
    dic_elpd = -dic / 2

    dic_df = pd.DataFrame(
        {
            "dic_elpd": dic_elpd,
            "dic_p": dic_p,
            "dic": dic,
        },
        index=[0],
    )
    logger.info("Returning DIC dataframe.")
    return dic_df


def model_quantile_score(
    samples: dict[str, Array],
    model: NewPTMLocScale,
    df: pd.DataFrame,
    scale_terms: bool = False,
):
    """This can also be regarded as a decomposed CRPS."""
    y = df["y"].to_numpy()
    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(x, scale_terms=scale_terms)
    dist = model.init_dist(samples, **kwargs)

    y_reshaped = jnp.reshape(y, (1, 1, y.shape[-1], 1))
    alphas = jnp.linspace(0.005, 0.995, 25)
    alphas = jnp.reshape(alphas, shape=(25, 1, 1, 1))
    quantiles = dist.quantile(alphas)

    quantiles = jnp.swapaxes(quantiles, 0, -1)
    quantiles = jnp.moveaxis(quantiles, 0, 2)

    alphas = jnp.swapaxes(alphas, 0, -1)
    alphas = jnp.moveaxis(alphas, 0, 2)

    deviation = quantiles - y_reshaped
    weight = 2 * (jnp.heaviside(deviation, 0.0) - alphas)
    quantile_score = weight * deviation
    mean_quantile_score = jnp.mean(quantile_score, axis=(0, 1, 2))  # mean over samples

    quantile_score_df = pd.DataFrame(
        {"quantile_score": mean_quantile_score, "alpha": alphas.squeeze()}
    )

    logger.info("Returning decomposed CRPS dataframe.")
    return quantile_score_df


def model_crps(
    samples: dict[str, Array],
    model: NewPTMLocScale,
    df: pd.DataFrame,
    scale_terms: bool = False,
):
    """This can also be regarded as a decomposed CRPS."""
    y = df["y"].to_numpy()
    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(x, scale_terms=scale_terms)
    dist = model.init_dist(samples, **kwargs)

    y_reshaped = jnp.reshape(y, (1, 1, y.shape[-1], 1))
    alphas = jnp.linspace(0.0005, 0.9995, 1000)
    quantiles = dist.quantile(alphas)

    deviation = quantiles - y_reshaped
    weight = 2 * (jnp.heaviside(deviation, 0.0) - alphas)
    quantile_score = weight * deviation

    crps_trap = jnp.mean(trapezoid(quantile_score, alphas, axis=3))

    crps_df = pd.DataFrame({"crps": crps_trap, "crps_trap": crps_trap}, index=[0])

    logger.info("Returning CRPS dataframe.")
    return crps_df
