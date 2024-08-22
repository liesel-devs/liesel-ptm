import json
import logging
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import liesel.goose as gs
import pandas as pd
from scipy.integrate import trapezoid

import liesel_ptm as ptm
from liesel_ptm.__version__ import __version__
from liesel_ptm.model import OnionPTMLocScale
from liesel_ptm.ptm_ls import waic
from liesel_ptm.util import log_exception_and_pass, standardize

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


def model_setup(
    df: pd.DataFrame,
    prior_tau2_covariates: tuple[type[ptm.Var], dict[str, float]],
    prior_tau2_normalization: tuple[type[ptm.Var], dict[str, float]],
    scale_terms: bool = False,
    nshape: int = 30,
    centered: bool = True,
    scaled: bool = True,
) -> OnionPTMLocScale:
    logger.info("Starting model setup.")

    y = df["y"].to_numpy()
    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    tau2_shape_cls, tau2_shape_kwargs = prior_tau2_normalization
    tau2_normalization = tau2_shape_cls(**tau2_shape_kwargs, name="normalization_tau2")
    model = OnionPTMLocScale(
        y=y, nparam=nshape, tau2=tau2_normalization, centered=centered, scaled=scaled
    )

    tau2_cov_cls, tau2_cov_kwargs = prior_tau2_covariates
    for i in range(x.shape[-1]):
        x = df[f"x{i}"].to_numpy()

        tau2 = tau2_cov_cls(**tau2_cov_kwargs, name=f"x{i}_loc_tau2")
        model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2, name=f"x{i}_loc")

        if scale_terms:
            tau2 = tau2_cov_cls(**tau2_cov_kwargs, name=f"x{i}_scale_tau2")
            model.log_scale_model += ptm.S.pspline(
                x, nparam=20, tau2=tau2, name=f"x{i}_scale"
            )

    return model


def model_fit(
    seed: int,
    model: OnionPTMLocScale,
    warmup: int,
    posterior: int,
    cache_path: Path | None = None,
    sample_transformation: bool = True,
) -> tuple[SamplingResults, float, float]:
    pre_opt_start = time.time()
    model.optimize_locscale(max_iter=10_000)
    model.optimize_transformation(max_iter=10_000)
    pre_opt_end = time.time()

    pre_opt_duration = pre_opt_end - pre_opt_start

    eb = model.setup_engine_builder(
        seed=seed, num_chains=4, sample_transformation=sample_transformation
    )

    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    mcmc_start = time.time()
    if cache_path is not None:
        logger.info(f"Caching results at '{cache_path}'.")
        results = ptm.cache_results(eb, cache_path)
    else:
        logger.info("Not caching results.")
        engine = eb.build()
        logger.info("Engine successfully built. Starting posterior sampling.")
        engine.sample_all_epochs()
        results = engine.get_results()

    mcmc_end = time.time()
    mcmc_duration = mcmc_end - mcmc_start

    logger.info("Posterior sampling finished.")
    return results, pre_opt_duration, mcmc_duration


def model_mse(
    samples: dict[str, Array],
    model: OnionPTMLocScale,
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
    model: OnionPTMLocScale,
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
    model: OnionPTMLocScale,
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
    model: OnionPTMLocScale,
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
    model: OnionPTMLocScale,
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
    model: OnionPTMLocScale,
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

    deviation = quantiles - y_reshaped
    weight = 2 * (jnp.heaviside(deviation, 0.0) - alphas)
    quantile_score = weight * deviation
    mean_quantile_score = jnp.mean(quantile_score, axis=(0, 1, 2))  # mean over samples

    quantile_score_df = pd.DataFrame(
        {"quantile_score": mean_quantile_score, "alpha": alphas}
    )

    logger.info("Returning decomposed CRPS dataframe.")
    return quantile_score_df


def model_crps(
    samples: dict[str, Array],
    model: OnionPTMLocScale,
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


def model_analysis(
    model: OnionPTMLocScale,
    results,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    show_df: pd.DataFrame,
    scale_terms: bool = False,
    nshape: int = 30,
    sample_transformation: bool = True,
) -> dict[str, pd.DataFrame]:
    logger.info("Starting model analysis.")

    samples = results.get_posterior_samples()
    if not sample_transformation:
        samples["normalization_shape_transformed"] = jnp.zeros(
            samples["loc_intercept"].shape + (nshape - 2,)
        )
    summary = gs.Summary(results)

    data = dict()

    data["param_df"] = summary.to_dataframe().reset_index()
    data["error_df"] = summary._error_df().reset_index()

    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    y = jnp.linspace(show_df.y.min() - 0.25, show_df.y.max() + 0.25, 200)

    with log_exception_and_pass(logger):
        summaries = []
        index = 3

        for val in [-1.0, 0.0, 1.0]:
            kwargs = kwargs_helper_fixed(val, index, x, scale_terms=scale_terms)

            summary = model.summarise_density_by_quantiles(
                y=y, samples=samples, **kwargs
            )

            non_x_cols = jnp.logical_not(summary.columns.str.startswith("x"))
            summary = summary[summary.columns[non_x_cols]]

            summary["x0"] = 0.0
            summary["x1"] = 0.0
            summary["x2"] = 0.0
            summary["x3"] = 0.0
            summary[f"x{index}"] = val

            summaries.append(summary)

        data["summary_by_quantiles"] = pd.concat(summaries)

    with log_exception_and_pass(logger):
        zmin = model.knots.knots.min()
        zmax = model.knots.knots.max()
        z = jnp.linspace(zmin - 1.0, zmax + 1.0, 300)

        data["trafo_df"] = model.summarise_transformation_by_quantiles(
            residuals=z, samples=samples
        )

    with log_exception_and_pass(logger):
        mse = model_mse(samples, model, df, scale_terms=scale_terms)
        data["mse_df"] = model_mse_summary(mse)

    with log_exception_and_pass(logger):
        data["mad_df"] = model_mad(samples, model, test_df, scale_terms=scale_terms)

    with log_exception_and_pass(logger):
        data["kld_df"] = model_kld(samples, model, test_df, scale_terms=scale_terms)

    with log_exception_and_pass(logger):
        data["waic_df"] = model_waic(samples, model, df, scale_terms=scale_terms)

    with log_exception_and_pass(logger):
        data["waic_df_test"] = model_waic(
            samples, model, test_df, scale_terms=scale_terms
        )

    with log_exception_and_pass(logger):
        data["quantile_score_df"] = model_quantile_score(
            samples, model, df, scale_terms=scale_terms
        )

    logger.info("Finished model analysis.")
    return data


def save_results(
    path: Path,
    prefix: str,
    data: dict[str, pd.DataFrame],
    additional_info: dict[str, Any],
):
    logger.info("Saving results.")

    for df_name, df in data.items():
        df["identifier"] = prefix

        for colname, value in additional_info.items():
            df = df.assign(**{colname: [value] * df.shape[0]})

        dir_ = path / df_name

        with log_exception_and_pass(logger):
            dir_.mkdir(exist_ok=True, parents=True)
            df.to_csv(dir_ / f"{prefix}-{df_name}.csv", index=False)

    logger.info("Results saved.")


TAU2_CLASSES: dict[str, type[ptm.Var]] = {
    "VarHalfCauchy": ptm.VarHalfCauchy,
    "VarInverseGamma": ptm.VarInverseGamma,
    "VarWeibull": ptm.VarWeibull,
}


def convert_tau2_string_to_tuple(
    tau2_specification: str,
) -> tuple[type[ptm.Var], dict[str, float]]:
    tau2_spec_dict = json.loads(tau2_specification)
    class_name = tau2_spec_dict.pop("class")
    class_ = TAU2_CLASSES[class_name]
    return class_, tau2_spec_dict


def prefix_dict(input_dict: dict, prefix: str) -> dict:
    return {f"{prefix}{key}": value for key, value in input_dict.items()}


def _one_run(
    seed: int,
    data_path: str | Path,
    out_path: str | Path,
    warmup: int,
    posterior: int,
    scale_terms: bool,
    sample_transformation: bool,
    identifier: str,
    n: int,
    prior_tau2_covariates: str,
    prior_tau2_normalization: str,
    id_data: dict[str, str],
    cache_path: Path | str | None = None,
    skip_if_results_exist: bool = True,
    centered: bool = True,
    scaled: bool = True,
):
    prefix = f"{identifier}-seed_{seed}"
    data_path = Path(data_path).resolve()
    out_path = Path(out_path).resolve()

    if cache_path is not None:
        cache_file = Path(cache_path) / f"{prefix}-results.pickle"
    else:
        cache_file = None

    train_path = data_path / "train" / f"train-seed_{seed}.csv"
    test_path = data_path / "test" / f"test-seed_{seed}.csv"
    show_path = data_path / "show" / f"show-seed_{seed}.csv"
    out = out_path / "out"
    out.mkdir(exist_ok=True, parents=True)

    info_df_path = out / "info" / f"{prefix}-info.csv"
    logger.info(f"Checking if {info_df_path} exists.")
    logger.info(f"{skip_if_results_exist=}")
    logger.info(f"{info_df_path.exists()=}")

    if skip_if_results_exist and info_df_path.exists():
        logger.warning(f"{info_df_path} exists. Skipping this run.")
        return "skipped"

    cache_path = Path(cache_path).resolve() if cache_path is not None else None

    logger.info(f"Starting run. Seed: {seed}.")

    df = pd.read_csv(train_path)
    df = df.iloc[:n, :]
    test_df = pd.read_csv(test_path)
    show_df = pd.read_csv(show_path)

    nshape = 30
    model = model_setup(
        df,
        scale_terms=scale_terms,
        prior_tau2_covariates=convert_tau2_string_to_tuple(prior_tau2_covariates),
        prior_tau2_normalization=convert_tau2_string_to_tuple(prior_tau2_normalization),
        nshape=nshape,
        centered=centered,
        scaled=scaled,
    )

    fit = model_fit(
        seed,
        model,
        warmup=warmup,
        posterior=posterior,
        cache_path=cache_file,
        sample_transformation=sample_transformation,
    )
    results, pre_opt_duration, mcmc_duration = fit

    info: dict[str, Any] = dict()
    info["seed"] = seed
    info["df_train"] = str(train_path)
    info["df_test"] = str(test_path)
    info["pre_opt_duration"] = pre_opt_duration
    info["mcmc_duration"] = mcmc_duration
    info["nobs"] = df.shape[0]
    info["identifier"] = identifier
    info["version"] = f"liesel_ptm-{__version__}"

    tau2_covariate_info = prefix_dict(
        json.loads(prior_tau2_covariates), prefix="tau2_covariates_"
    )
    tau2_normalization_info = prefix_dict(
        json.loads(prior_tau2_normalization), prefix="tau2_normalization_"
    )

    tau2_info = tau2_covariate_info | tau2_normalization_info
    additional_info: dict[str, Any] = {}
    additional_info["centered"] = centered
    additional_info["scaled"] = scaled

    additional_info |= id_data
    additional_info |= tau2_info

    data = model_analysis(
        model,
        results,
        df,
        test_df,
        show_df,
        scale_terms=scale_terms,
        nshape=nshape,
        sample_transformation=sample_transformation,
    )
    data["info"] = pd.DataFrame(info, index=[0])

    save_results(path=out, prefix=prefix, data=data, additional_info=additional_info)

    logger.info(f"Finished run. Seed: {seed}.")
    if cache_file is None:
        return
    logger.info(f"Deleting cache file. Seed: {seed}.")
    cache_file.unlink()
    if cache_file.exists():
        logger.error("Deleting cache file unsuccessful.")
    else:
        logger.info("Deleting cache file successful.")


def one_run(
    seed: int,
    data_path: str | Path,
    out_path: str | Path,
    warmup: int,
    posterior: int,
    scale_terms: bool,
    sample_transformation: bool,
    identifier: str,
    n: int,
    prior_tau2_covariates: str,
    prior_tau2_normalization: str,
    id_data: dict[str, str],
    cache_path: Path | str | None = None,
    skip_if_results_exist: bool = True,
    centered: bool = True,
    scaled: bool = True,
):
    try:
        logger = logging.getLogger("sim")
        mailog = logging.getLogger("mail")

        return _one_run(
            seed,
            data_path,
            out_path,
            warmup,
            posterior,
            scale_terms=scale_terms,
            sample_transformation=sample_transformation,
            identifier=identifier,
            n=n,
            prior_tau2_covariates=prior_tau2_covariates,
            prior_tau2_normalization=prior_tau2_normalization,
            cache_path=cache_path,
            skip_if_results_exist=skip_if_results_exist,
            id_data=id_data,
            centered=centered,
            scaled=scaled,
        )
    except Exception:
        logger.exception(f"Exception in run. Seed: {seed}. Identifier: {identifier}")
        mailog.exception(f"Exception in run. Seed: {seed}. Identifier: {identifier}")
