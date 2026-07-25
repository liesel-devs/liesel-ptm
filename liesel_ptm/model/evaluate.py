import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd

# this is a guard against a spurious syntaxwarning-turned-error
# caused by an invalid escape in the docstrings of properscoring
try:
    import properscoring as ps
except SyntaxError:
    pass

from jax.scipy.integrate import trapezoid

from ..util.split_dict import split_dict_rough
from ..util.summary import subsample_tree
from ..waic import waic as waic_fun
from .model import LocScalePTM

Array = Any
KeyArray = Any

logger = logging.getLogger(__name__)


class EvaluatePTM:
    """Helpers to evaluate predictive performance of a PTM model.

    Parameters
    ----------
    model
        A :class:`LocScalePTM` model instance.
    samples
        Posterior samples dictionary used for predictive evaluation.

    Attributes
    ----------
    model
        The provided model instance.
    samples
        The posterior samples dictionary.
    """

    def __init__(self, model: LocScalePTM, samples: dict[str, Array]) -> None:
        self.model = model
        self.samples = samples

    def __waic(self) -> pd.DataFrame:
        """Compute WAIC using current posterior samples (private helper)."""
        dist = self.model.init_dist(self.samples)
        log_prob = dist.log_prob(self.model.response.value)
        return waic_fun(log_prob)

    def log_prob(self, newdata: dict[str, Array] | None = None) -> Array:
        """Compute log-probabilities of `response` under posterior samples.

        The `newdata` dict must contain the response under the model's
        response name; it will be removed before building the predictive
        distribution.
        """

        samples = self.samples
        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, None)
        if response is None:
            raise ValueError("No response values provided in newdata.")

        dist = self.model.init_dist(samples, newdata=newdata)
        return dist.log_prob(response)

    def _lppdi(self, newdata: dict[str, Array] | None = None) -> Array:
        """
        Log pointwise predictive density contributions.
        """
        log_prob_samples = self.log_prob(newdata)

        nsamples = log_prob_samples.shape[0] * log_prob_samples.shape[1]

        lppd_sum = jax.scipy.special.logsumexp(log_prob_samples, axis=(0, 1))
        lppd_i = lppd_sum - jnp.log(nsamples)
        return lppd_i

    def lppdi(self, newdata: dict[str, "Array"] | None = None) -> "Array":
        """Compute pointwise log predictive density (averaged over samples).

        Returns an array of length N with the log pointwise predictive density
        for each observation.
        """
        return self._lppdi(newdata)

    def _waic(self):
        """Compute WAIC aggregates from log-probability samples (private)."""

        dist = self.model.init_dist(self.samples)
        log_prob_samples = dist.log_prob(self.model.response.value)

        nsamples = log_prob_samples.shape[0] * log_prob_samples.shape[1]
        nobs = log_prob_samples.shape[-1]

        waic_lppd_i = jax.scipy.special.logsumexp(
            log_prob_samples, axis=(0, 1)
        ) - jnp.log(nsamples)
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
            index=pd.Index([0]),
        )
        return waic_df

    def waic(self):
        """Compute WAIC table for the model from posterior samples.

        Returns a one-row DataFrame with WAIC aggregates and a warning count.
        """
        return self._waic()

    def log_score(self, newdata: dict[str, Array] | None = None) -> Array:
        """Negative log pointwise predictive density (sum over observations)."""

        return -self.lppdi(newdata).sum()

    def predictive_pdf(self, newdata: dict[str, Array] | None = None) -> Array:
        """Return the pointwise predictive pdf (exponentiated lppd)."""

        return jnp.exp(self.lppdi(newdata))

    def cdf_mad(
        self,
        true_cdf: Array,
        newdata: dict[str, Array] | None = None,
    ) -> Array:
        """Mean absolute deviation between true CDF and predictive CDF.

        Returns an array over posterior samples with per-sample MAD values.
        """

        samples = self.samples
        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, None)
        if response is None:
            raise ValueError("No response values provided in newdata.")

        dist = self.model.init_dist(samples, newdata=newdata)
        cdf_samples = dist.cdf(response)
        w1_samples = jnp.mean(jnp.abs(true_cdf - cdf_samples), axis=-1)
        return w1_samples

    def mad(
        self,
        true_cdf: "Array",
        newdata: dict[str, "Array"] | None = None,
    ) -> "Array":
        """Alias for :meth:`_cdf_mad` returning MAD reshaped to samples dims."""

        return self._cdf_mad(true_cdf, newdata)

    def _cdf_mad(
        self,
        true_cdf: "Array",
        newdata: dict[str, "Array"] | None = None,
    ) -> "Array":
        """Compute per-sample MAD between true and predictive CDFs (private)."""
        return self.cdf_mad(true_cdf, newdata)

    def quantile_mse(
        self,
        true_cdf: Array | None = None,
        newdata: dict[str, Array] | None = None,
    ) -> Array:
        """Mean squared error between true values and predictive quantiles.

        The input `true_cdf` is interpreted as probabilities for which quantiles
        are compared to observed responses.
        """

        samples = self.samples
        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, None)
        if response is None:
            raise ValueError("No response values provided in newdata.")

        dist = self.model.init_dist(samples, newdata=newdata)
        q_samples = jax.jit(dist.quantile)(true_cdf)
        w2_samples = jnp.mean(jnp.square(response - q_samples), axis=-1)

        return w2_samples

    def kld(
        self,
        true_log_prob: Array,
        newdata: dict[str, Array] | None = None,
    ) -> Array:
        """Estimate KL divergence between true log-prob and posterior predictive.

        Returns the average difference `E[true_log_prob - lppd]`.
        """

        lppdi = self.lppdi(newdata)
        kld = jnp.mean(true_log_prob - lppdi)
        return kld

    def quantile_score(
        self,
        probs: Array,
        newdata: dict[str, Array] | None = None,
    ) -> pd.DataFrame:
        """Compute mean and sd of the quantile score at given probabilities.

        Returns a DataFrame with mean, sd and probability values.
        """

        samples = self.samples
        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, self.model.response.value)
        dist = self.model.init_dist(samples, newdata=newdata)

        def qs_(probs):
            probs = jnp.reshape(probs, (jnp.shape(probs)[0], 1, 1, 1))
            quantiles = dist.quantile(probs)

            probs = jnp.swapaxes(probs, 0, -1)
            probs = jnp.moveaxis(probs, 0, 2)
            quantiles = jnp.swapaxes(quantiles, 0, -1)
            quantiles = jnp.moveaxis(quantiles, 0, 2)

            response_reshaped = jnp.reshape(response, (1, 1, jnp.shape(response)[0], 1))

            deviation = quantiles - response_reshaped
            weight = 2 * (jnp.heaviside(deviation, 0.0) - probs)
            quantile_score = weight * deviation

            mean_quantile_score = jnp.mean(
                quantile_score, axis=(0, 1, 2)
            )  # mean over samples and observations

            quantile_score_std = jnp.std(
                quantile_score, axis=(0, 1, 2)
            )  # mean over samples and observations
            return mean_quantile_score, quantile_score_std

        mean_quantile_score, quantile_score_std = jax.jit(qs_)(probs)

        quantile_score_df = pd.DataFrame(
            {
                "quantile_score_mean": mean_quantile_score,
                "quantile_score_sd": quantile_score_std,
                "prob": probs.squeeze(),
            }
        )
        return quantile_score_df

    def quantile_score_samples(
        self,
        probs: Array,
        newdata: dict[str, Array] | None = None,
    ) -> pd.DataFrame:
        """Return per-sample quantile scores averaged over observations."""

        samples = self.samples
        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, self.model.response.value)
        dist = self.model.init_dist(samples, newdata=newdata)

        def qs_(probs):
            probs = jnp.reshape(probs, (jnp.shape(probs)[0], 1, 1, 1))
            quantiles = dist.quantile(probs)

            probs = jnp.swapaxes(probs, 0, -1)
            probs = jnp.moveaxis(probs, 0, 2)
            quantiles = jnp.swapaxes(quantiles, 0, -1)
            quantiles = jnp.moveaxis(quantiles, 0, 2)

            response_reshaped = jnp.reshape(response, (1, 1, jnp.shape(response)[0], 1))

            deviation = quantiles - response_reshaped
            weight = 2 * (jnp.heaviside(deviation, 0.0) - probs)
            quantile_score = weight * deviation

            # mean over observations
            quantile_score_samples = jnp.mean(quantile_score, axis=2)

            return quantile_score_samples

        return jax.jit(qs_)(probs)

    def _crps_by_integrated_quantile_score(
        self,
        probs: Array,
        newdata: dict[str, Array] | None = None,
        k: int = 1,
        quantile_weight_fn: Callable[[Array], Array] = lambda q: jnp.ones_like(q),
    ):
        """
        CRPS by numerical integration over quantile score.

        To reduce memory burden, the computation proceeds in batches, based on
        partitioning the posterior samples into ``k`` chunks.

        The returned array has shape ``(Chains, Samples)``; each element of the array
        is already averaged over Ntest.
        """

        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, self.model.response.value)
        samples = self.samples

        def crps_(probs, samples):
            probs = jnp.reshape(probs, (jnp.shape(probs)[0], 1, 1, 1))
            dist = self.model.init_dist(samples, newdata=newdata)
            quantiles = dist.quantile(probs)
            quantile_weights = quantile_weight_fn(probs)

            probs = jnp.swapaxes(probs, 0, -1)
            probs = jnp.moveaxis(probs, 0, 2)
            quantiles = jnp.swapaxes(quantiles, 0, -1)
            quantiles = jnp.moveaxis(quantiles, 0, 2)
            quantile_weights = jnp.swapaxes(quantile_weights, 0, -1)
            quantile_weights = jnp.moveaxis(quantile_weights, 0, 2)

            response_reshaped = jnp.reshape(response, (1, 1, jnp.shape(response)[0], 1))

            deviation = quantiles - response_reshaped
            weight = 2 * (jnp.heaviside(deviation, 0.0) - probs)
            quantile_score = weight * deviation * quantile_weights

            crps_samples = trapezoid(quantile_score, probs, axis=3)
            return crps_samples.mean(axis=-2)

        samples_partitions = split_dict_rough(samples, k=k)
        crps_partitions = []
        for samp in samples_partitions:
            crps_partition = crps_(probs, samp)
            crps_partitions.append(crps_partition)
        return jnp.concatenate(crps_partitions, axis=1)

    def crps_by_estimated_quantiles(
        self,
        key: jax.Array,
        probs: Array,
        newdata: dict[str, Array] | None = None,
        quantile_weight_fn: Callable[[Array], Array] = lambda q: jnp.ones_like(q),
        m: int = 1,
    ):
        """
        Computes CRPS as follows:

        1. Draw samples from the posterior predictive distribution.
        2. Use these samples to estimate the quantiles for the probability levels
           in ``probs``.
        3. Compute the quantile score using these quantiles
        4. Integrate over the quantile score

        The output has shape ``(N_test,)``.

        By default, one predictive sample is drawn for each posterior sample.
        The argument ``m`` can be used to multiply the number of samples, i.e. for
        ``m=2``, two predictive samples are drawn for each posterior sample.
        """

        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, self.model.response.value)
        samples = self.samples

        def sample_(key, samples, n):
            dist = self.model.init_dist(samples, newdata=newdata)
            return dist.sample(n, key)

        def quantile_(samples, probs, key):
            event_samples = sample_(key, samples, m)
            quantiles = jnp.quantile(event_samples, q=probs, axis=(0, 1, 2))
            return quantiles

        def crps_(probs, samples, key):
            quantiles = quantile_(samples, probs, key)  # (probs, n)

            probs = jnp.reshape(probs, (jnp.shape(probs)[0], 1))
            quantile_weights = quantile_weight_fn(probs)

            probs = jnp.swapaxes(probs, 0, -1)
            quantiles = jnp.swapaxes(quantiles, 0, -1)
            quantile_weights = jnp.swapaxes(quantile_weights, 0, -1)

            response_reshaped = jnp.reshape(response, (jnp.shape(response)[0], 1))

            deviation = quantiles - response_reshaped
            weight = 2 * (jnp.heaviside(deviation, 0.0) - probs)
            quantile_score = weight * deviation * quantile_weights

            crps_contributions = trapezoid(quantile_score, probs, axis=1)
            return crps_contributions

        key, subkey = jax.random.split(key)
        return crps_(probs, samples, subkey)

    def crps(
        self,
        probs: Array,
        newdata: dict[str, Array] | None = None,
        k: int = 10,
        quantile_weight_fn: Callable[[Array], Array] = lambda q: jnp.ones_like(q),
    ):
        return self._crps_by_integrated_quantile_score(
            probs,
            newdata,
            k=k,
            quantile_weight_fn=quantile_weight_fn,
        )

    def crps_sample(
        self,
        key: KeyArray,
        predictive_samples_n: int,
        newdata: dict[str, Array] | None,
        subsamples_n: int | None = None,
        n_chunk: int = 500,
    ):
        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, self.model.response.value)
        if response is None:
            raise ValueError("No response values provided in newdata.")

        test_data = response
        samples = self.samples

        if subsamples_n:
            key, subkey = jax.random.split(key)
            subsamples = subsample_tree(subkey, self.samples, num_samples=subsamples_n)
            samples = subsamples

        ntest = test_data.shape[0]
        dist = self.model.init_dist(samples, newdata=newdata)
        key, subkey = jax.random.split(key)
        pred_samples = dist.sample(predictive_samples_n, seed=key)
        nsamp, c, s, ntest_ = pred_samples.shape
        if ntest_ == 1:
            pred_samples = dist.sample((predictive_samples_n, ntest), seed=key)
            nsamp, _, c, s, _ = pred_samples.shape
        pred_samples = jnp.reshape(pred_samples, shape=(nsamp * c * s, ntest))

        n_inf = jnp.isinf(pred_samples).sum()
        if n_inf > 0:
            logger.warning(
                f"Found {n_inf} infinite values in predictive sample. Changing to NaN."
            )
            pred_samples = pred_samples.at[jnp.where(jnp.isinf(pred_samples))].set(
                jnp.nan
            )

        crps_vals = []

        for i in range(0, ntest, n_chunk):
            chunk = pred_samples[:, i : i + n_chunk]  # (nsamples, nchunk)
            crps_chunk = ps.crps_ensemble(
                test_data[i : i + n_chunk], chunk.T
            )  # shape (nchunk,)
            crps_vals.append(crps_chunk.mean())

        return jnp.mean(jnp.stack(crps_vals))

    def _crps(self, probs: jnp.ndarray, newdata: dict[str, jnp.ndarray] | None = None):
        """Compute CRPS via integration over predictive quantiles.

        Accepts an array of probabilities and returns the CRPS averaged over
        observations.

        This function has some problem that I do not understand yet. Do not use!
        """

        samples = self.samples
        newdata = {} if newdata is None else dict(newdata)
        response = newdata.pop(self.model.response.name, self.model.response.value)
        dist = self.model.init_dist(samples, newdata=newdata)

        probs = jnp.asarray(probs).reshape(-1)
        probs = jnp.clip(probs, 0.0, 1.0)
        probs = jnp.sort(probs)

        y = jnp.asarray(response)  # may be shape [Nobs], broadcast against quantiles

        # Mean over (nchains, nsamples) at each p to keep memory small.
        def f_mean_at_p(p):
            q = dist.quantile(p)  # shape: [1, nchains, nsamples, 1]
            q = q[0, ...]  # shape: [nchains, nsamples, 1]
            dev = q - y  # broadcasts y to q's shape
            f = 2.0 * (jnp.heaviside(dev, 0.0) - p) * dev
            return f.mean(axis=(0, 1))  # shape: [1] (or [Nobs] if your last dim > 1)

        def _single_prob_case(ps):
            f0m = f_mean_at_p(ps[0])
            return jnp.zeros_like(f0m)

        def _general_case(ps):
            p0 = ps[0]
            f0m = f_mean_at_p(p0)
            acc0 = jnp.zeros_like(f0m)

            def step(carry, p_i):
                prev_p, prev_fm, acc = carry
                fm_i = f_mean_at_p(p_i)
                acc = acc + 0.5 * (prev_fm + fm_i) * (p_i - prev_p)
                return (p_i, fm_i, acc), ()

            (_, _, acc), _ = jax.lax.scan(step, (p0, f0m, acc0), ps[1:])
            return acc

        crps_mean = jax.lax.cond(
            probs.size <= 1, _single_prob_case, _general_case, probs
        )
        return crps_mean
