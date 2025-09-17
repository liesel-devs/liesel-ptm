import logging
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

from ..util.summary import subsample_tree
from ..waic import waic as waic_fun
from .model import LocScalePTM

Array = Any
KeyArray = Any

logger = logging.getLogger(__name__)


def _flatten_first_two_sample_dims(samples_pytree):
    """[S, C, ...] -> [S*C, ...] for every leaf in the PyTree."""
    leaves, treedef = jax.tree_util.tree_flatten(samples_pytree)
    S, C = leaves[0].shape[:2]

    def reshape(x):
        return x.reshape((S * C,) + x.shape[2:])

    return jax.tree_util.tree_unflatten(treedef, [reshape(x) for x in leaves]), (S * C)


def _index_pytree(pytree, i):
    """Take pytree[i] along axis 0 for each leaf."""
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x[i], (0, 1)), pytree)


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

        # Don’t mutate caller input; remove response before passing to init_dist
        nd = {} if newdata is None else dict(newdata)
        response = nd.pop(self.model.response.name, None)
        if response is None:
            raise ValueError("No response values provided in newdata.")
        nd = nd if nd else {}

        # Flatten [S, C, ...] -> [S*C, ...] so we can iterate samples on device
        samples_flat, nsamples = _flatten_first_two_sample_dims(self.samples)

        # Initialize running log-sum-exp over samples per observation i
        N = response.shape[0]
        lse0 = jnp.full((N,), -jnp.inf)

        def body(i, lse):
            sample_i = _index_pytree(samples_flat, i)
            dist_i = self.model.init_dist(sample_i, newdata=nd)
            lp_i = dist_i.log_prob(response).squeeze((0, 1))  # shape [N]
            return jnp.logaddexp(lse, lp_i)

        lse = jax.lax.fori_loop(0, nsamples, body, lse0)  # [N]
        return lse - jnp.log(nsamples)  # lppd_i: [N]

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
            index=[0],
        )
        return waic_df

    def waic(self):
        """Compute WAIC table for the model from posterior samples.

        Returns a one-row DataFrame with WAIC aggregates and a warning count.
        """

        # Response values
        y = self.model.response.value  # shape [N]
        N = y.shape[0]

        # Flatten two sample dims so we can iterate along axis 0 on device
        samples_flat, nsamples = _flatten_first_two_sample_dims(self.samples)

        # Accumulators (per observation):
        # - lse: running log-sum-exp of log p(y_i | theta) over samples
        # - mean, m2: Welford accumulators for variance of log p(y_i | theta)
        lse0 = jnp.full((N,), -jnp.inf)
        mean0 = jnp.zeros((N,))
        m2_0 = jnp.zeros((N,))
        n0 = jnp.array(0, dtype=jnp.int32)

        def body(i, state):
            lse, mean, m2, n = state
            # get i-th sample (no batch left on params)
            sample_i = _index_pytree(samples_flat, i)
            dist_i = self.model.init_dist(sample_i)
            lp_i = dist_i.log_prob(y).squeeze((0, 1))  # shape [N]

            # accumulate log-sum-exp for lppd
            lse = jnp.logaddexp(lse, lp_i)

            # Welford update for variance of log-likelihoods
            # (population variance, ddof=0)
            n_new = n + 1
            delta = lp_i - mean
            mean_new = mean + delta / n_new
            m2_new = m2 + delta * (lp_i - mean_new)

            return (lse, mean_new, m2_new, n_new)

        lse, mean, m2, n = jax.lax.fori_loop(0, nsamples, body, (lse0, mean0, m2_0, n0))

        # Pointwise WAIC pieces
        waic_lppd_i = lse - jnp.log(nsamples)  # log mean_s p(y_i | theta_s)
        waic_p_i = (
            m2 / n
        )  # variance over samples of log p (ddof=0 to match jnp.var default)
        waic_elpd_i = waic_lppd_i - waic_p_i

        # Aggregates
        waic_se = jnp.std(waic_elpd_i) * jnp.sqrt(N)
        waic_p = waic_p_i.sum()
        waic_lppd = waic_lppd_i.sum()
        waic_elpd = waic_lppd - waic_p
        waic_deviance = -2 * waic_elpd

        # Common warning: count observations with large posterior variance of log-lik
        n_var_greater_4 = jnp.sum(waic_p_i > 4)

        waic_df = pd.DataFrame(
            {
                "waic_lppd": [float(waic_lppd)],
                "waic_elpd": [float(waic_elpd)],
                "waic_se": [float(waic_se)],
                "waic_p": [float(waic_p)],
                "waic_deviance": [float(waic_deviance)],
                "n_warning": [int(n_var_greater_4)],
            }
        )
        return waic_df

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

        # Copy and extract response without mutating caller's dict
        nd = {} if newdata is None else dict(newdata)
        response = nd.pop(self.model.response.name, None)
        if response is None:
            raise ValueError("No response values provided in newdata.")

        leaves, treedef = jax.tree_util.tree_flatten(self.samples)
        S, C = leaves[0].shape[:2]

        # Flatten sample dims so we can iterate on device
        samples_flat, nsamples = _flatten_first_two_sample_dims(self.samples)

        # Output buffer: one scalar per sample
        out0 = jnp.zeros((nsamples,), dtype=jnp.result_type(true_cdf))

        def body(i, acc):
            sample_i = _index_pytree(samples_flat, i)
            dist_i = self.model.init_dist(sample_i, newdata=nd)
            cdf_i = dist_i.cdf(response).squeeze((0, 1))  # [N]
            w1_i = jnp.mean(jnp.abs(true_cdf - cdf_i))  # scalar
            return acc.at[i].set(w1_i)

        w1_flat = jax.lax.fori_loop(0, nsamples, body, out0)  # [nsamples]
        return w1_flat.reshape(S, C)  # matches original shape

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

    def _crps_inefficient(
        self,
        probs: Array,
        newdata: dict[str, Array] | None = None,
    ):
        """Inefficient CRPS by numerical integration over quantiles (private)."""

        samples = self.samples
        newdata = newdata if newdata is not None else {}
        response = newdata.pop(self.model.response.name, self.model.response.value)
        dist = self.model.init_dist(samples, newdata=newdata)

        def crps_(probs):
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

            crps_samples = trapezoid(quantile_score, probs, axis=3)
            return crps_samples

        return crps_(probs)

    def crps(
        self,
        probs: Array,
        newdata: dict[str, Array] | None = None,
    ):
        return self._crps_inefficient(probs, newdata)

    def crps_sample(
        self,
        key: KeyArray,
        predictive_samples_n: int,
        newdata: dict[str, Array] | None,
        subsamples_n: int | None = None,
        n_chunk: int = 500,
    ):
        newdata = newdata.copy() if newdata is not None else {}
        response = newdata.pop(self.model.response.name, None)
        if response is None:
            raise ValueError("No response values provided in newdata.")

        test_data = response
        samples = self.samples

        if subsamples_n:
            key, subkey = jax.random.split(key)
            subsamples = subsample_tree(subkey, self.samples, num_samples=subsamples_n)
            samples = subsamples

        dist = self.model.init_dist(samples, newdata=newdata)
        key, subkey = jax.random.split(key)
        pred_samples = dist.sample(predictive_samples_n, seed=key)
        nsamp, c, s, ntest = pred_samples.shape
        pred_samples = jnp.reshape(pred_samples, shape=(nsamp * c * s, ntest))

        n_inf = jnp.isinf(pred_samples).sum()
        if n_inf > 0:
            logger.warning(
                f"Found {n_inf} infinite values in predictive sample. Changing to NaN."
            )
            pred_samples = pred_samples.at[jnp.where(jnp.isinf(pred_samples))].set(
                jnp.nan
            )

        ntest = test_data.shape[0]
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
