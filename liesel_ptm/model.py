from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from itertools import chain

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import optax
import pandas as pd

from . import nodes as nd
from .bsplines import BSpline
from .custom_types import Array, KeyArray
from .dist import LocScaleTransformationDist
from .ptm_ls import (
    LocationIntercept,
    ScaleInterceptExp,
    _product,
    _sum,
    array_to_dict,
    waic,
)
from .sampling import summarise_by_quantiles, summarise_by_samples

logger = logging.getLogger(__name__)


class NewPTMLocScale:
    def __init__(
        self,
        y: Array,
        coef: nd.PTMCoef | nd.OnionCoefParam,
        centered: bool = False,
        scaled: bool = False,
        knots_class: type | None = None,
    ) -> None:
        self._knots = coef.knots
        self._knots_class = knots_class
        self.coef = coef
        self.order = 3
        self.nparam = coef.log_increments.transformed.value.shape[-1]

        self.loc_model: nd.Predictor = nd.Predictor("loc_model").update()
        """Predictor for the location model part. Does not include an intercept."""
        self.log_scale_model: nd.Predictor = nd.Predictor("log_scale_model").update()
        """Predictor for the log scale model part. Does not include an intercept."""

        self.scale_model = lsl.Calc(jnp.exp, self.log_scale_model).update()
        """The exponential of :attr:`.log_scale_model`."""

        self._y_data = lsl.obs(y, name="_y_data")
        # This has to be an observed value, otherwise liesel.goose.optim_flat will
        # stumble in the batching.

        self.loc_intercept = lsl.Var(
            LocationIntercept(self._y_data, self.loc_model, self.scale_model).update(),
            name="loc_intercept",
        )
        """Intercept :math:`\\beta_0` of the location model part."""

        self.scale_intercept_exp = lsl.Var(
            ScaleInterceptExp(
                self._y_data, self.loc_intercept, self.loc_model, self.scale_model
            ).update(),
            name="scale_intercept_exp",
        )
        r"""
        Exponential of the intercept :math:`\exp(\gamma_0)` of the scale model part.
        """

        self.loc = lsl.Var(
            lsl.Calc(_sum, self.loc_model, self.loc_intercept).update(), name="loc"
        )
        """The response's location, including an intercept."""

        self.scale = lsl.Var(
            lsl.Calc(_product, self.scale_model, self.scale_intercept_exp).update(),
            name="scale",
        )
        """The response's scale, including an intercept."""

        self._dist_centered = centered
        self._dist_scaled = scaled

        self._bspline = BSpline(
            knots=self.knots,
            order=self.order,
            approx=True,
            extrapolate=True,
            target_slope=1.0,
        )

        self._dist_class = partial(
            LocScaleTransformationDist,
            basis_dot_and_deriv_fn=self.bspline.dot_and_deriv,
            centered=self._dist_centered,
            scaled=self._dist_scaled,
            knots=self.knots,
        )

        response_dist = lsl.Dist(
            self.dist_class,
            coef=self.coef,
            loc=self.loc,
            scale=self.scale,
        )

        self.response = lsl.obs(y, response_dist, name="response")

        self.intercept_names = [self.loc_intercept.name, self.scale_intercept_exp.name]
        """Names of the intercept nodes."""

        self._graph: lsl.Model | None = None

    @property
    def knots(self):
        return self._knots

    @knots.setter
    def knots(self, value: Array) -> None:
        self.graph.pop_nodes_and_vars()
        self._knots = value

        self._bspline = BSpline(
            knots=value,
            order=self.bspline.order,
            approx=self.bspline.approx,
            ngrid=self.bspline.ngrid,
            postmultiply_by=self.bspline.postmultiply_by,
            extrapolate=self.bspline.extrapolate,
            eps=self.bspline.eps,
            target_slope=self.bspline.target_slope,
        )

        self._dist_class = partial(
            LocScaleTransformationDist,
            basis_dot_and_deriv_fn=self.bspline.dot_and_deriv,
            centered=self._dist_centered,
            scaled=self._dist_scaled,
            knots=value,
        )

        self.coef.knots = self._knots

        self.response.dist_node.distribution = self.dist_class
        self.response.update()

        self._graph = self._build_graph()

    @property
    def bspline(self) -> BSpline:
        return self._bspline

    @property
    def dist_class(self):
        return self._dist_class

    def _build_graph(self) -> lsl.Model:
        gb = lsl.GraphBuilder().add(self.response)
        self._graph = gb.build_model()
        return self._graph

    def _set_kwargs(self, **kwargs) -> dict:
        for key in self.loc_model.terms.keys():
            if key not in kwargs:
                kwargs[key] = None

        for key in self.log_scale_model.terms.keys():
            if key not in kwargs:
                kwargs[key] = None

        for key, value in kwargs.items():
            loc_key = key in self.loc_model.terms
            scale_key = key in self.log_scale_model.terms

            if not loc_key and not scale_key:
                raise ValueError(f"Key {key} not found in loc or scale model.")

            if value is not None:
                kwargs[key] = jnp.atleast_1d(value)

            if value is None:
                logger.info(f"Fixed value of {key} to corresponding node value.")
                loc_term = self.loc_model.terms.get(key, None)
                scale_term = self.log_scale_model.terms.get(key, None)
                term = loc_term if loc_term is not None else scale_term
                if term is not None:
                    kwargs[key] = term.observed_value

        return kwargs

    def _augment_samples(self, samples: dict[str, Array]) -> dict[str, Array]:
        """
        If there are no samples for a certain parameter, this helper method adds the
        current values of the parameter node from the model graph into the samples.
        """
        augmentations = {}

        for name, var in self.graph.vars.items():
            if not var.parameter:
                continue

            if name in samples:
                continue

            logger.info(f"Fixed value of {name} to corresponding node value.")
            augmentations[name] = var.value

        return samples | augmentations

    @property
    def graph(self) -> lsl.Model:
        if self._graph is None:
            logger.info(
                "Building model graph. No changes to the model can be made after this"
                " step."
            )
            return self._build_graph()
        else:
            return self._graph

    def optimize_locscale(
        self,
        max_iter: int = 1000,
        patience: int = 10,
        atol: float = 0.01,
        include_loc: bool = True,
        include_scale: bool = True,
        include_hyperparameters: bool = False,
        **kwargs,
    ) -> gs.OptimResult:
        params = []  # type: ignore

        if include_loc:
            params += self.loc_model.parameters
            if include_hyperparameters:
                params += self.loc_model.hyper_parameters
        if include_scale:
            params += self.log_scale_model.parameters
            if include_hyperparameters:
                params += self.log_scale_model.hyper_parameters

        if "optimizer" not in kwargs:
            scheduler = optax.cosine_decay_schedule(
                init_value=0.01, decay_steps=max_iter
            )
            optimizer = optax.adam(learning_rate=scheduler)
            kwargs["optimizer"] = optimizer

        logger.info("Optimizing location and scale parameters.")
        stopper = gs.Stopper(max_iter=max_iter, patience=patience, atol=atol)
        model_validation = kwargs.get("model_validation", self.graph)
        result_loc_scale = gs.optim_flat(
            model_train=self.graph,
            model_validation=model_validation,
            params=params,
            stopper=stopper,
            **kwargs,
        )
        logger.info(
            "Optimizing location and scale parameters finished after"
            f" {result_loc_scale.iteration} iterations."
        )
        self.graph.state = result_loc_scale.model_state

        return result_loc_scale

    def updated_ab(
        self, q: tuple[float, float] = (0.0, 1.0), stretch: float = 1.0
    ) -> tuple[float, float]:
        dist = self.init_dist()

        y = self.response.value
        resid = dist.transformation_and_logdet_parametric(y)[0]
        quantiles = jnp.quantile(resid, jnp.array(q))

        range_ = quantiles[1] - quantiles[0]
        mid = quantiles[0] + range_ / 2

        a = float(mid - (range_ / 2) * stretch)
        b = float(mid + (range_ / 2) * stretch)

        return a, b

    def update_knots(
        self,
        q: tuple[float, float] = (0.0, 1.0),
        stretch: float = 1.0,
        knots_class: type | None = None,
    ) -> Array:
        knots_class = knots_class if knots_class is not None else self._knots_class
        if knots_class is None:
            raise ValueError("No knot class available.")

        a, b = self.updated_ab(q=q, stretch=stretch)
        knots = knots_class(a=a, b=b, nparam=self.nparam, order=self.order)
        self.knots = knots.knots

        return knots.knots

    def optimize_transformation(
        self,
        max_iter: int = 1000,
        patience: int = 10,
        atol: float = 0.01,
        include_hyperparameters: bool = False,
        **kwargs,
    ) -> gs.OptimResult:
        params = [self.coef.log_increments.transformed.name]  # type: ignore

        if not isinstance(self.coef.intercept, float):
            params.append(self.coef.intercept.name)

        if not isinstance(self.coef.slope, float):
            slope_param = nd.find_param(self.coef.slope)
            if slope_param is not None:
                params.append(slope_param.name)

        if include_hyperparameters:
            tau2_param = nd.find_param(self.coef.tau2)
            if tau2_param is not None:
                params.append(tau2_param.name)

        logger.info("Optimizing transformation parameters.")
        stopper = gs.Stopper(max_iter=max_iter, patience=patience, atol=atol)

        if "optimizer" not in kwargs:
            scheduler = optax.cosine_decay_schedule(
                init_value=0.01, decay_steps=max_iter
            )
            optimizer = optax.adam(learning_rate=scheduler)
            kwargs["optimizer"] = optimizer

        model_validation = kwargs.get("model_validation", self.graph)
        result_trafo = gs.optim_flat(
            model_train=self.graph,
            model_validation=model_validation,
            params=params,
            stopper=stopper,
            **kwargs,
        )
        logger.info(
            "Optimizing transformation parameters finished after"
            f" {result_trafo.iteration} iterations."
        )
        self.graph.state = result_trafo.model_state

        return result_trafo

    def setup_engine_builder(
        self, seed: int, num_chains: int, sample_transformation: bool = True
    ) -> gs.EngineBuilder:
        eb = gs.EngineBuilder(seed=seed, num_chains=num_chains)

        eb.set_model(gs.LieselInterface(self.graph))
        eb.set_initial_values(self.graph.state)

        loc_terms = self.loc_model.terms.values()
        scale_terms = self.log_scale_model.terms.values()

        for term in chain(loc_terms, scale_terms):
            for kernel in term.mcmc_kernels:
                eb.add_kernel(kernel)

        if sample_transformation:
            for kernel in self.coef.mcmc_kernels:
                eb.add_kernel(kernel)

        eb.positions_included = self.intercept_names

        return eb

    def _predict_loc_intercept(self, samples: dict[str, Array]) -> Array:
        try:
            loc_intercept = samples[self.loc_intercept.name]
        except KeyError:
            kwargs = self._set_kwargs()
            loc_model = self.loc_model.predict(samples, **kwargs)
            scale_model = jnp.exp(self.log_scale_model.predict(samples, **kwargs))
            loc_intercept = LocationIntercept.compute_intercept(
                y=self.response.value, loc_model=loc_model, scale_model=scale_model
            )

        return loc_intercept

    def _predict_scale_intercept(self, samples: dict[str, Array]) -> Array:
        try:
            scale_intercept = samples[self.scale_intercept_exp.name]
        except KeyError:
            kwargs = self._set_kwargs()
            log_scale_model = self.log_scale_model.predict(samples, **kwargs)
            loc_model = self.loc_model.predict(samples, **kwargs)
            loc_intercept = self._predict_loc_intercept(samples)

            scale_intercept = ScaleInterceptExp.compute_intercept(
                y=self.response.value,
                loc_model=loc_model,
                loc_intercept=loc_intercept,
                scale_model=jnp.exp(log_scale_model),
            )

        return scale_intercept

    def predict_loc(self, samples: dict[str, Array], **kwargs) -> Array:
        kwargs = self._set_kwargs(**kwargs)
        samples = self._augment_samples(samples)

        loc_model = self.loc_model.predict(samples, **kwargs)
        loc_intercept = self._predict_loc_intercept(samples)

        return loc_model + jnp.expand_dims(loc_intercept, -1)

    def predict_scale(self, samples: dict[str, Array], **kwargs) -> Array:
        kwargs = self._set_kwargs(**kwargs)
        samples = self._augment_samples(samples)

        log_scale_model = self.log_scale_model.predict(samples, **kwargs)
        log_scale_intercept = jnp.log(self._predict_scale_intercept(samples))

        return jnp.exp(log_scale_model + jnp.expand_dims(log_scale_intercept, -1))

    def init_dist(
        self, samples: dict[str, Array] | None = None, **kwargs
    ) -> LocScaleTransformationDist:
        if samples is None:
            dist = self.dist_class(
                coef=self.coef.value,
                loc=self.loc.value,
                scale=self.scale.value,
            )

        else:
            kwargs = self._set_kwargs(**kwargs)
            samples = self._augment_samples(samples)

            loc = self.predict_loc(samples, **kwargs)
            scale = self.predict_scale(samples, **kwargs)
            coef = self.coef.predict(samples)

            dist = self.dist_class(
                coef=coef,
                loc=loc,
                scale=scale,
            )
        return dist

    def waic(self, samples: dict[str, Array]) -> pd.DataFrame:
        dist = self.init_dist(samples)
        log_prob = dist.log_prob(self.response.value)
        return waic(log_prob_samples=log_prob)

    def summarise_density_by_quantiles(
        self,
        y: Array,
        samples: dict[str, Array],
        axis: Sequence[int] | int = (0, 1),
        lo: float = 0.1,
        hi: float = 0.9,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Returns a summary dataframe to quickly assess the response's conditional
        distribution.
        """
        dist = self.init_dist(samples, **kwargs)
        z, _ = dist.transformation_and_logdet(y)
        cdf = dist.cdf(y)
        pdf = dist.prob(y)

        z_df = summarise_by_quantiles(z, "z", axis=axis, lo=lo, hi=hi)
        pdf_df = summarise_by_quantiles(pdf, "pdf", axis=axis, lo=lo, hi=hi)
        cdf_df = summarise_by_quantiles(cdf, "cdf", axis=axis, lo=lo, hi=hi)

        df = pd.concat([z_df, pdf_df, cdf_df], axis=1)

        df["y"] = y

        for xname, xval in kwargs.items():
            x_dict = array_to_dict(xval, names_prefix=xname)
            for key, value in x_dict.items():
                df[key] = np.asarray(np.squeeze(value))

        return df

    def summarise_density_by_samples(
        self,
        key: KeyArray | int,
        y: Array,
        samples: dict[str, Array],
        n: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        key = jax.random.PRNGKey(key) if isinstance(key, int) else key

        dist = self.init_dist(samples, **kwargs)
        z, _ = dist.transformation_and_logdet(y)
        cdf = dist.cdf(y)
        pdf = dist.prob(y)

        z_df = summarise_by_samples(key, z, "z", n=n)
        cdf_df = summarise_by_samples(key, cdf, "cdf", n=n)
        pdf_df = summarise_by_samples(key, pdf, "pdf", n=n)

        df = pd.concat([z_df.z, cdf_df.cdf, pdf_df.pdf], axis=1)
        df["index"] = z_df.index
        df["obs"] = z_df.obs
        df["chain"] = z_df.chain
        df["sample"] = z_df["sample"]

        df["y"] = np.tile(jnp.squeeze(y), n)

        for xname, xval in kwargs.items():
            xval = jnp.squeeze(xval)
            if jnp.atleast_1d(xval).shape[-1] == 1:
                df[xname] = np.asarray(xval)
            elif len(xval.shape) > 1:
                for k in range(xval.shape[-1]):
                    df[f"{xname}{k}"] = np.asarray(np.tile(xval[:, k], n))
            else:
                df[xname] = np.asarray(np.tile(xval, n))

        return df

    def summarise_transformation_by_samples(
        self,
        key: KeyArray | int,
        residuals: Array,
        samples: dict[str, Array],
        n: int = 100,
    ) -> pd.DataFrame:
        key = jax.random.PRNGKey(key) if isinstance(key, int) else key

        dist = self.init_dist(samples)
        z, _ = dist.transformation_and_logdet_spline(residuals)
        cdf = dist.cdf_spline(residuals)
        pdf = dist.prob_spline(residuals)

        z_df = summarise_by_samples(key, z, "z", n=n)
        cdf_df = summarise_by_samples(key, cdf, "cdf", n=n)
        pdf_df = summarise_by_samples(key, pdf, "pdf", n=n)

        df = pd.concat([z_df.z, cdf_df.cdf, pdf_df.pdf], axis=1)
        df["index"] = z_df.index
        df["obs"] = z_df.obs
        df["chain"] = z_df.chain
        df["sample"] = z_df["sample"]

        df["residual"] = np.tile(jnp.squeeze(residuals), n)

        return df

    def summarise_transformation_by_quantiles(
        self,
        residuals: Array,
        samples: dict[str, Array],
        axis: Sequence[int] | int = (0, 1),
        lo: float = 0.1,
        hi: float = 0.9,
    ) -> pd.DataFrame:
        """
        Returns a summary dataframe to quickly assess the transformation function.
        """
        samples = self._augment_samples(samples)

        coef = self.coef.predict(samples)
        dist = self.dist_class(
            coef=coef,
            loc=0.0,
            scale=1.0,
        )

        z, _ = dist.transformation_and_logdet(residuals)

        cdf = dist.cdf(residuals)
        pdf = dist.prob(residuals)

        z_df = summarise_by_quantiles(z, "z", axis=axis, lo=lo, hi=hi)
        pdf_df = summarise_by_quantiles(pdf, "pdf", axis=axis, lo=lo, hi=hi)
        cdf_df = summarise_by_quantiles(cdf, "cdf", axis=axis, lo=lo, hi=hi)

        df = pd.concat([z_df, pdf_df, cdf_df], axis=1)

        df["residual"] = np.squeeze(residuals)

        return df

    @classmethod
    def new_onion(
        cls,
        y: Array,
        nparam: int,
        tau2: nd.TransformedVar | None = None,
        a: float = -4.0,
        b: float = 4.0,
        centered: bool = False,
        scaled: bool = False,
        trafo_intercept: bool = False,
        trafo_slope: bool = False,
        trafo_combined_kernels: bool = False,
        reparameterize: bool = True,
    ) -> NewPTMLocScale:
        if tau2 is None:
            tau2 = nd.VarInverseGamma(
                value=1.0, concentration=1.0, scale=0.01, name="tau2"
            )

        knots = nd.OnionKnots(a=a, b=b, nparam=nparam)

        trafo_intercept_var = (
            lsl.param(0.0, name="coef_intercept") if trafo_intercept else None
        )
        trafo_slope_var = (
            nd.TransformedVar(1.0, name="coef_slope") if trafo_slope else None
        )

        coef = nd.OnionCoefParam(
            knots=knots,
            tau2=tau2,
            name="coef",
            intercept=trafo_intercept_var,
            slope=trafo_slope_var,
            combined_kernels=trafo_combined_kernels,
            reparameterize=reparameterize,
        )

        return cls(
            y=y, coef=coef, centered=centered, scaled=scaled, knots_class=nd.OnionKnots
        )

    @classmethod
    def new_ptm(
        cls,
        y: Array,
        nparam: int,
        tau2: nd.TransformedVar | None = None,
        a: float = -3.0,
        b: float = 3.0,
        centered: bool = False,
        scaled: bool = False,
        trafo_intercept: bool = False,
        trafo_slope: bool = False,
        trafo_combined_kernels: bool = False,
        reparameterize: bool = True,
    ) -> NewPTMLocScale:
        if tau2 is None:
            tau2 = nd.VarInverseGamma(
                value=1.0, concentration=1.0, scale=0.01, name="tau2"
            )

        knots = nd.EquidistantKnots(a=a, b=b, nparam=nparam)

        trafo_intercept_var = (
            lsl.param(0.0, name="coef_intercept") if trafo_intercept else None
        )
        trafo_slope_var = (
            nd.TransformedVar(1.0, name="coef_slope") if trafo_slope else None
        )

        coef = nd.PTMCoef(
            knots=knots,
            tau2=tau2,
            name="coef",
            intercept=trafo_intercept_var,
            slope=trafo_slope_var,
            combined_kernels=trafo_combined_kernels,
            reparameterize=reparameterize,
        )

        return cls(
            y=y,
            coef=coef,
            centered=centered,
            scaled=scaled,
            knots_class=nd.EquidistantKnots,
        )
