from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import plotnine as p9
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_ptm.gam.roles import Roles as GamRoles

from ..bspline import LogIncKnots, OnionSpline, PTMSpline
from ..cckernel import freeze_parental_submodel
from ..dist import GaussianPseudoTransformationDist, LocScaleTransformationDist
from ..iwls_proposals import (
    CholInfo,
    GaussianLocCholInfo,
    GaussianScaleCholInfo,
    ObservedCholInfoOrIdentity,
    PTMCholInfoFixed,
)
from ..logprob import FlatLogProb
from ..predictor import (
    LocPredictor,
    ScalePredictor,
    SimplePTMPredictor,
    setup_loc_scale,
)
from ..swap_dists import SwapSpec, TemporarilySwapDists, spec_bounded
from ..util.summary import cache_results, summarise_by_samples
from ..var import PTMCoef, ScaleWeibull

Array = Any
KeyArray = Any
InferenceTypes = Any

logger = logging.getLogger(__name__)

HYPERPARAMETER_ROLES = [
    "hyperparam",
    GamRoles.variance_smooth,
    GamRoles.scale_smooth,
]


class PTMDist(lsl.Dist):
    """
    Distribution wrapper that builds a location-scale transformation distribution
    using a spline-based transformation.

    Parameters
    ----------
    knots
        Spline knot sequence.
    loc
        Location variable for the distribution.
    scale
        Scale variable for the distribution.
    shape
        Shape variable for the transformation.
    centered
        If True, use centered parameterization.
    scaled
        If True, use scaled parameterization.
    trafo_lambda
        Controls transition sharpness for the transformation spline.
    bspline
        Which spline variant to use: 'ptm', 'onion', or 'identity'.
    trafo_target_slope
        How to handle tail slopes for the transformation.
    **kwargs
        Forwarded to the parent distribution constructor.

    Attributes
    ----------
    partial_dist_class
        Partial distribution class used to construct per-observation distributions.
    """

    def __init__(
        self,
        knots: Array,
        loc: lsl.Var,
        scale: lsl.Var,
        shape: lsl.Var,
        centered: bool = False,
        scaled: bool = False,
        trafo_lambda: float = 0.1,
        bspline: Literal["ptm", "onion", "identity"] = "ptm",
        trafo_target_slope: Literal["identity", "continue_linearly"] = "identity",
        **kwargs,
    ) -> None:
        match bspline:
            case "ptm":
                continue_linearly = trafo_target_slope == "continue_linearly"
                bspline_inst: PTMSpline | OnionSpline = PTMSpline(
                    knots=knots, eps=trafo_lambda, continue_linearly=continue_linearly
                )

                partial_dist_class = partial(
                    LocScaleTransformationDist,
                    bspline=bspline_inst,
                    centered=centered,
                    scaled=scaled,
                )
            case "onion":
                bspline_inst = OnionSpline(knots)
                partial_dist_class = partial(
                    LocScaleTransformationDist,
                    bspline=bspline_inst,
                    centered=centered,
                    scaled=scaled,
                )
            case "identity":
                partial_dist_class = partial(
                    GaussianPseudoTransformationDist,
                    centered=centered,
                    scaled=scaled,
                )

        self.partial_dist_class = partial_dist_class

        partial_dist_class = partial(self.partial_dist_class, batched=False)

        super().__init__(partial_dist_class, loc=loc, scale=scale, coef=shape, **kwargs)


class LocScalePTM:
    """
    A Penalized Transformation Model for Location and Scale.

    Parameters
    ----------
    response
        Array of response values.
    knots
        Array of equidistant knots. Should correspond to the chosen bspline variant. \
        See :class:`.PTMKnots` and :class:`.OnionKnots`.
    intercepts
        How to handle intercepts in the location and scale model parts.
        The options are:

        - "compute": Intercepts are assumed constant and re-computed any time a value \
           in the location or scale model part changes.
        - "pseudo_sample": Intercepts are assumed constant and re-computed once in \
            every \
           MCMC iteration given the current values of the location and scale model \
           parts.
        - "sample": Intercepts are treated as ordinary parameters and sampled according\
           to their inference specification in the arguments \
           ``loc_intercept_inference`` \
           and ``scale_intercept_inference``. This can lead to identification issues.
        - "constant": Intercepts are kept constant.
        - "sample_mh": Intercepts are sampled with bespoke Metropolis-Hastings \
            proposals. Experimental, undocumented.
    loc_intercept_inference, scale_intercept_inference
        :class:`liesel.goose.MCMCSpec` objects that define MCMC inference for \
        intercepts, if ``intercepts="sample"``.
    centered, scaled
        Whether the transformation distribution should be centered and scaled to \
        negate any side-effects the transformation might have on the location and \
        scale of the response distribution. Can be used with ``intercepts="sample"``.
        See also :class:`.TransformationDist`.
    trafo_lambda
        Parameter controlling the sharpness of transition to tail extrapolation. \
        Is used to compute ``transition_width = eps * (knots[3] - knots[-4])``, \
        where ``transition_width`` indicates the width of the transition interval.
        Relevant only for ``bspline="ptm"``.
    trafo_target_slope
        If "continue_linearly", there is no transition to the identity function. \
        Instead, \
        the spline will continue linearly in the tails with the slope fixed \
        to the slopes at the boundaries of the core interval for left and right \
        extrapolation, respectively. Relevant only for ``bspline="ptm"``.
    bspline
        Which B-spline formulation to use. The option ``"onion"`` is experimental.
    to_float32
        Whether to convert appropriate values in the model to 32-bit floats.

    Attributes
    ----------
    intercepts
        How to handle intercepts in the location and scale model parts.
    knots
        Array of knots.
    response
        Response variable, an instance of :class:`liesel.model.Var`.
    graph
        The model graph, an instance of :class:`liesel.model.Model`. Only available
        after :meth:`.build` has been called.
    to_float32
        Whether to convert appropriate values in the model to 32-bit floats.
    interface
        An instance of :class:`liesel.goose.LieselInterface` representing
        :attr:`.graph`. Only available after :meth:`.build` has been called.
    is_initialized
        Boolean, indicating whether the model as been initialized with posterior
        modes.

    Examples
    --------

    A basic unconditional model::

        import liesel_ptm as ptm
        import jax

        y = jax.random.normal(jax.random.key(0), (50,))

        model = ptm.LocScalePTM.new_ptm(y, a=-4.0, b=4.0, nparam=20)
        results = model.run_mcmc(seed=1, warmup=300, posterior=500)
        samples = results.get_posterior_samples()

        model.plot(samples)

        dist = model.init_dist(samples) # initialize a distribution object

    A basic linear location-scale model::

        import jax
        import liesel_ptm as ptm
        from liesel_ptm import lin, term

        y = jax.random.normal(jax.random.key(0), (50,))
        x = jax.random.uniform(jax.random.key(1), (50,))

        model = ptm.LocScalePTM.new_ptm(y, a=-4.0, b=4.0, nparam=20)

        # location and scale predictors can be filled by adding terms.
        xlin = lin(x, xname="x")
        model.loc += term.f(xlin, fname="s")

        # when adding terms to the scale model part, they are applied additively
        # to the log-level automatically
        model.scale += term.f(xlin, fname="g")

        results = model.run_mcmc(seed=1, warmup=300, posterior=500)
        samples = results.get_posterior_samples()

        model.plot(samples)

    A basic model with one P-spline::

        import jax
        import liesel_ptm as ptm
        from liesel_ptm import term, ps

        y = jax.random.normal(jax.random.key(0), (50,))
        x = jax.random.uniform(jax.random.key(1), (50,))

        model = ptm.LocScalePTM.new_ptm(y, a=-4.0, b=4.0, nparam=20)

        xps = ps(x, nbases=20, xname="x")
        model.loc += term.f_ig(xps, fname="s")

        results = model.run_mcmc(seed=1, warmup=300, posterior=500)
        samples = results.get_posterior_samples()

        model.plot(samples)

    """

    sample_intercepts_under_constant_trafo: bool = True
    """
    If True (default), intercepts are sampled under the assumption of an identity
    transformation. Only takes effect when the init argument ``intercepts=""sample"``
    is used.
    """

    def __init__(
        self,
        response: Array | pd.Series,
        knots: Array,
        intercepts: Literal[
            "compute", "pseudo_sample", "sample", "constant", "sample_mh"
        ] = "pseudo_sample",
        loc_intercept_inference: InferenceTypes = None,
        scale_intercept_inference: InferenceTypes = None,
        centered: bool = False,
        scaled: bool = False,
        trafo_lambda: float = 0.1,
        trafo_target_slope: Literal["identity", "continue_linearly"] = "identity",
        bspline: Literal["ptm", "onion", "identity"] = "ptm",
        to_float32: bool = True,
    ) -> None:
        response_name: str = "response"

        if isinstance(response, pd.Series):
            response = jnp.asarray(response.to_numpy())

        if response.var() <= 0.0:
            response_val = None
        else:
            response_val = lsl.Value(response, _name="_response_value_helper")

        loc, scale = setup_loc_scale(
            loc_intercept=intercepts,
            scale_intercept=intercepts,
            response_name=response_name,
            loc_intercept_inference=loc_intercept_inference,
            scale_intercept_inference=scale_intercept_inference,
            response_value=response_val,
            loc_name="$\\mu$",
            scale_name="$\\sigma$",
        )

        self.intercepts = intercepts
        self._loc = loc
        self._scale = scale
        self._trafo = SimplePTMPredictor.new_sum(name="trafo")
        self.knots = knots

        dist = PTMDist(
            knots=knots,
            loc=loc,
            scale=scale,
            shape=self.trafo,
            centered=centered,
            scaled=scaled,
            trafo_lambda=trafo_lambda,
            bspline=bspline,
            trafo_target_slope=trafo_target_slope,
        )
        self._response_value_helper = response_val
        self.response = lsl.Var.new_obs(response, dist, name=response_name)
        self.graph: lsl.Model | None = None
        self.to_float32 = to_float32
        self.interface: gs.LieselInterface | None = None
        self.is_initialized: bool = False
        self._hyperparameter_initial_values: dict[str, Array] = {}

    @classmethod
    def new_ptm(
        cls,
        response: Array | pd.Series,
        nparam: int = 20,
        a: float = -4.0,
        b: float = 4.0,
        tau2_scale: float = 0.5,
        trafo_lambda: float = 0.1,
        trafo_target_slope: Literal["identity", "continue_linearly"] = "identity",
        to_float32: bool = True,
    ):
        """
        Shortcut for convenient model setup.
        """
        knots = LogIncKnots(a, b, nparam=nparam)

        model = cls(
            response=response,
            knots=knots.knots,
            trafo_lambda=trafo_lambda,
            trafo_target_slope=trafo_target_slope,
            bspline="ptm",
            to_float32=to_float32,
        )

        trafo_scale = ScaleWeibull(
            value=1.0,
            scale=tau2_scale,
            name="$\\tau_\\delta$",
            bijector=tfb.Exp(),
        )
        trafo_scale.variance_param.name = "$\\tau^2_\\delta$"
        trafo_scale.variance_param.value_node[0].name = "$\\log(\\tau^2_\\delta)$"

        trafo0 = PTMCoef.new_rw1_sumzero(
            knots=knots.knots, scale=trafo_scale, name="$\\delta$", noncentered=False
        )
        trafo0.latent_coef.name = "$\\delta_z$"

        model.trafo += trafo0

        return model

    @classmethod
    def new_gaussian(
        cls,
        response: Array | pd.Series,
        loc_intercept_inference: InferenceTypes = gs.MCMCSpec(gs.IWLSKernel),
        scale_intercept_inference: InferenceTypes = gs.MCMCSpec(gs.IWLSKernel),
        to_float32: bool = True,
    ) -> LocScalePTM:
        """
        Shortcut for initializing a Gaussian model.

        Parameters
        ----------
        response
            Array of response values.
        loc_intercept_inference, scale_intercept_inference
            :class:`liesel.goose.MCMCSpec` objects that define MCMC inference for
            intercepts.
        to_float32
            Whether to convert appropriate values in the model to 32-bit floats.
        """
        return cls(
            response=response,
            intercepts="sample",
            loc_intercept_inference=loc_intercept_inference,
            scale_intercept_inference=scale_intercept_inference,
            knots=jnp.linspace(-3.0, 3.0, 10),
            to_float32=to_float32,
            centered=False,
            scaled=False,
            bspline="identity",
        )

    @property
    def loc(self) -> LocPredictor:
        """Location predictor"""
        return self._loc

    @loc.setter
    def loc(self, value):
        if value is not self._loc:
            raise ValueError("Cannot overwrite .loc attribute.")
        self._loc = value

    @property
    def scale(self) -> ScalePredictor:
        """Location predictor"""
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not self._scale:
            raise ValueError("Cannot overwrite .scale attribute.")
        self._scale = value

    @property
    def trafo(self):
        """Predictor for the transformation function."""
        return self._trafo

    @trafo.setter
    def trafo(self, value):
        if value is not self._trafo:
            raise ValueError("Cannot overwrite .trafo attribute.")
        self._trafo = value

    def _set_intercept_inference(self) -> Self:
        li = self.loc.loc_intercept
        si = self.scale.log_scale_intercept

        if li.inference is not None:
            li.inference = freeze_parental_submodel(
                li.inference, of=self.trafo, exclude_roles=["hyperparam"]
            )

        if si.inference is not None:
            si.inference = freeze_parental_submodel(
                si.inference, of=self.trafo, exclude_roles=["hyperparam"]
            )

        return self

    def build(self) -> Self:
        """Build the model graph."""
        if self.graph is not None:
            raise ValueError("Graph was already built.")
        if self.intercepts == "sample" and self.sample_intercepts_under_constant_trafo:
            self._set_intercept_inference()
        vars_: list[lsl.Var | lsl.Node] = [self.response]
        if self._response_value_helper is not None:
            vars_.append(self._response_value_helper)
        self.graph = lsl.Model(vars_, to_float32=self.to_float32)
        self.interface = gs.LieselInterface(self.graph)
        return self

    def optim(
        self,
        exclude_roles: list[Literal["hyperparam", "transformation_coef"] | str]
        | None = None,
        exclude_params: Sequence[str] | None = None,
        swap_pairs: Sequence[SwapSpec] | None = None,
        stopper: gs.Stopper | None = None,
        progress_bar: bool = False,
        test_for_positive_definiteness: bool = False,
        update_parameters_inplace: bool = False,
        **kwargs,
    ) -> gs.OptimResult:
        """
        Optimize model parameters using the selected optimizer.

        Parameters
        ----------
        exclude_roles
            Roles to exclude from optimization.
        exclude_params
            Specific parameter names to exclude.
        swap_pairs
            Swap specifications used to temporarily replace distribution parts during
            optimization.
        stopper
            Optional stopper controlling optimization iterations.
        progress_bar
            Whether to show a progress bar.
        test_for_positive_definiteness
            If True, test Fisher information matrices for positive definiteness.
        update_parameters_inplace
            If True, apply optimized parameters to the model.
            Otherwise restore the previous state after optimization.
        **kwargs
            Forwarded to the underlying optimizer.

        Returns
        -------
        OptimResult
            Result of the optimization run.
        """

        if self.graph is None:
            raise ValueError("Model must be built with .build() first.")

        state_before = self.graph.state

        exclude_roles = exclude_roles if exclude_roles is not None else []
        exclude_params = exclude_params if exclude_params is not None else []

        logger.debug(f"Exlcuding roles: {exclude_roles}")
        logger.debug(f"Exlcuding params: {exclude_params}")

        if swap_pairs is None:
            swap_pairs = []

            hyp = [
                p
                for p in self.graph.parameters.values()
                if p.role in HYPERPARAMETER_ROLES
            ]
            hyp = [p for p in hyp if p.name not in exclude_params]
            for param in hyp:
                logger.debug(f"Setting up temporary bounding for {param.name}.")
                spec = spec_bounded(
                    param,
                    lower_bound=0.05**2,
                    upper_bound=50000.0,
                )
                swap_pairs.append(spec)

        for param in hyp:
            if param not in self._hyperparameter_initial_values:
                logger.debug(
                    f"Saving initial value {param.value} for parameter{param.name}."
                )
                self._hyperparameter_initial_values[param.name] = param.value

        tmp = partial(
            TemporarilySwapDists, pairs=swap_pairs, to_float32=self.to_float32
        )

        with tmp(self.graph) as model_:
            params = [
                var_.name
                for var_ in model_.parameters.values()
                if var_.role not in exclude_roles and var_.name not in exclude_params
            ]

            logger.debug(f"Optimizing params: {params}")
            i = 0
            is_positive_definite = False
            maxi = 5

            while not is_positive_definite and i < maxi:
                logger.debug(f"Optimization step: {i} started")
                result = gs.optim_flat(
                    model_train=model_,
                    params=params,
                    model_validation=model_,
                    stopper=stopper,
                    progress_bar=progress_bar,
                    **kwargs,
                )
                logger.debug(f"Optimization step: {i} done")

                model_.state = result.model_state

                if not test_for_positive_definiteness:
                    break

                positive_definite_tests = []
                for param_name in result.position:
                    logprob = FlatLogProb(
                        model=gs.LieselInterface(model_),
                        model_state=result.model_state,
                        param_names=[param_name],
                    )

                    pos = model_.extract_position(
                        [param_name], model_state=result.model_state
                    )
                    flat_position, _ = jax.flatten_util.ravel_pytree(pos)
                    if len(flat_position) < 2:  # skip scalar parameters
                        continue
                    finfo = -logprob.hessian(
                        flat_position, model_state=result.model_state
                    )

                    augmentation = 1e-5 * jnp.eye(jnp.shape(flat_position)[-1])
                    evals = jnp.linalg.eigvalsh(finfo + augmentation)
                    eps = jnp.finfo(finfo.dtype).eps
                    tol = 100 * eps * jnp.linalg.norm(finfo, ord=jnp.inf)
                    is_pd = jnp.all(evals > tol)
                    positive_definite_tests.append(is_pd)

                    logger.debug(f"Tested parameter: {param_name}")
                    logger.debug(f"Positive definite Fisher info: {is_pd}")

                is_positive_definite = all(positive_definite_tests)
                logger.debug(
                    "Number of positive definitive "
                    f"Fisher Infos: {sum(positive_definite_tests)}"
                )
                logger.debug(
                    f"All Fisher Infos positive definite: {is_positive_definite}"
                )
                i += 1
                logger.debug("")

        if not update_parameters_inplace:
            self.graph.state = state_before

        return result

    def initialize(
        self,
        exclude_roles: list[Literal["hyperparam", "transformation_coef"] | str]
        | None = None,
        stopper: gs.Stopper | None = None,
        test_for_positive_definiteness: bool = False,
        **kwargs,
    ) -> tuple[gs.OptimResult, gs.OptimResult]:
        """
        Two-stage initialization that fits loc-scale, then transformation.

        Updates the model state with the resulting estimated parameters.

        Parameters
        ----------
        exclude_roles
            Roles to exclude from optimization.
        stopper
            Optional stopper controlling optimization iterations.
        test_for_positive_definiteness
            If True, check Fisher information matrices for positive definiteness.
        **kwargs
            Forwarded to the underlying optimization routine.

        Returns
        -------
        tuple
            Optimization results for (loc-scale, transformation).
        """

        if self.graph is None:
            raise ValueError("Model must be built with .build() first.")
        # step 1: fit only location-scale part
        trafo_submodel = self.graph.parental_submodel(self.trafo)
        trafo_params = list(trafo_submodel.parameters)
        trafo_params_transformed = [n + "_transformed" for n in trafo_params]

        result1 = self.optim(
            exclude_roles=exclude_roles,
            exclude_params=trafo_params + trafo_params_transformed,
            stopper=stopper,
            test_for_positive_definiteness=test_for_positive_definiteness,
            update_parameters_inplace=True,
            **kwargs,
        )

        # step 2: fit only transformation part
        locscale_params = [
            name for name in list(self.graph.parameters) if name not in trafo_params
        ]
        locscale_params_transformed = [n + "_transformed" for n in locscale_params]

        result2 = self.optim(
            exclude_roles=exclude_roles,
            exclude_params=locscale_params + locscale_params_transformed,
            stopper=stopper,
            test_for_positive_definiteness=test_for_positive_definiteness,
            update_parameters_inplace=True,
            **kwargs,
        )

        self.is_initialized = True

        # return both results
        return result1, result2

    def init_dist(
        self,
        samples: dict[str, Array],
        loc: Array | None = None,
        scale: Array | None = None,
        newdata: dict[str, Array] | None = None,
    ) -> LocScaleTransformationDist:
        """
        Construct a batched distribution from posterior samples.

        Parameters
        ----------
        samples
            Posterior samples dict used to build the distribution.
        loc, scale
            Optional explicit loc/scale arrays; if provided, `newdata` is ignored.
        newdata
            Optional newdata for prediction when loc/scale are not provided.

        Returns
        -------
        LocScaleTransformationDist
            A batched transformation distribution for prediction.
        """

        if not self.graph:
            raise ValueError("Model must be built with .build() first.")

        if self.is_gaussian:
            # assuming no samples are present, just use current values.
            trafo_samples = self.trafo.value
        else:
            trafo_samples = self.trafo.predict(samples)

        if loc is not None and newdata is not None:
            raise ValueError("If loc is not None, newdata is not used.")

        if scale is not None and newdata is not None:
            raise ValueError("If scale is not None, newdata is not used.")

        if loc is None or scale is None:
            locscale = self.graph.predict(
                samples, predict=[self.loc.name, self.scale.name], newdata=newdata
            )

            loc_ = locscale[self.loc.name] if loc is None else loc

            scale_ = locscale[self.scale.name] if scale is None else scale
        else:
            loc_ = loc
            scale_ = scale

        if trafo_samples.ndim > 0:
            # protection  for the Gaussian case, when trafo_samples is 0.0 (scalar)
            trafo_samples = jnp.expand_dims(trafo_samples, -2)

        loc_ = jnp.asarray(loc_)
        scale_ = jnp.asarray(scale_)

        ndim = max(loc_.ndim, scale_.ndim, (trafo_samples.ndim - 1))

        if loc_.ndim < ndim:
            loc_ = jnp.expand_dims(loc_, -1)
        if scale_.ndim < ndim:
            scale_ = jnp.expand_dims(scale_, -1)

        return self.response.dist_node.partial_dist_class(  # type: ignore
            loc=loc_, scale=scale_, coef=trafo_samples, batched=True
        )

    def summarise_dist(
        self,
        samples: dict[str, Array],
        loc: Array | None = None,
        scale: Array | None = None,
        grid: Array | None = None,
        newdata: dict[str, Array] | None = None,
    ) -> dict[str, Array]:
        """
        Return summary arrays (z, prob, log_prob, cdf) for a grid of values.

        Parameters
        ----------
        samples
            Posterior samples dict used to build the distribution.
        loc, scale
            Optional loc/scale arrays overriding predictions.
        grid
            Points to evaluate; if None, uses observed response values.
        newdata
            Optional newdata for prediction when loc/scale are None.

        Returns
        -------
        dict
            Keys: 'z', 'prob', 'log_prob', 'cdf' with arrays over samples.
        """

        grid_ = grid if grid is not None else self.response.value
        dist = self.init_dist(samples, loc=loc, scale=scale, newdata=newdata)

        z_samples, _ = dist.transformation_and_logdet(grid_)
        log_prob_samples = dist.log_prob(grid_)
        prob_samples = dist.prob(grid_)
        cdf_samples = dist.cdf(grid_)

        return {
            "z": z_samples,
            "prob": prob_samples,
            "log_prob": log_prob_samples,
            "cdf": cdf_samples,
        }

    def summarise_trafo_by_samples(
        self,
        key: KeyArray | int,
        grid: Array,
        samples: dict[str, Array],
        n: int = 100,
    ) -> pd.DataFrame:
        """
        Summarise transformation samples on a grid.

        Parameters
        ----------
        key
            PRNG key or integer seed used to subsample trajectories.
        grid
            Points at which the transformation is evaluated.
        samples
            Posterior samples dictionary used to build the distribution.
        n
            Number of sampled trajectories to return.

        Returns
        -------
        DataFrame
            DataFrame with sampled trajectories and plotting metadata.
        """

        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        dist = self.init_dist(samples, loc=0.0, scale=1.0)

        z_samples, _ = dist.transformation_and_logdet(grid)
        pdf_samples = jnp.exp(dist.log_prob(grid))
        cdf_samples = dist.cdf(grid)

        z_df = summarise_by_samples(key, z_samples, "z", n=n)
        cdf_df = summarise_by_samples(key, cdf_samples, "cdf", n=n)
        pdf_df = summarise_by_samples(key, pdf_samples, "pdf", n=n)

        df = pd.concat([z_df.z, cdf_df.cdf, pdf_df.pdf], axis=1)
        df["index"] = z_df.index
        df["obs"] = z_df.obs
        df["chain"] = z_df.chain
        df["sample"] = z_df["sample"]

        df["r"] = np.tile(np.squeeze(grid), n)

        return df

    @property
    def is_gaussian(self) -> bool:
        """Whether the model is Gaussian."""
        return len(list(self.trafo.terms.values())) == 0

    def setup_default_mcmc_kernels(
        self,
        strategy: Literal[
            "iwls_fixed", "iwls_fixed-nuts", "nuts", "iwls-nuts", "iwls-iwls_fixed"
        ] = "iwls_fixed",
        use_fallback_finfos: bool = True,
        locscale_kernel_kwargs: dict[str, Any] | None = None,
        trafo_kernel_kwargs_nuts: dict[str, Any] | None = None,
        trafo_kernel_kwargs_iwls: dict[str, Any] | None = None,
        override_existing_inference_on_locscale: bool = False,
        override_existing_inference_on_trafo: bool = False,
        jitter_dist: tfd.Distribution | None = None,
    ) -> LocScalePTM:
        """
        Configure default MCMC kernels for model parts.

        Parameters
        ----------
        strategy
            Kernel selection strategy for loc/scale/trafo parts. The strategies are:

            - "iwls_fixed": Metropolis-Hastings with proposals generated according to
               the iteratively re-weighted least squares kernel. The Fisher information
               matrices used here are fixed to the observed Fisher information
               matrices at initial estimates of the posterior modes.
            - "iwls_fixed-nuts": Uses "iwls_fixed" in the location and scale model parts
               and a No-U-turn sampler (NUTS) for the parameters of the transformation
               function.
            - "nuts": Uses NUTS for location, scale, and transformation. Does not
               scale well with increasing sample size.
            - "iwls-nuts": Like "iwls_fixed-nuts", but uses expected Fisher information
               matrices derived under a Gaussian assumption for the response as an
               approximation to generate proposals.

            None of the strategies set up MCMC kernels for hyperparameters like
            smoothing parameters in the location and scale model parts, these should
            be specified manually.

            If the random walk variance of the transformation function is transformed
            to the real line with a bijector, the default scheme will always set up a
            NUTS kernel for this parameter.

        use_fallback_finfos
            If True, fall back to Gaussian Fisher information matrices in
            location and scale model parts when needed.
        locscale_kernel_kwargs, trafo_kernel_kwargs
            Optional keyword arguments forwarded to kernel constructors.
        override_existing_inference_on_locscale, override_existing_inference_on_trafo
            If True, existing inference specifications are overridden.

        Returns
        -------
        The model with inference specs set up.
        """
        jitter_dist = (
            tfd.Normal(loc=0.0, scale=1.0) if jitter_dist is None else jitter_dist
        )
        if strategy not in [
            "iwls_fixed",
            "iwls_fixed-nuts",
            "nuts",
            "iwls-nuts",
            "iwls-iwls_fixed",
        ]:
            raise ValueError(f"Unknown strategy {strategy=}.")

        if "iwls_fixed" in strategy and not self.is_initialized:
            raise RuntimeError(
                f"Cannot setup MCMC scheme '{strategy}' if the model is not "
                "initialized. Please run .initialize() first."
            )

        locscale_kernel_kwargs = (
            locscale_kernel_kwargs if locscale_kernel_kwargs is not None else {}
        )
        trafo_kernel_kwargs_nuts = (
            trafo_kernel_kwargs_nuts if trafo_kernel_kwargs_nuts is not None else {}
        )
        trafo_kernel_kwargs_iwls = (
            trafo_kernel_kwargs_iwls if trafo_kernel_kwargs_iwls is not None else {}
        )

        iwls_locscale_kernel_kwargs = (
            locscale_kernel_kwargs
            if locscale_kernel_kwargs
            else {
                "initial_step_size": 1.0,
                "da_target_accept": 0.5,
            }
        )

        for term in self.loc.terms.values():
            if term is self.loc.loc_intercept and self.loc.loc_intercept is not None:
                continue

            if (
                not override_existing_inference_on_locscale
                and term.coef.inference is not None
            ):
                logger.debug(
                    f"Did not set up kernel for {term.coef.name}, "
                    f"because {term.coef.inference=} is not None."
                )
                continue

            if strategy == "nuts":
                term.coef.inference = gs.MCMCSpec(
                    gs.NUTSKernel,
                    kernel_kwargs=locscale_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(f"Set up NUTS kernel for {term.coef.name}")
                continue

            # iwls cases
            if self.interface is None:
                raise ValueError(f"{self.interface=} must not be None.")
            if self.is_gaussian or strategy in ["iwls-nuts", "iwls-iwls_fixed"]:
                # this fails if term is a linear term or for another reason has
                # no .scale attribute.
                try:
                    cinfo_class = GaussianLocCholInfo
                    cinfo = cinfo_class.from_smooth(
                        term, model=self.interface, n=int(self.response.value.size)
                    )
                except AttributeError:
                    # this will fall back to an ordinary IWLS kernel using the observed
                    # fisher information
                    # But if there are NaNs in the fisher information's Choleksy
                    # decomposition, it will use an identity matrix instead.
                    cinfo = ObservedCholInfoOrIdentity.from_smooth(term, self.interface)

                term.coef.inference = gs.MCMCSpec(
                    gs.IWLSKernel,
                    kernel_kwargs={"chol_info_fn": cinfo.chol_info}
                    | iwls_locscale_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(f"Set up IWLS kernel for {term.coef.name}")
                continue
            else:
                cinfo = CholInfo.from_smooth(term, model=self.interface)

            if cinfo.nan_in_cholesky_of_unprocessed_finfo:
                if use_fallback_finfos:
                    try:
                        logger.warning(
                            "NaNs von in the Cholesky decomposition of the "
                            "Fisher information matrix at current values for "
                            f"term {term.name}. Falling back to the expected Fisher "
                            "information of a Gaussian model. This can lead to less "
                            "efficient sampling. Consider running "
                            ".initialize() for longer."
                        )
                        cinfo = GaussianLocCholInfo.from_smooth(
                            term, model=self.interface, n=int(self.response.value.size)
                        )
                    except Exception:
                        logger.exception("Failed to use Gaussian fallback. Continuing.")

                logger.warning(
                    "NaNs von in the Cholesky decomposition of the "
                    "Fisher information matrix at current values for "
                    f"term {term.name}. Falling back to an augmented Fisher "
                    "information to ensure positive definitness. This is likely "
                    "to lead to less efficient sampling. Consider running "
                    ".initialize() for longer."
                )

            term.coef.inference = gs.MCMCSpec(
                gs.IWLSKernel,
                kernel_kwargs={"chol_info_fn": cinfo.chol_info}
                | iwls_locscale_kernel_kwargs,
                jitter_dist=jitter_dist,
            )
            logger.debug(f"Set up IWLS kernel for {term.coef.name}")

        for term in self.scale.terms.values():
            if (
                term is self.scale.log_scale_intercept
                and self.scale.log_scale_intercept is not None
            ):
                continue

            if (
                not override_existing_inference_on_locscale
                and term.coef.inference is not None
            ):
                logger.debug(
                    f"Did not set up kernel for {term.coef.name}, "
                    f"because {term.coef.inference=} is not None."
                )
                continue

            if strategy == "nuts":
                term.coef.inference = gs.MCMCSpec(
                    gs.NUTSKernel,
                    kernel_kwargs=locscale_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(f"Set up NUTS kernel for {term.coef.name}")
                continue

            # iwls cases
            if self.interface is None:
                raise ValueError(f"{self.interface=} must not be None.")
            if self.is_gaussian or strategy == "iwls-nuts":
                # this fails if term is a linear term or for another reason has
                # no .scale attribute.
                try:
                    cinfo = GaussianScaleCholInfo.from_smooth(
                        term, model=self.interface, n=int(self.response.value.size)
                    )
                except AttributeError:
                    # this will fall back to an ordinary IWLS kernel using the observed
                    # fisher information
                    # But if there are NaNs in the fisher information's Choleksy
                    # decomposition, it will use an identity matrix instead.
                    cinfo = ObservedCholInfoOrIdentity.from_smooth(term, self.interface)

                term.coef.inference = gs.MCMCSpec(
                    gs.IWLSKernel,
                    kernel_kwargs={"chol_info_fn": cinfo.chol_info}
                    | iwls_locscale_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(f"Set up IWLS kernel for {term.coef.name}")
                continue
            else:
                cinfo = CholInfo.from_smooth(term, model=self.interface)

            if cinfo.nan_in_cholesky_of_unprocessed_finfo:
                if use_fallback_finfos:
                    try:
                        logger.warning(
                            "NaNs von in the Cholesky decomposition of the "
                            "Fisher information matrix at current values for "
                            f"term {term.name}. Falling back to the expected Fisher "
                            "information of a Gaussian model. This can lead to less "
                            "efficient sampling. Consider running "
                            ".initialize() for longer."
                        )
                        cinfo = GaussianScaleCholInfo.from_smooth(
                            term, model=self.interface, n=int(self.response.value.size)
                        )
                    except Exception:
                        logger.exception("Failed to use Gaussian fallback. Continuing.")

                logger.warning(
                    "NaNs von in the Cholesky decomposition of the "
                    "Fisher information matrix at current values for "
                    f"term {term.name}. Falling back to an augmented Fisher "
                    "information to ensure positive definitness. This is likely "
                    "to lead to less efficient sampling. Consider running "
                    ".initialize() for longer."
                )

            term.coef.inference = gs.MCMCSpec(
                gs.IWLSKernel,
                kernel_kwargs={"chol_info_fn": cinfo.chol_info}
                | iwls_locscale_kernel_kwargs,
                jitter_dist=jitter_dist,
            )
            logger.debug(f"Set up IWLS kernel for {term.coef.name}")

        trafo_terms = list(self.trafo.terms.values())
        if not trafo_terms:
            return self

        trafo_coef = list(self.trafo.terms.values())[0]
        trafo_scale = trafo_coef.scale  # type: ignore
        trafo_var = trafo_scale.variance_param  # type: ignore

        if trafo_kernel_kwargs_nuts:
            nuts_kernel_kwargs = trafo_kernel_kwargs_nuts
        else:
            nuts_kernel_kwargs = {
                "da_target_accept": 0.9,
                "mm_diag": False,
                "max_treedepth": 10,
            }
        if trafo_kernel_kwargs_iwls:
            iwls_kernel_kwargs = trafo_kernel_kwargs_iwls
        else:
            iwls_kernel_kwargs = {
                "initial_step_size": 1.0,
                "da_target_accept": 0.5,
            }

        if trafo_var.weak:
            if (
                trafo_var.value_node[0].inference is None
                or override_existing_inference_on_trafo
            ):
                trafo_var.value_node[0].inference = gs.MCMCSpec(
                    gs.NUTSKernel,
                    kernel_group="trafo",
                    kernel_kwargs=nuts_kernel_kwargs,
                )
                logger.debug(f"Set up NUTS kernel for {trafo_var.value_node[0].name}")
            else:
                logger.debug(
                    f"Did not set up kernel for {trafo_var.value_node[0].name}, "
                    f"because {trafo_var.value_node[0].inference=} is not None."
                )
        else:
            logger.warning(
                f"Did not set up kernel for {trafo_var.name}, "
                f"because it is strong. It has inference {trafo_var.inference=}."
            )

        if (
            trafo_coef.inference is not None
            and not override_existing_inference_on_trafo
        ):
            logger.debug(
                f"Did not set up kernel for {trafo_coef.name}, "
                f"because {trafo_coef.inference=} is not None."
            )
            return self

        match strategy:
            case "iwls_fixed" | "iwls-iwls_fixed":
                if self.interface is None:
                    raise ValueError(f"{self.interface=} must not be None.")
                cinfo = PTMCholInfoFixed.from_coef(
                    coef=trafo_coef,
                    model=self.interface,  # type: ignore
                )
                if cinfo.nan_in_cholesky_of_unprocessed_finfo:
                    logger.warning(
                        "NaNs von in the Cholesky decomposition of the "
                        "Fisher information matrix at current values for "
                        f"term {trafo_coef.name}. Falling back to an augmented Fisher "
                        "information to ensure positive definitness. This is likely "
                        "to lead to less efficient sampling. Consider running "
                        ".initialize() for longer."
                    )

                trafo_coef.latent_coef.inference = gs.MCMCSpec(
                    gs.IWLSKernel,
                    kernel_kwargs={"chol_info_fn": cinfo.chol_info}
                    | iwls_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(
                    "Set up ILWS kernel with fixed Fisher "
                    f"information for {trafo_coef.name}"
                )

            case "iwls_fixed-nuts":
                trafo_coef.latent_coef.inference = gs.MCMCSpec(
                    gs.NUTSKernel,
                    kernel_group="trafo",
                    kernel_kwargs=nuts_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(f"Set up NUTS kernel for {trafo_coef.name}")

            case "nuts":
                trafo_coef.latent_coef.inference = gs.MCMCSpec(
                    gs.NUTSKernel,
                    kernel_group="trafo",
                    kernel_kwargs=nuts_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(f"Set up NUTS kernel for {trafo_coef.name}")

            case "iwls-nuts":
                trafo_coef.latent_coef.inference = gs.MCMCSpec(
                    gs.NUTSKernel,
                    kernel_group="trafo",
                    kernel_kwargs=nuts_kernel_kwargs,
                    jitter_dist=jitter_dist,
                )
                logger.debug(f"Set up NUTS kernel for {trafo_coef.name}")

        self.show_mcmc()

        return self

    def show_mcmc(self) -> None:
        """
        Logs the current MCMC configuration. If you do not see any output, you need
        to set up logging::

            import logging

            logger = logging.getLogger("liesel_ptm")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

        """
        if self.graph is None:
            raise ValueError("Model must be built with .build() first.")

        for param in self.graph.parameters.values():
            spec = param.inference
            if spec is None:
                logger.info(f"MCMC Setup for {param.name}: None.")
                continue
            if not isinstance(spec, gs.MCMCSpec):
                logger.info(f"Inference for {param.name}: {spec}.")
                continue

            logger.info(
                f"MCMC Setup for {param.name}: {spec.kernel} in "
                f"group '{spec.kernel_group}'."
            )

    def run_mcmc(
        self,
        seed: int,
        warmup: int,
        posterior: int,
        num_chains: int = 4,
        fast_warmup: float = 0.5,
        thinning_posterior: int = 1,
        thinning_warmup: int = 1,
        warm_start: bool = True,
        which: str | None = None,
        strategy: Literal[
            "iwls-nuts",
            "iwls_fixed",
            "iwls_fixed-nuts",
            "nuts",
            "iwls-iwls_fixed",
            "manual",
        ] = "iwls-nuts",
        cache_path: str | Path | None = None,
        apply_jitter: bool = False,
        **initialization_kwargs,
    ) -> gs.SamplingResults:
        """Run MCMC sampling and return sampling results.

        Parameters
        ----------
        seed, warmup, posterior
            MCMC scheduling parameters: seed and durations.
        num_chains
            Number of parallel chains to run.
        strategy
            Which kernel strategy to use for sampling. See
            :meth:`.setup_default_mcmc_kernels`.
        cache_path
            If provided, load/save cached sampling results.
        apply_jitter
            Whether to apply initial jitter to chain initialisations. Only has an
            effect if jittering is specified in the :class:`liesel.goose.MCMCSpec`
            for any one variable. Think of this rather as an off-switch than an
            on-switch.
        warm_start
            If True, the model will be initialized by finding posterior modes via
            :meth:`.initialize`.

        **initialization_kwargs
            Forwarded to :meth:`.initialize` when ``warm_start`` is True.

        Returns
        -------
        SamplingResults
            The :class:`liesel.goose.SamplingResults` sampling results object
            containing chains and diagnostics.
        """

        if self.graph is None:
            self.build()

        if cache_path is not None:
            fp = Path(cache_path)
            if fp.exists():
                return gs.engine.SamplingResults.pkl_load(fp)

        if warm_start:
            self.initialize(**initialization_kwargs)

        if strategy != "manual":
            self.setup_default_mcmc_kernels(strategy=strategy)

        if apply_jitter:
            for param_name in self._hyperparameter_initial_values:
                old = self._hyperparameter_initial_values[param_name]
                logger.debug(f"Restoring initial value for '{param_name}' to {old}.")
                if self.graph is None:
                    raise ValueError("Must build graph first")
                current = self.graph.vars[param_name].value
                self.graph.vars[param_name].value = jnp.array(
                    old, dtype=jnp.asarray(current).dtype
                )

        eb = gs.LieselMCMC(self.graph, which=which).get_engine_builder(  # type: ignore
            seed=seed, num_chains=num_chains, apply_jitter=apply_jitter
        )
        if self.interface is None:
            raise ValueError(f"{self.interface=} must not be None.")
        eb.set_model(self.interface)

        fast_warmup_duration = fast_warmup * warmup
        init_duration = int(fast_warmup_duration / 2)
        term_duration = init_duration

        slow_warmup_duration = warmup - init_duration - term_duration
        warmup = slow_warmup_duration + init_duration + term_duration

        epochs = gs.stan_epochs(
            warmup_duration=warmup,
            posterior_duration=posterior,
            thinning_posterior=thinning_posterior,
            thinning_warmup=thinning_warmup,
            init_duration=init_duration,
            term_duration=term_duration,
        )
        eb.set_epochs(epochs)

        if cache_path is not None:
            results = cache_results(eb, filename=cache_path)
        else:
            engine = eb.build()
            engine.sample_all_epochs()
            results = engine.get_results()

        return results

    def plot_qq(
        self,
        samples: dict[str, Array],
    ) -> p9.ggplot:
        """Produce a QQ-plot comparing transformed r and its Gaussian reference.

        Parameters
        ----------
        samples
            Posterior samples dict used to build the distribution.

        Returns
        -------
        ggplot
            A ggplot object with QQ comparison of r and h(r) where applicable.
        """

        dist = self.init_dist(samples)

        r_samples, _ = dist.transformation_and_logdet_parametric(self.response.value)
        z_samples, _ = dist.transformation_and_logdet_spline(r_samples)

        r_summary = gs.SamplesSummary.from_array(r_samples).to_dataframe()
        r_summary["variable"] = "r"

        if jnp.asarray(self.trafo.value).ndim == 0:
            summary = r_summary
        else:
            z_summary = gs.SamplesSummary.from_array(z_samples).to_dataframe()
            z_summary["variable"] = "h(r)"
            summary = pd.concat((r_summary, z_summary), axis=0)

        p = (
            p9.ggplot()
            + p9.geom_abline(color="black")
            + p9.geom_qq(p9.aes(sample="mean", color="variable"), data=summary)
            # + p9.geom_rug(p9.aes(x=self.knots), sides="b")
            + p9.labs(
                title="QQ plot of posterior average r and h(r)",
                x="Theoretical Quantile",
                y="Observed Quantile",
            )
        )

        return p

    def plot_trafo(
        self,
        samples: dict[str, Array],
        grid: Array | None = None,
        ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
        hdi_prob: float | None = None,
        show_n_samples: int | None = 50,
        seed: int | KeyArray = 1,
    ) -> p9.ggplot:
        """Plot the posterior mean and credible bands of the transformation h(r).

        Parameters
        ----------
        samples
            Posterior samples dict used to build the distribution.
        grid
            Points at which to evaluate h(r); if None, uses a grid over responses.
        ci_quantiles
            Credible interval quantiles to display as ribbon.
        hdi_prob
            Optional highest-density-interval probability to annotate.
        show_n_samples
            Number of sampled trajectories to overlay.
        seed
            RNG seed for subsampling trajectories.

        Returns
        -------
        ggplot
            A ggplot object of the transformation with credible bands.
        """
        dist = self.init_dist(samples)
        r_train = dist.transformation_and_logdet_parametric(self.response.value)[0]

        grid_ = (
            grid
            if grid is not None
            else jnp.linspace(min(r_train.min(), -4.0), max(r_train.max(), 4.0), 300)
        )
        dist = self.init_dist(samples, loc=0.0, scale=1.0)
        z_samples, _ = dist.transformation_and_logdet(grid_)
        while z_samples.ndim < 3:
            z_samples = jnp.expand_dims(z_samples, 0)

        ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
        hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
        z_summary = gs.SamplesSummary.from_array(
            z_samples, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        ).to_dataframe()
        z_summary["r"] = grid_

        p = (
            p9.ggplot()
            + p9.labs(
                title="Transformation function h(r)",
                subtitle="Dotted: Identity function for reference",
                x="r",
                y="h(r)",
            )
            + p9.geom_abline(linetype="dotted")
        )

        if ci_quantiles is not None:
            p = p + p9.geom_ribbon(
                p9.aes(
                    "r",
                    ymin=f"q_{str(ci_quantiles[0])}",
                    ymax=f"q_{str(ci_quantiles[1])}",
                ),
                fill="#56B4E9",
                alpha=0.5,
                data=z_summary,
            )

        if hdi_prob is not None:
            p = p + p9.geom_line(
                p9.aes("r", "hdi_low"),
                linetype="dashed",
                data=z_summary,
            )

            p = p + p9.geom_line(
                p9.aes("r", "hdi_high"),
                linetype="dashed",
                data=z_summary,
            )

        if jnp.asarray(self.trafo.value).ndim == 0:
            show_n_samples = 0

        if show_n_samples is not None and show_n_samples > 0:
            key = jax.random.key(seed) if isinstance(seed, int) else seed
            summary_samples_df = self.summarise_trafo_by_samples(
                key=key, grid=grid_, samples=samples, n=show_n_samples
            )

            p = p + p9.geom_line(
                p9.aes("r", "z", group="sample"),
                color="grey",
                data=summary_samples_df,
                alpha=0.3,
            )

        p = p + p9.geom_line(
            p9.aes("r", "mean"), data=z_summary, size=1.3, color="blue"
        )

        return p

    def plot_r_density(
        self,
        samples: dict[str, Array],
        grid: Array | None = None,
        ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
        hdi_prob: float | None = None,
        show_n_samples: int | None = 50,
        seed: int | KeyArray = 1,
    ) -> p9.ggplot:
        """Plot the posterior density of the transformed variable r.

        Parameters
        ----------
        samples
            Posterior samples dict used to build the distribution.
        grid
            Points at which to evaluate the density; if None, uses a response grid.
        ci_quantiles
            Credible interval quantiles for ribbons.
        hdi_prob
            Optional HDI probability to annotate.
        show_n_samples
            Number of sampled densities to overlay.
        seed
            RNG seed for subsampling trajectories.

        Returns
        -------
        ggplot
            A ggplot object of the density with credible bands.
        """

        dist = self.init_dist(samples)
        r_train = dist.transformation_and_logdet_parametric(self.response.value)[0]

        grid_ = (
            grid
            if grid is not None
            else jnp.linspace(min(r_train.min(), -4.0), max(r_train.max(), 4.0), 300)
        )
        dist = self.init_dist(samples, loc=0.0, scale=1.0)
        prob_samples = dist.prob(grid_)
        while prob_samples.ndim < 3:
            prob_samples = jnp.expand_dims(prob_samples, 0)

        ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
        hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
        prob_summary = gs.SamplesSummary.from_array(
            prob_samples, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        ).to_dataframe()
        prob_summary["r"] = grid_

        p = p9.ggplot() + p9.labs(
            title="Transformation density $f_R(r)$",
            subtitle="Dotted: Standard Gaussian PDF for reference",
            x="r",
            y="$f_R(r)$",
        )

        pdf_norm = tfd.Normal(loc=0.0, scale=1.0).prob(grid_)
        p = p + p9.geom_line(p9.aes(grid_, pdf_norm), linetype="dotted")

        if ci_quantiles is not None:
            p = p + p9.geom_ribbon(
                p9.aes(
                    "r",
                    ymin=f"q_{str(ci_quantiles[0])}",
                    ymax=f"q_{str(ci_quantiles[1])}",
                ),
                fill="#56B4E9",
                alpha=0.5,
                data=prob_summary,
            )

        if hdi_prob is not None:
            p = p + p9.geom_line(
                p9.aes("r", "hdi_low"),
                linetype="dashed",
                data=prob_summary,
            )

            p = p + p9.geom_line(
                p9.aes("r", "hdi_high"),
                linetype="dashed",
                data=prob_summary,
            )

        if jnp.asarray(self.trafo.value).ndim == 0:
            show_n_samples = 0

        if show_n_samples is not None and show_n_samples > 0:
            key = jax.random.key(seed) if isinstance(seed, int) else seed
            summary_samples_df = self.summarise_trafo_by_samples(
                key=key, grid=grid_, samples=samples, n=show_n_samples
            )

            p = p + p9.geom_line(
                p9.aes("r", "pdf", group="sample"),
                color="grey",
                data=summary_samples_df,
                alpha=0.3,
            )

        p = p + p9.geom_line(
            p9.aes("r", "mean"), data=prob_summary, size=1.3, color="blue"
        )

        return p

    def plot_r_cdf(
        self,
        samples: dict[str, Array],
        grid: Array | None = None,
        ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
        hdi_prob: float | None = None,
        show_n_samples: int | None = 50,
        seed: int | KeyArray = 1,
    ) -> p9.ggplot:
        """Plot the posterior CDF of the transformed variable r.

        Parameters
        ----------
        samples
            Posterior samples dict used to build the distribution.
        grid
            Points at which to evaluate the CDF; if None, uses a response grid.
        ci_quantiles
            Credible interval quantiles for ribbons.
        hdi_prob
            Optional HDI probability to annotate.
        show_n_samples
            Number of sampled CDF trajectories to overlay.
        seed
            RNG seed for subsampling trajectories.

        Returns
        -------
        ggplot
            A ggplot object of the CDF with credible bands.
        """
        dist = self.init_dist(samples)
        r_train = dist.transformation_and_logdet_parametric(self.response.value)[0]
        grid_ = (
            grid
            if grid is not None
            else jnp.linspace(min(r_train.min(), -4.0), max(r_train.max(), 4.0), 300)
        )
        dist = self.init_dist(samples, loc=0.0, scale=1.0)
        cdf_samples = dist.cdf(grid_)
        while cdf_samples.ndim < 3:
            cdf_samples = jnp.expand_dims(cdf_samples, 0)

        ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
        hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
        cdf_summary = gs.SamplesSummary.from_array(
            cdf_samples, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        ).to_dataframe()
        cdf_summary["r"] = grid_

        p = p9.ggplot() + p9.labs(
            title="Transformation CDF $F_R(r)$",
            subtitle="Dotted: Standard Gaussian CDF for reference",
            x="r",
            y="$F_R(r)$",
        )

        pdf_norm = tfd.Normal(loc=0.0, scale=1.0).cdf(grid_)
        p = p + p9.geom_line(p9.aes(grid_, pdf_norm), linetype="dotted")

        if ci_quantiles is not None:
            p = p + p9.geom_ribbon(
                p9.aes(
                    "r",
                    ymin=f"q_{str(ci_quantiles[0])}",
                    ymax=f"q_{str(ci_quantiles[1])}",
                ),
                fill="#56B4E9",
                alpha=0.5,
                data=cdf_summary,
            )

        if hdi_prob is not None:
            p = p + p9.geom_line(
                p9.aes("r", "hdi_low"),
                linetype="dotted",
                data=cdf_summary,
            )

            p = p + p9.geom_line(
                p9.aes("r", "hdi_high"),
                linetype="dotted",
                data=cdf_summary,
            )

        if jnp.asarray(self.trafo.value).ndim == 0:
            show_n_samples = 0

        if show_n_samples is not None and show_n_samples > 0:
            key = jax.random.key(seed) if isinstance(seed, int) else seed
            summary_samples_df = self.summarise_trafo_by_samples(
                key=key, grid=grid_, samples=samples, n=show_n_samples
            )

            p = p + p9.geom_line(
                p9.aes("r", "cdf", group="sample"),
                color="grey",
                data=summary_samples_df,
                alpha=0.3,
            )

        p = p + p9.geom_line(
            p9.aes("r", "mean"), data=cdf_summary, size=1.3, color="blue"
        )

        return p

    def plot(
        self,
        samples: dict[str, Array],
        grid: Array | None = None,
        ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
        hdi_prob: float | None = None,
        show_n_samples: int | None = 50,
        seed: int | KeyArray = 1,
        show: bool = True,
    ) -> tuple[p9.ggplot, p9.ggplot, p9.ggplot, p9.ggplot]:
        """Produce a set of diagnostic plots (qq, trafo, pdf, cdf).

        Parameters
        ----------
        samples
            Posterior samples dict used to build the distributions.
        grid, ci_quantiles, hdi_prob
            Plotting options forwarded to individual plot functions.
        show_n_samples, seed
            Controls subsampled trajectories overlay.
        show
            If True, call `.show()` on each plot before returning.

        Returns
        -------
        tuple
            Tuple of ggplot objects in order: (qq, trafo, pdf, cdf).
        """

        qq = self.plot_qq(samples)
        trafo = self.plot_trafo(
            samples,
            grid=grid,
            ci_quantiles=ci_quantiles,
            hdi_prob=hdi_prob,
            show_n_samples=show_n_samples,
            seed=seed,
        )
        pdf = self.plot_r_density(
            samples,
            grid=grid,
            ci_quantiles=ci_quantiles,
            hdi_prob=hdi_prob,
            show_n_samples=show_n_samples,
            seed=seed,
        )

        cdf = self.plot_r_cdf(
            samples,
            grid=grid,
            ci_quantiles=ci_quantiles,
            hdi_prob=hdi_prob,
            show_n_samples=show_n_samples,
            seed=seed,
        )

        if show:
            qq.show()
            trafo.show()
            pdf.show()
            cdf.show()
        return qq, trafo, pdf, cdf
