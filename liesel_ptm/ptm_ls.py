from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from enum import Enum, auto
from functools import partial
from itertools import chain

import dill
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.scipy.special import logsumexp
from liesel.goose.builder import EngineBuilder
from liesel.goose.kernel import Kernel
from liesel.goose.nuts import NUTSKernel

from .bsplines import ExtrapBSplineApprox
from .custom_types import Array, KeyArray
from .inverse_fn import approximate_inverse
from .liesel_internal import splines
from .nodes import (
    NormalCoef,
    Predictor,
    PSplineCoef,
    SymmetricallyBoundedScalar,
    TransformationDist,
    TransformedVar,
    VarHalfCauchy,
    array_to_dict,
    find_param,
    normalization_coef,
)
from .sampling import optimize_parameters, summarise_by_quantiles, summarise_by_samples

kn = splines.create_equidistant_knots

logger = logging.getLogger(__name__)


class ShapePrior(Enum):
    """
    Used to define the prior for the shape parameters of the transformation function.
    """

    RANDOM_WALK = auto()
    """First order random walk prior."""

    RIDGE = auto()
    """Ridge prior, which means an i.i.d. normal prior."""


def _scaled_normalization_and_deriv(
    unscaled_normalization_and_deriv: tuple[Array, Array], norm_mean, norm_sd
) -> tuple[Array, Array]:
    (
        unscaled_normalization,
        unscaled_normalization_deriv,
    ) = unscaled_normalization_and_deriv

    scaled_normalization = (unscaled_normalization - norm_mean) / norm_sd
    scaled_normalization_deriv = unscaled_normalization_deriv / norm_sd
    return scaled_normalization, scaled_normalization_deriv


def _centered_normalization_and_deriv(
    unscaled_normalization_and_deriv: tuple[Array, Array], norm_mean, norm_sd
) -> tuple[Array, Array]:
    (
        unscaled_normalization,
        unscaled_normalization_deriv,
    ) = unscaled_normalization_and_deriv

    scaled_normalization = unscaled_normalization - norm_mean
    scaled_normalization_deriv = unscaled_normalization_deriv
    return scaled_normalization, scaled_normalization_deriv


class Normalization:
    def __init__(
        self,
        knots: Array,
        standardized_response: Array,
        tau2: lsl.Var,
        transition_width: float = 0.3,
        scale_after_transformation: bool = True,
        scaling_factor: SymmetricallyBoundedScalar | None = None,
        shape_param_prior: ShapePrior = ShapePrior.RANDOM_WALK,
    ) -> None:
        self.dknots = jnp.diff(knots).mean()
        self.tau2 = tau2

        self.approx_bspline = ExtrapBSplineApprox(knots, order=3, eps=transition_width)
        self.basis_dot_and_deriv_fn = (
            self.approx_bspline.get_extrap_basis_dot_and_deriv_fn()
        )

        self.nshape = self.approx_bspline.basis.shape[-1] - 1

        if shape_param_prior == ShapePrior.RANDOM_WALK:
            self.shape_param = PSplineCoef(
                self.nshape, tau2=self.tau2, diff=1, name="normalization_shape"
            )
        elif shape_param_prior == ShapePrior.RIDGE:
            self.shape_param = NormalCoef(
                self.nshape, tau2=self.tau2, name="normalization_shape"
            )

        if scaling_factor is None:
            self.scaling_factor = lsl.Data(1.0, _name="normalization_scaling_factor")
            self.scaling_factor_supplied = False
        else:
            self.scaling_factor = scaling_factor
            self.scaling_factor_supplied = True
            if scale_after_transformation:
                scale_after_transformation = False
                logger.info(
                    "Scaling factor supplied - turning scaling after transformation"
                    " off."
                )

        self.coef = lsl.Calc(
            lambda shape, dknots, scaling_factor: scaling_factor
            * normalization_coef(shape, dknots),
            self.shape_param,
            self.dknots,
            self.scaling_factor,
            _name="normalization_coef",
        ).update()

        self.unscaled_normalization_and_deriv = lsl.Calc(
            self.basis_dot_and_deriv_fn,
            standardized_response,
            self.coef,
            _name="unscaled_normalization_and_deriv",
        ).update()

        self.norm_mean = lsl.Calc(
            lambda x: jnp.mean(x[0]),
            self.unscaled_normalization_and_deriv,
            _name="unscaled_normalization_mean",
        ).update()

        self.norm_sd = lsl.Calc(
            lambda x: jnp.std(x[0]),
            self.unscaled_normalization_and_deriv,
            _name="unscaled_normalization_sd",
        ).update()

        if scale_after_transformation:
            self.scaling_fn = _scaled_normalization_and_deriv
        else:
            self.scaling_fn = _centered_normalization_and_deriv

        self.scaled_normalization_and_deriv = lsl.Calc(
            self.scaling_fn,
            self.unscaled_normalization_and_deriv,
            self.norm_mean,
            self.norm_sd,
            _name="scaled_normalization_and_deriv",
        ).update()

        self.scaled_normalization = lsl.Calc(
            lambda x: x[0],
            self.scaled_normalization_and_deriv,
            _name="scaled_normalization",
        ).update()

        self.scaled_normalization_deriv = lsl.Calc(
            lambda x: x[1],
            self.scaled_normalization_and_deriv,
            _name="scaled_normalization_deriv",
        ).update()

        self._nuts_params = [self.shape_param.transformed_name]
        self._hyper_parameters = []
        self._parameters = [self.shape_param.transformed_name]

        tau2_param = find_param(self.tau2)
        if tau2_param is not None:
            self._hyper_parameters.append(tau2_param.name)
            self._nuts_params.append(tau2_param.name)

        if scaling_factor is not None:
            scaling_factor_param = find_param(self.scaling_factor)
            if scaling_factor_param is not None:
                self._parameters.append(scaling_factor_param.name)

        self.mean_and_sd_names = [self.norm_mean.name, self.norm_sd.name]

        self.mcmc_kernels: list[Kernel] = []
        self.mcmc_kernels.append(NUTSKernel(self.parameters))
        self.mcmc_kernels.append(NUTSKernel(self.hyper_parameters))

    @property
    def nuts_params(self) -> list[str]:
        warnings.warn(
            "nuts_param is deprecated. Use `parameters` and `hyper_parameters`"
            " instead.",
            FutureWarning,
        )
        return self._nuts_params

    @property
    def parameters(self) -> list[str]:
        return self._parameters

    @property
    def hyper_parameters(self) -> list[str]:
        return self._hyper_parameters

    def predict_normalization_and_deriv(
        self, samples: dict[str, Array], standardized_response: Array
    ) -> tuple[Array, Array]:
        shape_param = self.shape_param.shape_calc.predict(samples)
        coef = normalization_coef(shape_param, self.dknots)

        if self.scaling_factor_supplied:
            scaling_factor = jnp.expand_dims(self.scaling_factor.predict(samples), -1)
            coef *= scaling_factor

        norm_mean = jnp.expand_dims(samples[self.norm_mean.name], -1)
        norm_sd = jnp.expand_dims(samples[self.norm_sd.name], -1)

        unscaled_normalization_and_deriv = self.basis_dot_and_deriv_fn(
            standardized_response, coef
        )

        scaled_normalization_and_deriv = self.scaling_fn(
            unscaled_normalization_and_deriv, norm_mean, norm_sd
        )

        return scaled_normalization_and_deriv


class NormalizationFn:
    def __init__(
        self,
        knots: Array,
        order: int = 3,
        transition_width: float = 0.3,
    ):
        self.approx_bspline = ExtrapBSplineApprox(
            knots, order=order, eps=transition_width
        )
        self.basis_dot_and_deriv_fn = (
            self.approx_bspline.get_extrap_basis_dot_and_deriv_fn()
        )

    @partial(jnp.vectorize, excluded=[0], signature="(n),(p),(1),(1)->(n)")
    def __call__(
        self, y: Array, coef: Array, norm_mean: float, norm_sd: float
    ) -> Array:
        smooth, _ = self.basis_dot_and_deriv_fn(y, coef)
        return (smooth - norm_mean) / norm_sd

    @partial(jnp.vectorize, excluded=[0], signature="(n),(p),(1)->(n)")
    def deriv(self, y: Array, coef: Array, norm_sd: float) -> Array:
        _, deriv = self.basis_dot_and_deriv_fn(y, coef)
        return deriv / norm_sd

    def _find_y_lo(
        self,
        min_z: float | Array,
        coef: Array,
        norm_mean: float,
        norm_sd: float,
    ) -> Array:
        y_lo = self.approx_bspline.min_knot
        left_shift = jnp.array(0.1)
        min_zgrid = jnp.min(
            self(jnp.atleast_1d(y_lo - left_shift), coef, norm_mean, norm_sd)
        )

        def _cond_fun_l(val):
            _, min_zgrid = val
            return jnp.squeeze(min_zgrid >= min_z)

        def _body_fun_l(val):
            left_shift, min_zgrid = val
            left_shift = left_shift + 0.5
            min_zgrid = jnp.min(
                self(jnp.atleast_1d(y_lo - left_shift), coef, norm_mean, norm_sd)
            )
            return left_shift, min_zgrid

        # while _cond_fun_l((left_shift, min_zgrid)):
        #     left_shift, min_zgrid = _body_fun_l((left_shift, min_zgrid))

        left_shift, _ = jax.lax.while_loop(
            _cond_fun_l, _body_fun_l, (left_shift, min_zgrid)
        )
        return y_lo - left_shift

    def _find_y_hi(
        self,
        max_z: float | Array,
        coef: Array,
        norm_mean: float,
        norm_sd: float,
    ) -> Array:
        y_hi = self.approx_bspline.max_knot
        right_shift = jnp.array(0.1)
        max_zgrid = jnp.max(
            self(jnp.atleast_1d(y_hi + right_shift), coef, norm_mean, norm_sd)
        )

        def _cond_fun_r(val):
            _, max_zgrid = val
            return jnp.squeeze(max_zgrid <= max_z)

        def _body_fun_r(val):
            right_shift, max_zgrid = val
            right_shift = right_shift + 0.5
            max_zgrid = jnp.max(
                self(jnp.atleast_1d(y_hi + right_shift), coef, norm_mean, norm_sd)
            )
            return right_shift, max_zgrid

        # while _cond_fun_r((right_shift, max_zgrid)):
        #     right_shift, max_zgrid = _body_fun_r((right_shift, max_zgrid))
        right_shift, _ = jax.lax.while_loop(
            _cond_fun_r, _body_fun_r, (right_shift, max_zgrid)
        )
        return y_hi + right_shift

    @partial(jnp.vectorize, excluded=[0, 1, 2, 6], signature="(p),(1),(1)->(n),(n)")
    def _find_grid(
        self,
        y_lo: float | Array,
        y_hi: float | Array,
        coef: Array,
        norm_mean: float,
        norm_sd: float,
        ngrid: int = 200,
    ) -> tuple[Array, Array]:
        """
        Finds a grid of y values such that h(y) covers the range of z values.
        """

        ygrid = jnp.linspace(y_lo, y_hi, ngrid)
        zgrid = self(ygrid, coef, norm_mean, norm_sd)

        return ygrid, zgrid

    def inverse(
        self, z: Array, coef: Array, norm_mean: Array, norm_sd: Array, ngrid: int = 200
    ) -> Array:
        """
        Params
        ------
        ngrid
            The number of grid points to use for the approximation in each sub-interval
            of length 1. The total number of grid points will be
            ``ngrid * (int(max_z - min_z) + 1)``.
        """
        min_z, max_z = jnp.min(z), jnp.max(z)
        y_lo = self._find_y_lo(min_z, coef, norm_mean, norm_sd)
        y_hi = self._find_y_hi(max_z, coef, norm_mean, norm_sd)
        ngrid = int((jnp.array(y_hi - y_lo, int) + 1) * ngrid)
        ygrid, zgrid = self._find_grid(y_lo, y_hi, coef, norm_mean, norm_sd, ngrid)
        y_approx = approximate_inverse(ygrid, zgrid, z)
        return y_approx

    def inverse_newton(
        self,
        z: Array,
        coef: Array,
        norm_mean: Array,
        norm_sd: Array,
        tol=1e-6,
        max_iter=100,
    ) -> Array:
        @jax.jit
        @partial(jnp.vectorize, signature="(n),(n),(p),(1),(1)->(n)")
        def inverse_fn(z, initial_guess, coef, norm_mean, norm_sd):
            def newton_step(y, z):
                h = self(y, coef, norm_mean, norm_sd)
                h_deriv = self.deriv(y, coef, norm_sd)
                return y - (h - z) / h_deriv

            def cond_fn(state):
                y, iter_count, z = state
                h = self(y, coef, norm_mean, norm_sd)
                return jnp.logical_and(
                    jnp.any(jnp.abs(h - z) >= tol), iter_count < max_iter
                )

            def body_fn(state):
                y, iter_count, z = state
                y_new = newton_step(y, z)
                return y_new, iter_count + 1, z

            # Initial state: (current x, new x, iteration count)
            state = (initial_guess, 0, z)
            y_new, i, _ = jax.lax.while_loop(cond_fn, body_fn, state)

            return y_new

        initial_guess = self.inverse(z, coef, norm_mean, norm_sd, 10)

        return inverse_fn(z, initial_guess, coef, norm_mean, norm_sd)


def _extract_position(
    position_keys: list[str], graph: lsl.Model, state: gs.ModelState
) -> dict[str, Array]:
    position = {}

    for key in position_keys:
        try:
            name = graph.nodes[key].name
            position[name] = state[name].value
        except KeyError:
            name = graph.vars[key].value_node.name
        position[name] = state[name].value

    return position


def _update_state(position: dict[str, Array], graph: lsl.Model) -> lsl.Model:
    graph.auto_update = False

    for node in graph.nodes.values():
        node._outdated = False

    for key, value in position.items():
        try:
            graph.nodes[key].value = value  # type: ignore  # data node
        except KeyError:
            graph.vars[key].value = value

    graph.update()
    graph.auto_update = True
    return graph


class LocationIntercept(lsl.Calc):
    def __init__(self, y, loc_model, scale_model, _name: str = "") -> None:
        super().__init__(self.compute_intercept, y, loc_model, scale_model, _name=_name)

    @staticmethod
    def compute_intercept(y, loc_model, scale_model):
        inv_scale_model_mean = 1 / jnp.mean(1 / scale_model)
        residual_model_mean = jnp.mean((y - loc_model) / scale_model)
        return inv_scale_model_mean * residual_model_mean


class ScaleInterceptExp(lsl.Calc):
    def __init__(
        self, y, loc_intercept, loc_model, scale_model, _name: str = ""
    ) -> None:
        super().__init__(
            self.compute_intercept,
            y,
            loc_intercept,
            loc_model,
            scale_model,
            _name=_name,
        )

    @staticmethod
    def compute_intercept(y, loc_intercept, loc_model, scale_model):
        return jnp.std((y - loc_intercept - loc_model) / scale_model)


class LocScaleResiduals(lsl.Calc):
    def __init__(self, y, loc, scale, _name: str = "") -> None:
        super().__init__(self.compute_residuals, y, loc, scale, _name=_name)

    @staticmethod
    def compute_residuals(y, loc, scale):
        return (y - loc) / scale


def _sum(*args):
    return sum(args)


def _product(a, b):
    return a * b


def default_scaling_factor() -> TransformedVar:
    prior = lsl.Dist(tfd.TruncatedNormal, loc=1.0, scale=0.1, low=0.01, high=jnp.inf)
    scaling_factor = TransformedVar(
        value=1.0,
        prior=prior,
        name="scaling_factor",
    )
    return scaling_factor


class PTMLocScale:
    """
    Penalized transformation model for location and scale.

    Parameters
    ----------
    knots
        The knots used to the define the transformation function.
    y
        The observed response variable.
    normalization_tau2
        The variance parameter :math:`\\tau^2_\\delta` for the prior of the \
        transformation function's shape parameters.
    transition_width
        The width of the transition segment of the transformation function. A number \
        larger than 0, indicating the length in terms of a multiple of the range of \
        interior knots. Called :math:`\\lambda`.
    scale_after_transformation
        Whether to scale the response variable after the transformation. It is not \
        avisable to switch this to ``True``.
    scaling_factor
        The scaling factor :math:`\\omega` for the transformation function. \
        If ``None``, the scaling factor will be fixed to one.
    shape_param_prior
        The prior distribution for the shape parameter.
    """

    def __init__(
        self,
        knots: Array,
        y: Array,
        normalization_tau2: lsl.Var,
        transition_width: float = 0.3,
        scale_after_transformation: bool = False,
        scaling_factor: lsl.Var | None = None,
        shape_param_prior: ShapePrior = ShapePrior.RANDOM_WALK,
    ) -> None:
        self.knots = knots
        """The knots of the transformation function's spline segment."""

        self.transition_width = transition_width
        """The transition width."""

        self.loc_model: Predictor = Predictor("loc_model").update()
        """Predictor for the location model part. Does not include an intercept."""

        self.log_scale_model: Predictor = Predictor("log_scale_model").update()
        """Predictor for the log scale model part. Does not include an intercept."""

        self.scale_model = lsl.Calc(jnp.exp, self.log_scale_model).update()
        """The exponential of :attr:`.log_scale_model`."""

        self.normalization_tau2 = normalization_tau2
        """The hyperparameter :math:`\\tau^2_\\delta`."""

        self.scale_after_transformation = scale_after_transformation
        """Whether to scale the response variable after the transformation."""

        self.scaling_factor = scaling_factor
        """The scaling factor :math:`\\omega` for the transformation function."""

        self.shape_param_prior = shape_param_prior
        """The prior distribution for the shape parameter."""

        y_data = lsl.Data(y, _name="y_data")
        y_data.observed = True

        self.loc_intercept = lsl.Var(
            LocationIntercept(y_data, self.loc_model, self.scale_model).update(),
            name="loc_intercept",
        )
        """Intercept :math:`\\beta_0` of the location model part."""

        self.scale_intercept_exp = lsl.Var(
            ScaleInterceptExp(
                y_data, self.loc_intercept, self.loc_model, self.scale_model
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

        self.residuals = lsl.Var(
            LocScaleResiduals(y_data, self.loc, self.scale).update(), name="residuals"
        )
        """The residuals."""

        self.normalization: Normalization = Normalization(
            self.knots,
            self.residuals,
            tau2=self.normalization_tau2,
            transition_width=transition_width,
            scale_after_transformation=scale_after_transformation,
            scaling_factor=scaling_factor,
            shape_param_prior=shape_param_prior,
        )
        """Organizes the transformation model part :math:`h(\\varepsilon`)."""

        self.z_deriv = lsl.Calc(
            lambda norm_deriv, scale: norm_deriv / scale,
            self.normalization.scaled_normalization_deriv,
            self.scale,
            _name="z_deriv",
        ).update()
        """Derivative of :math:`h(\\varepsilon)` with respect to the response."""

        self.z = self.normalization.scaled_normalization
        """Variable for the transformation model part :math:`h(\\varepsilon`)."""

        self.refdist = tfd.Normal(loc=0.0, scale=1.0)
        """
        The reference distribution, currently fixed to the standard normal distribution.
        """

        response_dist = TransformationDist(self.z, self.z_deriv, refdist=self.refdist)
        self.response = lsl.obs(y, response_dist, name="response").update()
        """Response variable."""

        self.intercept_names = [self.loc_intercept.name, self.scale_intercept_exp.name]
        """Names of the intercept nodes."""

    @classmethod
    def from_nparam(
        cls,
        y: Array,
        nparam: int,
        knots_lo: float = -3.0,
        knots_hi: float = 3.0,
        transition_width: float = 0.3,
        normalization_tau2: lsl.Var | None = None,
        scale_after_transformation: bool = False,
        scaling_factor: lsl.Var | None = None,
        shape_param_prior: ShapePrior = ShapePrior.RANDOM_WALK,
    ) -> PTMLocScale:
        """
        Initializes a penalized transformation model automatically from the desired
        number of shape parameters.

        Parameters
        ----------
        y
            The observed response variable.
        nparam
            The number of parameters in the spline segment of the transformation \
            function. Called :math:`J` in the paper.
        knots_lo, knots_hi
            Lower and upper boundary knot, used to define an equidistant knot grid
            for the transformation function.
        transition_width
            The width of the transition segment of the transformation function. A number \
            larger than 0, indicating the length in terms of a multiple of the range of \
            interior knots. Called :math:`\\lambda`.
        scale_after_transformation
            Whether to scale the transformed residuals.
        scaling_factor
            The scaling factor :math:`\\omega` for the transformation function. \
            If ``None``, the scaling factor will be fixed to one.
        shape_param_prior
            The prior distribution for the shape parameter.
        """
        knots = kn(np.array([knots_lo, knots_hi]), order=3, n_params=nparam)
        if normalization_tau2 is None:
            normalization_tau2 = VarHalfCauchy(
                10.0, scale=25.0, name="normalization_tau2"
            )

        model = cls(
            knots,
            y,
            normalization_tau2,
            transition_width=transition_width,
            scale_after_transformation=scale_after_transformation,
            scaling_factor=scaling_factor,
            shape_param_prior=shape_param_prior,
        )
        return model

    def _pkl_knots(self, path: str):
        """For testing."""
        with open(path, "wb") as f:
            dill.dump(self.knots, f)

    @classmethod
    def _from_knots(
        cls, knots_path: str, y: Array, transition_width: float = 0.3
    ) -> PTMLocScale:
        """For testing."""
        with open(knots_path, "rb") as f:
            knots = dill.load(f)

        normalization_tau2 = VarHalfCauchy(10.0, scale=25.0, name="normalization_tau2")

        return cls(knots, y, normalization_tau2, transition_width=transition_width)

    def new_from_graph(
        self, graph: lsl.Model, knot_prob_levels: tuple[float, float] = (0.01, 0.99)
    ) -> PTMLocScale:
        """
        Given a Liesel model graph, this creates a new :class:`.PTMLocScale` with
        updated knots.

        Parameters
        ----------
        graph
            The liesel model graph to base the new model on.
        knot_prob_levels
            At which probability levels of the observed residuals in ``graph`` should
            the new minimum and maximum knots be placed.

        Returns
        -------
        A new model instance with updated knots. All prior and hyperprior specifications
        for the transformation and the location and scale terms are copied from
        the original model.
        """
        residuals = graph.state[f"{self.residuals.name}_value"].value

        min_z = np.quantile(residuals, knot_prob_levels[0])
        max_z = np.quantile(residuals, knot_prob_levels[1])

        nparam = self.normalization.nshape + 1

        model = PTMLocScale.from_nparam(
            self.response.value,
            nparam,
            knots_lo=min_z,
            knots_hi=max_z,
            transition_width=self.transition_width,
            scale_after_transformation=self.scale_after_transformation,
            scaling_factor=self.scaling_factor,
            shape_param_prior=self.shape_param_prior,
            normalization_tau2=self.normalization_tau2,
        )

        for term in self.loc_model.terms.values():
            model.loc_model += term

        for term in self.log_scale_model.terms.values():
            model.log_scale_model += term

        graph.pop_nodes_and_vars()

        return model

    def optimize_knots(
        self,
        optimize_params: list[list[str]] | None = None,
        maxiter: int = 10_000,
        knot_prob_levels: tuple[float, float] = (0.0, 1.0),
        **optim_args,
    ) -> tuple[PTMLocScale, dict[str, Array]]:
        """
        Finds good knots for the transformation function.

        This function runs stochastic gradient descent on the location and scale
        model parts while fixing the transformation function to a straight line.
        This yields a preliminary set of residuals, which are then used to define
        the new knots for the transformation function's spline segment.

        Parameterss
        -----------
        *optimize_params
            The parameters to optimize. If not specified, uses the parameters of the\
            location and scale models, but not their hyperparameters. \
            Do not include the\
            normalization parameters here. The values supplied here should be lists of\
            strings. Each string should be the name of a parameter in the model. Each\
            list is a group of parameters that should be optimized together while\
            holding the remaining parameters of the model fixed.
        maxiter
            Additional argument to pass to the optimizer :func:`.optmize_parameters`.
        knot_prob_levels
            At which probability levels of the observed residuals in ``graph`` should
            the new minimum and maximum knots be placed.
        **optim_args
            Additional arguments to pass to the optimizer :func:`.optmize_parameters`.

        Returns
        -------
        A tuple. The first element is a new instance of :class:`.PTMLocScale` with
        improved knots. The second element is a position dictionary, holding the
        final parameter values from the initial optimization for possible further
        use.
        """
        graph = lsl.GraphBuilder().add(self.response).build_model()

        if optimize_params is None:
            params = self.loc_model.parameters + self.log_scale_model.parameters
            optimize_params = [params]

        position: dict[str, Array] = {}

        for params in optimize_params:
            logger.info(f"Optimizing {len(params)} parameters.")

            results = optimize_parameters(graph, params, maxiter=maxiter, **optim_args)
            graph.state = results.model_state
            logger.info(f"Finished optimization after {results.iteration} iterations.")
            position |= _extract_position(params, graph, results.model_state)

        new_model = self.new_from_graph(graph, knot_prob_levels=knot_prob_levels)

        return new_model, position

    def build_graph(
        self,
        optimize_params: list[list[str]] | None = None,
        position: dict[str, Array] | None = None,
        optimize_start_values: bool = True,
        optimize_normalization: bool = True,
        maxiter: int = 10_000,
        **optim_args,
    ) -> lsl.Model:
        """
        Builds the model graph.

        Parameters
        ----------
        *optimize_params
            The parameters to optimize. If not specified, all parameters are optimized.
            The values supplied here should be lists of
            strings. Each string should be the name of a parameter in the model. Each
            list is a group of parameters that should be optimized together while
            holding the remaining parameters of the model fixed.
        position
            Dictionary of model parameter names and values to use as starting values.\
            Values of parameters included in the position will not be included\
            in pre-optimization.
        optimize_start_values
            Whether to optimize the start values of the model using stochastic gradient\
            descent.
        optimize_normalization
            Whether to include the shape parameters in the optimizaiton of start values.
        maxiter, **optim_args
            Additional arguments to pass to the optimizer :func:`.optmize_parameter`.
        """
        position = position if position is not None else {}
        gb = lsl.GraphBuilder().add(self.response)
        graph = gb.build_model()
        graph = _update_state(position, graph)

        if not optimize_start_values:
            return graph

        if optimize_params is None:
            params = []  # type: ignore
            params = self.loc_model.parameters + self.log_scale_model.parameters

            if optimize_normalization:
                params += self.normalization.parameters  # type: ignore

            optimize_params = [params]

        i = 1
        for param in optimize_params:
            logger.info(f"Optimizing parameter block #{i} (len={len(param)}).")
            results = optimize_parameters(graph, param, maxiter=maxiter, **optim_args)
            graph.state = results.model_state
            logger.info(f"Finished optimization after {results.iteration} iterations.")
            i += 1

        return graph

    def setup_engine_builder(
        self,
        eb: EngineBuilder,
        graph: lsl.Model | None = None,
        sample_normalization: bool = True,
    ) -> EngineBuilder:
        """
        Prepares a ``liesel.goose.EngineBuilder`` for MCMC sampling.

        This method sets the model, the initial values, and defines the MCMC kernels
        for the model parameters. You still need to specify the warmup and posterior
        durations.
        """
        graph = graph if graph is not None else self.build_graph()

        eb.set_model(gs.LieselInterface(graph))
        eb.set_initial_values(graph.state)

        loc_terms = self.loc_model.terms.values()
        scale_terms = self.log_scale_model.terms.values()

        for term in chain(loc_terms, scale_terms):
            for kernel in term.mcmc_kernels:
                eb.add_kernel(kernel)

        if sample_normalization:
            for kernel in self.normalization.mcmc_kernels:
                eb.add_kernel(kernel)

        eb.positions_included = self.intercept_names
        eb.positions_included += self.normalization.mean_and_sd_names

        return eb

    def waic(self, samples: dict[str, Array]) -> pd.DataFrame:
        """
        Assembles a ``pd.DataFrame`` that gives information on the widely applicable
        information criterion for this model.

        The column ``n_warning`` indicates the number of samples for which the variance
        of the log predictive density is greater than 0.4.
        """
        pred = PTMLocScalePredictions(samples, self)
        log_prob = pred.predict_log_prob()
        return waic(log_prob)

    def all_sampled_parameter_names(self) -> list[str]:
        params = (
            self.normalization.parameters
            + self.normalization.hyper_parameters
            + self.loc_model.parameters
            + self.loc_model.hyper_parameters
            + self.log_scale_model.parameters
            + self.log_scale_model.hyper_parameters
        )

        return params

    def all_parameter_names(self) -> list[str]:
        params = (
            self.normalization.parameters
            + self.normalization.hyper_parameters
            + self.intercept_names
            + self.normalization.mean_and_sd_names
            + self.loc_model.parameters
            + self.loc_model.hyper_parameters
            + self.log_scale_model.parameters
            + self.log_scale_model.hyper_parameters
        )

        return params


def state_to_samples(
    position_keys: Sequence[str],
    graph: lsl.Model,
    expand_axes: Sequence[int] | None = (0, 1),
) -> dict[str, Array]:
    """
    Given a list of position keys (variable and node anmes) and a model graph
    represented by a :class:`liesel.model.Model`, this function
    constructs a samples dictionary.
    """
    position = gs.LieselInterface(graph).extract_position(position_keys, graph.state)

    def axes(x):
        return np.expand_dims(x, expand_axes) if expand_axes is not None else x

    samples = {name: axes(value) for name, value in position.items()}

    return samples


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


class PTMLocScalePredictions:
    """
    Posterior predictions for a penalized transformation model.

    Parameters
    ----------
    samples
        Dictionary of posterior samples.
    model
        The model object.
    y
        Response observations to use in predictions. If ``None``(default), the observed
        values are extracted from the model object.
    **kwargs
        Values to use for all terms in the location and scale model parts. Must have
        appropriate shapes. For terms that are not explicitly specified, the observed
        values are extracted from the model object.
    """

    def __init__(
        self,
        samples: dict[str, Array],
        model: PTMLocScale,
        y: Array | None = None,
        **kwargs,
    ) -> None:
        self.samples = samples
        """The posterior samples."""

        self.model = model
        """The model object."""

        self.kwargs = self._set_kwargs(**kwargs)
        self.y = y if y is not None else model.response.value

        self.loc_intercept = samples[model.loc_intercept.name]
        self.scale_intercept_exp = samples[model.scale_intercept_exp.name]

        self.norm_mean = samples[model.normalization.norm_mean.name]
        self.norm_std = samples[model.normalization.norm_sd.name]

        self._loc_model: None | Array = None
        self._scale_model: None | Array = None

        self._residuals: None | Array = None
        self._normalization_and_deriv: None | tuple[Array, Array] = None

        self._z_deriv: None | Array = None

        self._log_prob: None | Array = None
        self._cdf: None | Array = None
        self._pdf: None | Array = None

    def _set_kwargs(self, **kwargs):
        for key in self.model.loc_model.terms.keys():
            if key not in kwargs:
                kwargs[key] = None

        for key in self.model.log_scale_model.terms.keys():
            if key not in kwargs:
                kwargs[key] = None

        for key, value in kwargs.items():
            loc_key = key in self.model.loc_model.terms
            scale_key = key in self.model.log_scale_model.terms

            if not loc_key and not scale_key:
                raise ValueError(f"Key {key} not found in loc or scale model.")

            if value is not None:
                kwargs[key] = jnp.atleast_1d(value)

            if value is None:
                loc_term = self.model.loc_model.terms.get(key, None)
                scale_term = self.model.log_scale_model.terms.get(key, None)
                term = loc_term if loc_term is not None else scale_term
                kwargs[key] = term.observed_value

        return kwargs

    def predict_loc(self) -> Array:
        """Predicts the location."""
        if self._loc_model is None:
            self._loc_model = self.model.loc_model.predict(self.samples, **self.kwargs)

        return self._loc_model + jnp.expand_dims(self.loc_intercept, -1)

    def predict_scale(self) -> Array:
        """Predicts the location."""
        if self._scale_model is None:
            log_scale = self.model.log_scale_model.predict(self.samples, **self.kwargs)
            self._scale_model = np.exp(log_scale)

        return self._scale_model * jnp.expand_dims(self.scale_intercept_exp, -1)

    def predict_residuals(self) -> Array:
        """Predicts the residuals."""
        if self._residuals is None:
            loc = self.predict_loc()
            scale = self.predict_scale()
            self._residuals = (self.y - loc) / scale

        return self._residuals

    def predict_normalization_and_deriv(self) -> tuple[Array, Array]:
        """
        Predicts the transformed residuals and the derivative of the normalization.

        This derivative does *not* account for the scale model part.
        """
        if self._normalization_and_deriv is None:
            self._normalization_and_deriv = (
                self.model.normalization.predict_normalization_and_deriv(
                    self.samples, self.predict_residuals()
                )
            )
        return self._normalization_and_deriv

    def predict_normalization_deriv(self) -> Array:
        """
        Predicts the derivative of the normalization.

        This derivative does *not* account for the scale model part.
        """
        return self.predict_normalization_and_deriv()[1]

    def predict_z(self) -> Array:
        """
        Predicts the transformed residuals.

        Alias for :meth:`.predict_transformation`.
        """
        return self.predict_normalization_and_deriv()[0]

    def predict_z_deriv(self) -> Array:
        """
        Predicts the derivative of the transformation.

        This derivative *does* account for the scale model part.
        Alias for :meth:`.predict_transformation_deriv`.
        """
        if self._z_deriv is None:
            self._z_deriv = self.predict_normalization_deriv() / self.predict_scale()

        if np.any(self._z_deriv <= 0):
            smallest_gradient = np.min(self._z_deriv)
            n_seq_zero = np.sum(self._z_deriv <= 0)
            logger.warning(
                f"{n_seq_zero} gradient values <= 0 encountered. Smallest gradient:"
                f" {smallest_gradient}. This may be just a numerical artifact due to"
                " floating point imprecision, but it should be checked."
            )

        return self._z_deriv

    def predict_transformation(self) -> Array:
        """Predicts the transformed residuals."""
        return self.predict_z()

    def predict_transformation_deriv(self) -> Array:
        """
        Predicts the derivative of the transformation.

        This derivative *does* account for the scale model part.
        """
        return self.predict_z_deriv()

    def predict_log_prob(self) -> Array:
        """Predicts the response's conditional log probability."""
        if self._log_prob is not None:
            return self._log_prob
        z = self.predict_z()
        z_deriv = self.predict_z_deriv()
        z_deriv = jnp.maximum(z_deriv, 1e-30)
        base_log_prob = self.model.refdist.log_prob(z)
        log_prob_adjustment = jnp.log(z_deriv)
        self._log_prob = jnp.add(base_log_prob, log_prob_adjustment)
        return self._log_prob

    def predict_cdf(self) -> Array:
        """Predicts the response's conditional cumulative distribution function."""
        if self._cdf is not None:
            return self._cdf
        z = self.predict_z()
        self._cdf = self.model.refdist.cdf(z)
        return self._cdf

    def predict_pdf(self) -> Array:
        """Predicts the response's conditional probability density."""
        if self._pdf is not None:
            return self._pdf
        self._pdf = jnp.exp(self.predict_log_prob())
        return self._pdf

    def predict_normalization_inverse(self, z: Array) -> Array:
        """
        Predicts the inverse of the normalization function.

        This does *not* account for the location and scale model parts. It basically
        takes a :math:`z` value, following the reference distribution, and turns it
        into a residual :math:`\\varepsilon`.
        """
        samples = self.samples

        shape_param = self.model.normalization.shape_param.shape_calc.predict(samples)
        coef = normalization_coef(shape_param, jnp.diff(self.model.knots).mean())

        norm_mean = jnp.expand_dims(
            samples[self.model.normalization.norm_mean.name], -1
        )
        if self.model.scaling_factor is not None:
            scaling_factor = jnp.expand_dims(
                self.model.scaling_factor.predict(self.samples), -1
            )
            coef *= scaling_factor
            norm_sd = jnp.ones_like(norm_mean)
        elif self.model.scale_after_transformation:
            norm_sd = jnp.expand_dims(
                samples[self.model.normalization.norm_sd.name], -1
            )
        else:
            norm_sd = jnp.ones_like(norm_mean)

        normalization = NormalizationFn(self.model.knots, order=3)
        zt = normalization.inverse(z, coef, norm_mean, norm_sd)
        return zt

    def predict_transformation_inverse(self, z: Array) -> Array:
        """
        Predicts the inverse of the normalization function.

        This does *not* account for the location and scale model parts. It basically
        takes a :math:`z` value, following the reference distribution, and turns it
        into a response observation :math:`y`.
        """
        zt = self.predict_normalization_inverse(z)
        zt = jnp.expand_dims(zt, 2)

        loc = jnp.expand_dims(self.predict_loc(), -1)
        scale = jnp.expand_dims(self.predict_scale(), -1)

        yhat = scale * zt + loc

        return yhat

    def predict_quantile(self, p: Array) -> Array:
        """
        Predicts the conditional response quantiles at probability level ``p``.

        Currently, samples must be of shape (chains, iterations, param_dimension).
        The last dimension can be omitted for scalar parameters. Basically, this
        function assumes that you enter the samples just like you got them from
        goose.

        The returned value will have the following shape::

            [chain, iteration, covariates, len(p)]

        """
        z = jnp.atleast_1d(self.model.refdist.quantile(p))
        return self.predict_transformation_inverse(z)

    def predict_resid_quantile(self, p: Array) -> Array:
        """
        Predicts the residual quantiles at probability level ``p``.
        """
        z = jnp.atleast_1d(self.model.refdist.quantile(p))
        return self.predict_normalization_inverse(z)

    def sample(self, key: KeyArray, shape: Sequence[int]) -> Array:
        """
        Generates random samples from the posterior predictive distribution.

        The returned array will have the following shape::

            [chain, iteration, covariates, shape]
        """
        samples_shape = self.predict_residuals().shape
        q = jax.random.uniform(key, shape=samples_shape[:-1] + shape)
        return self.predict_quantile(q)

    def summarise_density_by_quantiles(
        self, axis: Sequence[int] | int = (0, 1), lo: float = 0.1, hi: float = 0.9
    ) -> pd.DataFrame:
        """
        Returns a summary dataframe to quickly assess the response's conditional
        distribution.
        """
        z = self.predict_transformation()
        cdf = self.predict_cdf()
        pdf = self.predict_pdf()

        z_df = summarise_by_quantiles(z, "z", axis=axis, lo=lo, hi=hi)
        pdf_df = summarise_by_quantiles(pdf, "pdf", axis=axis, lo=lo, hi=hi)
        cdf_df = summarise_by_quantiles(cdf, "cdf", axis=axis, lo=lo, hi=hi)

        df = pd.concat([z_df, pdf_df, cdf_df], axis=1)

        df["y"] = self.y

        for xname, xval in self.kwargs.items():
            x_dict = array_to_dict(xval, names_prefix=xname)
            for key, value in x_dict.items():
                df[key] = np.asarray(np.squeeze(value))

        return df

    def predict_residual_cdf(
        self, residuals: Array | None, axis: Sequence[int] | int = (0, 1)
    ) -> Array:
        """Evaluates the residual's posterior cumulative distribution function."""
        if residuals is None:
            residuals = self.predict_residuals().mean(axis=axis, keepdims=True)
        z, _ = self.model.normalization.predict_normalization_and_deriv(
            self.samples, residuals
        )
        return self.model.refdist.cdf(z)

    def predict_residual_log_prob(
        self, residuals: Array | None, axis: Sequence[int] | int = (0, 1)
    ) -> Array:
        """Evaluates the residual's posterior log probability function."""
        if residuals is None:
            residuals = self.predict_residuals().mean(axis=axis, keepdims=True)
        z, z_deriv = self.model.normalization.predict_normalization_and_deriv(
            self.samples, residuals
        )
        z_deriv = jnp.maximum(z_deriv, 1e-30)
        base_log_prob = self.model.refdist.log_prob(z)
        log_prob_adjustment = jnp.log(z_deriv)
        log_prob = jnp.add(base_log_prob, log_prob_adjustment)
        return log_prob

    def predict_residual_pdf(
        self, residuals: Array | None, axis: Sequence[int] | int = (0, 1)
    ) -> Array:
        """Evaluates the residual's posterior density function."""
        return np.exp(self.predict_residual_log_prob(residuals, axis))

    def summarise_transformation_by_quantiles(
        self,
        residuals: Array | None,
        axis: Sequence[int] | int = (0, 1),
        lo: float = 0.1,
        hi: float = 0.9,
    ) -> pd.DataFrame:
        """
        Returns a summary dataframe to quickly assess the transformation function.
        """
        if residuals is None:
            residuals = self.predict_residuals().mean(axis=axis, keepdims=True)

        z, _ = self.model.normalization.predict_normalization_and_deriv(
            self.samples, residuals
        )

        cdf = self.predict_residual_cdf(residuals, axis)
        pdf = self.predict_residual_pdf(residuals, axis)

        z_df = summarise_by_quantiles(z, "z", axis=axis, lo=lo, hi=hi)
        pdf_df = summarise_by_quantiles(pdf, "pdf", axis=axis, lo=lo, hi=hi)
        cdf_df = summarise_by_quantiles(cdf, "cdf", axis=axis, lo=lo, hi=hi)

        df = pd.concat([z_df, pdf_df, cdf_df], axis=1)

        df["residual"] = np.squeeze(residuals)

        return df

    def summarise_density_by_samples(
        self, key: KeyArray | int, n: int = 100
    ) -> pd.DataFrame:
        """
        Returns a summary dataframe to quickly assess the response's conditional
        distribution based on a random sample from the posterior.
        """
        key = jax.random.PRNGKey(key) if isinstance(key, int) else key

        # draw n random samples from an array of shape [chain, iteration, nobs]
        z = self.predict_transformation()
        cdf = self.predict_cdf()
        pdf = self.predict_pdf()

        z_df = summarise_by_samples(key, z, "z", n=n)
        cdf_df = summarise_by_samples(key, cdf, "cdf", n=n)
        pdf_df = summarise_by_samples(key, pdf, "pdf", n=n)

        df = pd.concat([z_df.z, cdf_df.cdf, pdf_df.pdf], axis=1)
        df["index"] = z_df.index
        df["obs"] = z_df.obs
        df["chain"] = z_df.chain
        df["sample"] = z_df["sample"]

        df["y"] = np.tile(jnp.squeeze(self.y), n)

        for xname, xval in self.kwargs.items():
            xval = jnp.squeeze(xval)
            if jnp.atleast_1d(xval).shape[-1] == 1:
                df[xname] = np.asarray(xval)
            elif len(xval.shape) > 1:
                for k in range(xval.shape[-1]):
                    df[f"{xname}{k}"] = np.asarray(np.tile(xval[:, k], n))
            else:
                df[xname] = np.asarray(np.tile(xval, n))

        return df
