from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf
from tensorflow_probability.substrates.jax.internal import (
    parameter_properties as tfp_parameter_properties,
)
from tensorflow_probability.substrates.jax.internal import slicing as tfp_slicing

from .bijector import (
    SplineBijector,
    _gauss_legendre_nodes_and_weights,
    _moment_diagnostic,
    _validate_gauss_legendre_order,
)
from .bijector import (
    integrate_piecewise_gauss_legendre as integrate_piecewise_gauss_legendre,
)
from .bspline import OnionKnots, OnionSpline, PTMSpline

KeyArray = Any
Array = Any


@jax.jit
def _composite_quantile(distribution: tfd.Distribution, value: Array) -> Array:
    # TFP's iterative quantiles otherwise compile fresh scans on every call.
    return distribution.quantile(value)


def onion_dist(
    a: float = -4.0,
    b: float = 4.0,
    nparam: int = 20,
    *,
    knots: Array | OnionKnots | None = None,
    order: int = 3,
    loc_scale: bool = True,
    **kwargs,
) -> Callable[..., TransformationDist]:
    """
    Return a reusable constructor for an OnionSpline transformation distribution.

    The OnionSpline is instantiated once in this factory and then reused by each
    distribution created by the returned constructor.
    """
    if knots is None:
        knots = OnionKnots(a, b, nparam=nparam, order=order).knots
    elif isinstance(knots, OnionKnots):
        knots = knots.knots

    bspline = OnionSpline(knots)
    dist_class = LocScaleTransformationDist if loc_scale else TransformationDist

    return partial(dist_class, bspline=bspline, **kwargs)


def _is_array_parameter(value: Any) -> bool:
    """Whether a parameter is an array or float, which JAX may trace."""
    return isinstance(value, jax.Array | np.ndarray | float | np.floating)


def _flatten_transformation_dist(
    dist: TransformationDist,
) -> tuple[tuple[Any, ...], tuple[tuple[str, ...], tuple[tuple[str, Any], ...]]]:
    parameters = dist.parameters
    keys = dist._array_parameter_keys()
    static = tuple((key, value) for key, value in parameters.items() if key not in keys)
    return tuple(parameters[key] for key in keys), (keys, static)


def _unflatten_transformation_dist(
    cls: type[TransformationDist],
    aux: tuple[tuple[str, ...], tuple[tuple[str, Any], ...]],
    children: Sequence[Any],
) -> TransformationDist:
    keys, static = aux
    return cls(**dict(static), **dict(zip(keys, children, strict=True)))


def _register_pytree(cls: type[TransformationDist]) -> None:
    jax.tree_util.register_pytree_node(
        cls,
        _flatten_transformation_dist,
        partial(_unflatten_transformation_dist, cls),
    )


class TransformationDist(tfd.Distribution):
    """
    Transformation distribution using a spline and optional parametric component.

    This combines a spline-based monotonically increasing transformation with a
    parametric distribution, using a reference distribution (default: standard
    normal) for mapping and likelihoods.

    Parameters
    ----------
    coef
        Coefficients for the spline basis, with shape
        ``batch_shape + (n_param,)``.
    bspline
        Spline object providing transformation and its inverse/derivative.
    parametric_distribution
        Parametric distribution class to include in the model.
    reference_distribution
        Reference distribution used for transformations; defaults to Normal(0, 1).
    validate_args
        Whether to validate input arguments.
    allow_nan_stats
        Whether to allow NaN statistics.
    name
        Name of the distribution.
    centered
        If True, center the intermediate spline distribution using its finite-bound \
        raw mean. This need not preserve the parametric response mean after a \
        nonlinear parametric inverse.
    scaled
        If True, scale the intermediate spline distribution using its finite-bound \
        raw variance. This need not preserve the parametric response variance after \
        a nonlinear parametric inverse.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for spline moments.
    integration_bounds
        Lower and upper bounds for raw spline moments used in centering/scaling. \
        Defaults to the first and last spline knot. Response moments integrate \
        the complete quantile function independently of these bounds, while \
        respecting the configured centering/scaling.
    response_moment_order
        Number of Gauss-Legendre points for response moments (default: 512). \
        Independent of spline quadrature; evaluated only when moments are requested. \
        Heavy tails may require a higher order and the optional response diagnostic.
    **parametric_distribution_kwargs
        Additional keyword arguments passed to the parametric distribution.

    Attributes
    ----------
    coef
        Coefficients for the spline basis.
    bspline
        Spline object used for transformations.
    reference_distribution
        Reference distribution used for mapping and likelihoods.
    parametric_distribution
        Instantiated parametric distribution if provided, else None.
    parametric_distribution_kwargs
        Keyword arguments used to construct the parametric distribution.
    centered
        Indicates whether centering is applied.
    scaled
        Indicates whether scaling is applied.

    Notes
    -----
    Like other TFP distributions, it supports `copy()`, batch slicing such as
    `dist[..., rows]`, and passing into `jax.jit`. Slicing takes the parameter
    properties of the parametric distribution, so a batched parameter that
    distribution does not declare cannot be sliced. The reference distribution
    is shared by all rows and must not have a batch shape for slicing.
    """

    def __init__(
        self,
        coef: Array,
        bspline: PTMSpline | OnionSpline,
        parametric_distribution: type[tfd.Distribution] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "TransformationDist",
        centered: bool = False,
        scaled: bool = False,
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        response_moment_order: int = 512,
        **parametric_distribution_kwargs,
    ):
        coef = jnp.asarray(coef)
        if not jnp.issubdtype(coef.dtype, jnp.floating):
            raise TypeError("Spline coefficients must have a floating-point dtype.")

        gauss_legendre_order = _validate_gauss_legendre_order(gauss_legendre_order)
        # Record the parametric keyword arguments at the top level, so that
        # type(self)(**self.parameters) rebuilds the distribution. TFP's copy(),
        # slicing and the pytree registration rely on this.
        parameters = dict(locals())
        parameters.update(parameters.pop("parametric_distribution_kwargs"))

        self.coef = coef
        self.parametric_distribution_kwargs = parametric_distribution_kwargs
        self.centered = centered
        self.scaled = scaled

        self.bspline = bspline
        self.knots = self.bspline.knots
        self.bspline._check_tfp_coef_core_shape(coef)
        self.gauss_legendre_order = gauss_legendre_order
        self.response_moment_order = _validate_gauss_legendre_order(
            response_moment_order, "response_moment_order"
        )
        if reference_distribution is None:
            self.reference_distribution = tfd.Normal(loc=0.0, scale=1.0)
        else:
            self.reference_distribution = reference_distribution

        self._spline_bijector = self.bspline.bijector(
            coef,
            centered=centered,
            scaled=scaled,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
            reference_distribution=self.reference_distribution,
            validate_args=validate_args,
        )
        self.integration_bounds = self._spline_bijector.integration_bounds
        self.integration_breaks = self._spline_bijector.integration_breaks

        if parametric_distribution is None and parametric_distribution_kwargs:
            raise ValueError(
                "Provided 'parametric_distribution_kwargs', but no value for"
                " 'parametric_distribution'."
            )

        if parametric_distribution is None:
            self.parametric_distribution = None
        else:
            self.parametric_distribution = parametric_distribution(
                **parametric_distribution_kwargs
            )

        super().__init__(
            dtype=coef.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _register_pytree(cls)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # Keyword arguments of the parametric distribution depend on the instance;
        # __getitem__ and the pytree registration take them from that
        # distribution.
        return {"coef": tfp_parameter_properties.ParameterProperties(event_ndims=1)}

    def _array_parameter_keys(self) -> tuple[str, ...]:
        parametric = tuple(
            key
            for key, value in self.parametric_distribution_kwargs.items()
            if _is_array_parameter(value)
        )
        return ("coef",) + parametric

    def __getitem__(self, slices: Any) -> Self:
        """Slice the batch dimensions, like other TFP distributions.

        Integer arrays select rows; axes of size one broadcast. The reference
        distribution is shared and left unchanged.

        Raises
        ------
        ValueError
            If the reference distribution has a batch shape, or a batched
            parameter of the parametric distribution lacks TFP parameter
            properties.
        """
        if tuple(self.reference_distribution.batch_shape) != ():
            raise ValueError(
                "Slicing a TransformationDist requires a reference distribution "
                "without batch shape."
            )
        event_ndims = {"coef": 1}
        properties: dict[str, Any] = {}
        if self.parametric_distribution is not None:
            try:
                properties = type(self.parametric_distribution).parameter_properties(
                    dtype=self.dtype
                )
            except NotImplementedError:
                properties = {}
        for key, value in self.parametric_distribution_kwargs.items():
            if key in properties:
                ndims = properties[key].event_ndims
                if callable(ndims):
                    ndims = ndims(self.parametric_distribution)
                event_ndims[key] = ndims
            elif getattr(value, "ndim", 0) > 0:
                raise ValueError(
                    f"Cannot slice the parametric distribution parameter {key!r}: "
                    f"{type(self.parametric_distribution).__name__} declares no TFP "
                    "parameter properties for it."
                )
        if not isinstance(slices, Sequence):
            slices = (slices,)
        batch_shape = tuple(int(dim) for dim in tuple(self.batch_shape))
        batch_shape_tensor = self.batch_shape_tensor()
        expected = jax.eval_shape(lambda: jnp.zeros(batch_shape)[tuple(slices)]).shape

        def sliced(keys: Sequence[str]) -> Self:
            parameters = dict(self.parameters)
            for key in keys:
                parameters[key] = tfp_slicing._slice_single_param(
                    parameters[key], event_ndims[key], slices, batch_shape_tensor
                )
            return type(self)(**parameters)

        # Parameters shared by all rows stay unsliced. Expanding a shared coef
        # to one row each would recompute the spline moments for every row.
        varying = [
            key
            for key, ndims in event_ndims.items()
            if any(
                dim != 1 for dim in jnp.shape(self.parameters[key])[: -ndims or None]
            )
        ]
        result = sliced(varying)
        if tuple(result.batch_shape) != tuple(expected):
            result = sliced(list(event_ndims))
        return result

    def _batch_shape_tuple(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in tuple(self.batch_shape))

    @property
    def spline_bijector(self) -> SplineBijector:
        """
        The spline transformation as a TFP bijector.

        The bijector comes from ``bspline.bijector(...)``, with this distribution's
        coefficients, centering, scaling, quadrature settings and reference
        distribution. Its ``inverse`` is the spline transformation and its
        ``forward`` is the inverse spline transformation. The parametric part is
        not included.
        """
        return self._spline_bijector

    # The bijector evaluates the spline with the coefficients' own batch shape.
    # With the full batch shape, per-observation parametric parameters would
    # broadcast shared coefficients to one copy per observation, and the inverse
    # would build one grid per observation. The cheap outputs are then broadcast
    # to the shape that the full batch shape gives.

    def _spline_output_shape(self, value: Array) -> tuple[int, ...]:
        _, sample_shape, _, result_batch_shape = self.bspline._tfp_value_layout(
            value, self._batch_shape_tuple()
        )
        return sample_shape + result_batch_shape

    def _response_moments(self, order: int | None = None) -> tuple[Array, Array]:
        """Integrate complete response quantiles, separately from spline normalization.

        These are numerical approximations over (0, 1), not certificates of moment
        existence. Heavy tails and finite-precision quantiles can require a higher
        order; use ``moment_quadrature_diagnostic(target="response")`` to compare
        resolutions. No additional integration is performed during fitting.
        """
        if order is None:
            order = self.response_moment_order
        nodes, weights = _gauss_legendre_nodes_and_weights(order, self.dtype)
        zero = jnp.asarray(0.0, dtype=self.dtype)
        one = jnp.asarray(1.0, dtype=self.dtype)
        probabilities = jnp.clip(
            (nodes + one) / 2,
            jnp.nextafter(zero, one),
            jnp.nextafter(one, zero),
        )
        shape = (-1,) + (1,) * len(self.batch_shape)
        values = self.quantile(jnp.reshape(probabilities, shape))
        weights = jnp.reshape(weights / 2, shape)
        mean = jnp.sum(weights * values, axis=0)
        variance = jnp.sum(weights * (values - mean) ** 2, axis=0)
        return mean, variance

    def _mean(self, **kwargs) -> Array:
        return self._response_moments()[0]

    def _variance(self, **kwargs) -> Array:
        return self._response_moments()[1]

    def _stddev(self, **kwargs) -> Array:
        return jnp.sqrt(self._variance(**kwargs))

    def _cdf(self, value: Array) -> Array | float:
        z = self._transformation(value)
        return self.reference_distribution.cdf(z)

    def _log_cdf(self, value: Array) -> Array | float:
        z = self._transformation(value)
        try:
            return self.reference_distribution.log_cdf(z)
        except (AttributeError, NotImplementedError):
            prob = self.reference_distribution.cdf(z)
            tiny = jnp.finfo(jnp.asarray(prob).dtype).tiny
            return jnp.log(jnp.clip(prob, tiny, 1.0))

    def _survival_function(self, value: Array, **kwargs) -> Array | float:
        z = self._transformation(value)
        try:
            return self.reference_distribution.survival_function(z)
        except (AttributeError, NotImplementedError):
            return 1.0 - self.reference_distribution.cdf(z)

    def _log_survival_function(self, value: Array, **kwargs) -> Array | float:
        z = self._transformation(value)
        try:
            return self.reference_distribution.log_survival_function(z)
        except (AttributeError, NotImplementedError):
            try:
                prob = self.reference_distribution.survival_function(z)
            except (AttributeError, NotImplementedError):
                prob = 1.0 - self.reference_distribution.cdf(z)
            tiny = jnp.finfo(jnp.asarray(prob).dtype).tiny
            return jnp.log(jnp.clip(prob, tiny, 1.0))

    # **kwargs matches TFP's typed override; reject overrides as before.
    def _log_prob(self, value: Array, **kwargs) -> Array | float:
        if kwargs:
            raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))!r}")
        z, logdet = self.transformation_and_logdet(value)
        return self.reference_distribution.log_prob(z) + logdet

    def _prob(self, value: Array) -> Array | float:
        return jnp.exp(self._log_prob(value))

    def _sample_n(
        self, n: int | Array, seed: KeyArray | None = None, **kwargs
    ) -> Array:
        shape = (n,) + self._batch_shape_tuple()
        # ensure 0 will be > 0 to avoid numerical instability
        eps = jnp.finfo(self.coef.dtype).eps
        u = jax.random.uniform(
            cast(Array, seed), shape=shape, minval=eps, maxval=1.0 - eps
        )

        return self._quantile(u)

    # **kwargs matches TFP's typed override; reject overrides as before.
    def _quantile(self, value: Array, **kwargs) -> Array:
        if kwargs:
            raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))!r}")
        z = self.reference_distribution.quantile(value)
        return self.inverse_transformation(z)

    def quantile_spline(self, value: Array) -> Array:
        """
        Quantile function using only the spline transformation.

        Parameters
        ----------
        value
            Quantile levels in (0, 1).

        Returns
        -------
            Values under the spline-only model at the requested quantiles.
        """
        z = self.reference_distribution.quantile(value)
        return self.inverse_transformation_spline(z)

    def _event_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        coef_shape = tf.TensorShape(self.bspline._tfp_coef_batch_shape(self.coef))

        if self.parametric_distribution is None:
            parametric_shape = tf.TensorShape([])
        else:
            parametric_shape = self.parametric_distribution.batch_shape

        return tf.broadcast_static_shape(coef_shape, parametric_shape)

    def _batch_shape_tensor(self, **kwargs):
        coef_shape = jnp.asarray(
            self.bspline._tfp_coef_batch_shape(self.coef), dtype=jnp.int32
        )

        if self.parametric_distribution is None:
            parametric_shape = jnp.asarray([], dtype=jnp.int32)
        else:
            parametric_shape = self.parametric_distribution.batch_shape_tensor()

        return tf.broadcast_dynamic_shape(coef_shape, parametric_shape)

    def log_prob_spline(self, value: Array):
        """
        Log probability under the spline-only transformation.

        Parameters
        ----------
        value
            Values at which to evaluate the log density.

        Returns
        -------
            Log probability evaluated using the spline transform.
        """
        z, logdet = self.transformation_and_logdet_spline(value)
        return self.reference_distribution.log_prob(z) + logdet

    def prob_spline(self, value: Array):
        """
        Probability under the spline-only transformation.

        Parameters
        ----------
        value
            Values at which to evaluate the density.

        Returns
        -------
            Probability density evaluated using the spline transform.
        """
        return jnp.exp(self.log_prob_spline(value))

    def cdf_spline(self, value: Array):
        """
        CDF under the spline-only transformation.

        Parameters
        ----------
        value
            Values at which to evaluate the CDF.

        Returns
        -------
            Cumulative distribution evaluated using the spline transform.
        """
        z = self._transformation_spline(value)
        return self.reference_distribution.cdf(z)

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        """
        Apply the parametric transformation and its log determinant.

        Parameters
        ----------
        value
            Input values.

        Returns
        -------
            A pair of transformed values and the corresponding log determinant.
        """
        if self.parametric_distribution is None:
            return value, jnp.zeros_like(value)

        value = jnp.asarray(value, dtype=self.dtype)
        F_apriori = self.parametric_distribution
        Fz = self.reference_distribution

        # Use jnp.finfo to get machine epsilon and min/max float values
        eps = jnp.finfo(value.dtype).eps
        tiny = jnp.finfo(value.dtype).tiny
        max_float = 1.0 - eps

        u = F_apriori.cdf(value)
        u = jnp.where(u >= 1.0, max_float, u)  # safeguard using max float
        u = jnp.where(u <= 0.0, tiny, u)  # safeguard using smallest positive float

        transf = Fz.quantile(u)
        logdet = F_apriori.log_prob(value) - Fz.log_prob(transf)

        return transf, logdet

    def transformation_parametric(self, value: Array) -> Array:
        """
        Apply the parametric transformation without computing its log determinant.
        """
        if self.parametric_distribution is None:
            return value

        value = jnp.asarray(value, dtype=self.dtype)
        F_apriori = self.parametric_distribution
        Fz = self.reference_distribution

        eps = jnp.finfo(value.dtype).eps
        tiny = jnp.finfo(value.dtype).tiny
        max_float = 1.0 - eps

        u = F_apriori.cdf(value)
        u = jnp.where(u >= 1.0, max_float, u)
        u = jnp.where(u <= 0.0, tiny, u)

        return Fz.quantile(u)

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        """
        Apply spline transformation with centering/scaling and compute logdet.

        Parameters
        ----------
        value
            Input values.

        Returns
        -------
            A pair of transformed values and the corresponding log determinant.
        """
        value = jnp.asarray(value, dtype=self.dtype)
        z, logdet = self.spline_bijector._inverse_and_log_det_jacobian(value)
        shape = self._spline_output_shape(value)
        return jnp.broadcast_to(z, shape), jnp.broadcast_to(logdet, shape)

    def _transformation_spline(self, value: Array) -> Array:
        value = jnp.asarray(value, dtype=self.dtype)
        z = self.spline_bijector._inverse(value)
        return jnp.broadcast_to(z, self._spline_output_shape(value))

    def _transformation(self, value: Array) -> Array:
        """Apply parametric then spline transforms without log determinants."""
        transf_param = self.transformation_parametric(value)
        return self._transformation_spline(transf_param)

    def transformation_and_logdet(self, value: Array) -> tuple[Array, Array]:
        """Apply parametric then spline transforms; return value and logdet."""
        transf_param, logdet_param = self.transformation_and_logdet_parametric(value)
        transf_spline, logdet_spline = self.transformation_and_logdet_spline(
            transf_param
        )

        logdet = logdet_param + logdet_spline

        return transf_spline, logdet

    def transformation_spline_mean(self) -> Array:
        """Finite-bound raw spline mean used for normalization, not a response mean."""
        return self.spline_bijector.mean_raw()

    def transformation_spline_variance(self, mean: Array | None = None) -> Array:
        """Finite-bound raw spline variance used for normalization."""
        return self.spline_bijector.variance_raw(mean)

    def moment_quadrature_diagnostic(
        self,
        reference_order: int | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        *,
        target: Literal["spline", "response"] = "spline",
    ) -> dict[str, Array]:
        """
        Compare raw spline or response moments against a higher-order rule.

        ``target="spline"`` (the default) checks the finite-bound moments used for
        centering/scaling. ``target="response"`` checks the full response moments
        at ``response_moment_order``. The returned arrays have coefficient batch
        shape for spline moments and full distribution batch shape for responses.

        This opt-in check adds no work to likelihood or moment calls. Agreement
        between resolutions does not prove that moments exist or that omitted
        spline tails / finite-precision response tails are negligible.
        """
        if target not in ("spline", "response"):
            raise ValueError("target must be 'spline' or 'response'.")
        if target == "spline":
            return self.spline_bijector.moment_quadrature_diagnostic(
                reference_order, rtol=rtol, atol=atol
            )

        order = self.response_moment_order
        if reference_order is None:
            reference_order = 2 * order
        reference_order = _validate_gauss_legendre_order(
            reference_order, "reference_order"
        )
        if reference_order <= order:
            raise ValueError(
                "reference_order must exceed the configured quadrature order."
            )

        return _moment_diagnostic(
            self._response_moments(order=order),
            self._response_moments(order=reference_order),
            rtol=rtol,
            atol=atol,
            dtype=self.dtype,
        )

    def inverse_transformation_spline(self, value: Array) -> Array:
        """
        Inverse of the spline transformation.

        Parameters
        ----------
        value
            Inputs on the reference scale.

        Returns
        -------
            Values on the original scale using the spline inverse.
        """
        value = jnp.asarray(value, dtype=self.dtype)
        y = self.spline_bijector._forward(value)
        return jnp.broadcast_to(y, self._spline_output_shape(value))

    def inverse_transformation_parametric(self, value: Array) -> Array:
        """
        Inverse of the parametric transformation.

        Parameters
        ----------
        value
            Inputs on the reference scale.

        Returns
        -------
            Values mapped back via the parametric distribution.
        """
        if self.parametric_distribution is None:
            return value

        # Use jnp.finfo to get machine epsilon and min/max float values
        eps = jnp.finfo(value.dtype).eps
        tiny = jnp.finfo(value.dtype).tiny
        max_float = 1.0 - eps

        u = self.reference_distribution.cdf(value)
        u = jnp.where(u >= 1.0, max_float, u)  # safeguard against numerical issues
        u = jnp.where(u <= 0.0, tiny, u)  # safeguard against numerical issues
        if isinstance(
            self.parametric_distribution, tfd.AutoCompositeTensorDistribution
        ):
            try:
                # Custom TFP subclasses may be registered but not flattenable.
                jax.tree_util.tree_flatten(self.parametric_distribution)
            except (TypeError, ValueError):
                pass
            else:
                return _composite_quantile(self.parametric_distribution, u)
        return self.parametric_distribution.quantile(u)

    def inverse_transformation(self, value: Array) -> Array:
        """
        Inverse transformation combining parametric and spline parts.

        Parameters
        ----------
        value
            Inputs on the reference scale.

        Returns
        -------
            Values on the original scale after both inverses.
        """
        y_tilde = self.inverse_transformation_spline(value)
        y = self.inverse_transformation_parametric(y_tilde)

        return y


_register_pytree(TransformationDist)


class LocScaleTransformationDist(TransformationDist):
    """
    Location–scale specialization of
    [TransformationDist][liesel_ptm.TransformationDist].

    Uses a Normal parametric layer with location ``loc`` and scale ``scale``,
    combined with a spline transformation and reference Normal(0, 1).

    Parameters
    ----------
    coef
        Coefficients for the spline basis.
    loc
        Location parameter for the Normal layer.
    scale
        Scale parameter for the Normal layer.
    bspline
        Spline object providing the transformation.
    validate_args
        Whether to validate input arguments.
    allow_nan_stats
        Whether to allow NaN statistics.
    name
        Name of the distribution.
    centered
        If True, the transformation is centered.
    scaled
        If True, the transformation is scaled.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for spline moments.
    integration_bounds
        Lower and upper bounds for raw spline moments used in centering/scaling. \
        Defaults to the first and last spline knot. Response moments integrate \
        the complete quantile function independently of these bounds, while \
        respecting the configured centering/scaling.
    response_moment_order
        Number of Gauss-Legendre points for response moments (default: 512). \
        Independent of spline quadrature; evaluated only when moments are requested. \
        Heavy tails may require a higher order and the optional response diagnostic.

    Notes
    -----
    Inherits public attributes from [TransformationDist][liesel_ptm.TransformationDist].
    """

    def __init__(
        self,
        coef: Array,
        loc: Array,
        scale: Array,
        bspline: PTMSpline | OnionSpline,
        parametric_distribution: type[tfd.Distribution] | None = tfd.Normal,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "LocScaleTransformationDist",
        centered: bool = False,
        scaled: bool = False,
        reference_distribution=tfd.Normal(loc=0.0, scale=1.0),
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        response_moment_order: int = 512,
    ) -> None:
        super().__init__(
            coef=coef,
            parametric_distribution=parametric_distribution,
            reference_distribution=reference_distribution,
            bspline=bspline,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            loc=loc,
            scale=scale,
            centered=centered,
            scaled=scaled,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
            response_moment_order=response_moment_order,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        properties = tfp_parameter_properties.ParameterProperties
        return {
            "coef": properties(event_ndims=1),
            "loc": properties(),
            "scale": properties(default_constraining_bijector_fn=tfb.Softplus),
        }

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        """
        Apply location–scale normalization and its log-determinant.

        Parameters
        ----------
        value
            Input values on the original scale.

        Returns
        -------
        transf, logdet
            Normalized values and corresponding log-determinant.
        """
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        transf = (value - self.parametric_distribution.mean()) / sd

        logdet = -jnp.log(sd)

        return transf, logdet

    def transformation_parametric(self, value: Array) -> Array:
        """
        Apply location–scale normalization without computing its log determinant.
        """
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        return (value - self.parametric_distribution.mean()) / sd

    def inverse_transformation_parametric(self, value: Array) -> Array:
        """
        Invert the location–scale normalization.

        Parameters
        ----------
        value
            Values on the normalized (reference) scale.

        Returns
        -------
        y
            Values mapped back to the original scale.
        """
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        m = self.parametric_distribution.mean()
        y = value * sd + m

        return y
