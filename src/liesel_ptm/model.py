"""PTM models and their Liesel likelihood node."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from numbers import Integral, Real
from typing import Literal

import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
from jax.typing import ArrayLike

from .bspline import OnionKnots, OnionSpline, PTMSpline
from .dist import LocScaleTransformationDist


@dataclass
class PTMPredictors:
    """Native liesel-gam predictors exposed by [PTM][liesel_ptm.PTM].

    Add terms with `+=`, `append` or `extend`. In a
    [LegacyPTM][liesel_ptm.LegacyPTM], location and scale exclude computed intercepts;
    scale still applies an exponential inverse link.
    """

    loc: gam.AdditivePredictor
    """Location predictor; identity link by default."""
    scale: gam.AdditivePredictor
    """Positive scale predictor; access log scale through `linear_predictor`."""
    shape: gam.MVAdditivePredictor
    """Shape coefficients, including their structured intercept."""


def _validate_geometry(
    nshape: int, bounds: tuple[float, float], *, minimum: int
) -> None:
    if isinstance(nshape, bool) or not isinstance(nshape, Integral) or nshape < minimum:
        raise ValueError(f"nshape must be an integer of at least {minimum}.")
    interval = np.asarray(bounds, dtype=float)
    if (
        interval.shape != (2,)
        or not np.isfinite(interval).all()
        or not interval[0] < interval[1]
    ):
        raise ValueError("bounds must contain two finite, strictly increasing values.")


def _make_response(y: ArrayLike | Sequence[float] | lsl.Var) -> lsl.Var:
    response = y if isinstance(y, lsl.Var) else lsl.Var.new_obs(y, name="y")
    if (
        response.model is not None
        or response.dist_node is not None
        or not response.observed
    ):
        raise ValueError(
            "y must be an unowned observed variable without a distribution."
        )
    return response


@contextmanager
def _response_distribution(
    model: lsl.Model, response: lsl.Var, distribution: lsl.Dist
) -> Generator[None]:
    try:
        response.dist_node = distribution
        yield
    except Exception:
        # Liesel may assign node ownership before failing to finish the graph.
        for node in getattr(model, "nodes", {}).values():
            if node.model is model:
                node._unset_model()
        response.dist_node = None
        raise


def _check_intercept_free(predictor: gam.AdditivePredictor) -> None:
    if isinstance(predictor.intercept, lsl.Var) or jnp.any(predictor.intercept.value):
        raise ValueError(f"{predictor.name!r} must be intercept-free.")


def _check_scale(scale: gam.AdditivePredictor, *, legacy: bool) -> None:
    # Liesel's deep copies do not preserve JAX callable identity. Nonzero probes
    # also distinguish exp(2*x) from exp(x) when the predictor starts at zero.
    probe = jnp.array([-1.0, 0.0, 1.0])
    if legacy and (
        not isinstance(scale.value_node, lsl.Calc)
        or not jnp.allclose(scale.value_node.function(probe), jnp.exp(probe))
    ):
        raise ValueError("LegacyPTM scale must use inv_link=jnp.exp.")
    if not legacy and isinstance(scale.value_node, lsl.Calc):
        values = scale.value_node.function(probe)
        if not jnp.all(jnp.isfinite(values) & (values > 0)):
            raise ValueError(
                "scale requires a positive inverse link, such as exp or softplus."
            )
    if not jnp.all(jnp.isfinite(scale.value) & (scale.value > 0)):
        raise ValueError("scale must have finite, positive values.")


def _make_predictors(
    nshape: int,
    loc: gam.AdditivePredictor | None,
    scale: gam.AdditivePredictor | None,
    shape: gam.MVAdditivePredictor | None,
    shape_scale: lsl.Var | gam.VarIGPrior | float | None,
    *,
    legacy: bool,
) -> PTMPredictors:
    if loc is None:
        loc = gam.AdditivePredictor(
            "loc_terms" if legacy else "loc", intercept=not legacy
        )
    if scale is None:
        scale = gam.AdditivePredictor(
            "scale_terms" if legacy else "scale", inv_link=jnp.exp, intercept=not legacy
        )
    _check_scale(scale, legacy=legacy)
    if isinstance(shape_scale, bool):
        raise TypeError("shape_scale must be a positive real number or a native scale.")
    if isinstance(shape_scale, Real):
        shape_scale = float(shape_scale)
        if not np.isfinite(shape_scale) or shape_scale <= 0:
            raise ValueError("shape_scale must be finite and positive.")
    if shape is None:
        shape = gam.MVAdditivePredictor.from_random_walk(
            "shape",
            ndim=nshape,
            intercept_scale=(
                gam.scale_wb(0.1, scale=0.05, name="shape_scale")
                if shape_scale is None
                else shape_scale
            ),
        ).constrain("sumzero_coef")
    elif shape_scale is not None:
        raise ValueError("Set shape_scale on the supplied shape predictor instead.")
    if shape.ndim != nshape or shape.constraint != "sumzero_coef":
        raise ValueError(
            "shape must have nshape dimensions and the sumzero_coef constraint."
        )
    return PTMPredictors(loc, scale, shape)


class PTM(lsl.Model):
    """PTM with location, scale and optional covariate-dependent shape effects.

    Uses the onion spline and affine standardization. Keep only the shape intercept
    for a fixed-shape PTM, or add shape terms for a conditional-shape PTM.

    Parameters
    ----------
    y
        Response values, or an observed `liesel.model.Var` without a
        distribution. An array becomes a response variable named `y`.
    nshape
        Number of shape coefficients before constraints; an integer of at least two.
    bounds
        Flexible interval on the standardized residual scale. Cubic, equidistant
        [OnionKnots][liesel_ptm.OnionKnots] are constructed internally.
    loc
        Optional native location predictor, retained by identity. Defaults to an
        identity link with a flat-prior intercept initialized at zero.
    scale
        Optional native scale predictor, retained by identity. Defaults to an
        exponential inverse link with a flat-prior log-scale intercept initialized
        at zero. Supplied scales require a positive inverse link, such as exp or
        softplus, and finite positive initial values. Identity links are rejected
        even when the starting scale is positive.
    shape
        Optional native shape predictor, retained by identity. Defaults to a
        predictor named `shape`: a first-order random walk constrained with
        `sumzero_coef`, with a
        zero-initialized structured intercept and `nshape - 1` latent dimensions.
        A supplied shape must have `nshape` dimensions and the same constraint.
    shape_scale
        Scale of the automatic shape intercept: a native scale variable,
        `liesel_gam.VarIGPrior`, or a fixed positive real. `None` creates
        `gam.scale_wb(0.1, scale=0.05)`: initial standard deviation 0.1 and an
        unclipped Weibull(1/2, 0.05) prior on its square. Configure this on the
        predictor itself when supplying `shape`.

    Raises
    ------
    ValueError
        If the shape dimension or interval is invalid, a supplied shape has an
        incompatible dimension/constraint, or `y` already belongs to a model or
        has a distribution.

    Notes
    -----
    Location and scale are the conditional mean and standard deviation. Add terms
    after initialization, then create ordinary Liesel optimization, VI or MCMC
    objects. Graph growth and locking follow `liesel.model.Model` directly.

    See Also
    --------
    [LegacyPTM][liesel_ptm.LegacyPTM]
        Legacy transformation with computed location and scale intercepts.
    [PTMDist][liesel_ptm.PTMDist]
        Likelihood node for constructing a model manually.

    Examples
    --------
    >>> import liesel_gam as gam
    >>> import liesel_ptm as ptm
    >>> model = ptm.PTM([0.1, -0.2, 0.4], nshape=4)
    >>> terms = gam.TermBuilder.from_dict({"x": [-1.0, 0.0, 1.0]})
    >>> model.pred.loc += terms.slin("x", scale=1.0)
    >>> model.pred.shape.name
    'shape'
    >>> model.pred.shape.ndim, model.pred.shape.latent_ndim
    (4, 3)
    """

    pred: PTMPredictors
    """Location, scale and shape predictors; terms can be added after initialization."""
    response: lsl.Var
    """Response carrying the likelihood; use for distribution summaries and plots."""

    def __init__(
        self,
        y: ArrayLike | Sequence[float] | lsl.Var,
        *,
        nshape: int,
        bounds: tuple[float, float] = (-4.0, 4.0),
        loc: gam.AdditivePredictor | None = None,
        scale: gam.AdditivePredictor | None = None,
        shape: gam.MVAdditivePredictor | None = None,
        shape_scale: lsl.Var | gam.VarIGPrior | float | None = None,
    ) -> None:
        _validate_geometry(nshape, bounds, minimum=2)
        response = _make_response(y)
        self.pred = _make_predictors(
            nshape, loc, scale, shape, shape_scale, legacy=False
        )
        knots = OnionKnots(*bounds, nparam=nshape)
        distribution = PTMDist(
            knots.knots,
            loc=self.pred.loc,
            scale=self.pred.scale,
            shape=self.pred.shape,
            bspline="onion",
            centered=True,
            scaled=True,
        )
        self.response = response
        with _response_distribution(self, response, distribution):
            super().__init__(response)


class PTMDist(lsl.Dist):
    """Construct a PTM likelihood from location, scale and shape predictors.

    Parameters
    ----------
    knots
        Knot sequence for the selected spline.
    loc, scale, shape
        Liesel variables/nodes or values supplying location, positive scale and
        shape coefficients. Shared coefficients have shape `(n_param,)`;
        covariate-dependent coefficients have shape `(n_obs, n_param)`.
    centered, scaled
        Whether to center and scale the spline distribution, respectively.
        Leave both false for the computed-intercept [LegacyPTM][liesel_ptm.LegacyPTM]
        workflow; use both true for the current mean/standard-deviation interpretation.
    trafo_lambda
        PTM spline transition width relative to knot spacing.
    bspline
        `"ptm"` for the legacy transformation or `"onion"` for the current
        fixed-shape and conditional-shape PTMs. Covariate-dependent shape requires
        the onion spline. The spline is constructed once and reused by the node.
    trafo_target_slope
        PTM tail behavior: transition to unit slope (`"identity"`) or continue
        with the boundary slope (`"continue_linearly"`).
    **kwargs
        Forwarded to `liesel.model.Dist`, including arguments for
        [LocScaleTransformationDist][liesel_ptm.LocScaleTransformationDist].

    Raises
    ------
    ValueError
        If the spline or tail-slope option is unsupported.
    """

    def __init__(
        self,
        knots: ArrayLike,
        loc: lsl.Var | lsl.Node | ArrayLike,
        scale: lsl.Var | lsl.Node | ArrayLike,
        shape: lsl.Var | lsl.Node | ArrayLike,
        centered: bool = False,
        scaled: bool = False,
        trafo_lambda: float = 0.1,
        bspline: Literal["ptm", "onion"] = "ptm",
        trafo_target_slope: Literal["identity", "continue_linearly"] = "identity",
        **kwargs,
    ) -> None:
        knots = jnp.asarray(knots)
        if trafo_target_slope not in ("identity", "continue_linearly"):
            raise ValueError(f"Unsupported tail slope: {trafo_target_slope!r}.")
        if bspline == "ptm":
            spline = PTMSpline(
                knots,
                eps=trafo_lambda,
                continue_linearly=trafo_target_slope == "continue_linearly",
            )
        elif bspline == "onion":
            spline = OnionSpline(knots)
        else:
            raise ValueError(f"Unsupported spline: {bspline!r}; use 'ptm' or 'onion'.")
        constructor = partial(
            LocScaleTransformationDist,
            bspline=spline,
            centered=centered,
            scaled=scaled,
        )
        super().__init__(constructor, loc=loc, scale=scale, coef=shape, **kwargs)
