"""Liesel distribution node for PTM likelihoods."""

from functools import partial
from typing import Literal

import jax.numpy as jnp
import liesel.model as lsl
from jax.typing import ArrayLike

from .bspline import OnionSpline, PTMSpline
from .dist import LocScaleTransformationDist


class PTMDist(lsl.Dist):
    """Construct a PTM likelihood from location, scale and shape predictors.

    Parameters
    ----------
    knots
        Knot sequence for the selected spline.
    loc, scale, shape
        Liesel variables/nodes or values supplying location, positive scale and
        shape coefficients. Shared coefficients have shape ``(n_param,)``;
        covariate-dependent coefficients have shape ``(n_obs, n_param)``.
    centered, scaled
        Whether to center and scale the spline distribution, respectively.
        Leave both false for the computed-intercept :class:`~liesel_ptm.LegacyPTM`
        workflow; use both true for the modern mean/standard-deviation interpretation.
    trafo_lambda
        PTM spline transition width relative to knot spacing.
    bspline
        ``"ptm"`` or ``"onion"``. The spline is constructed once and reused by
        the distribution node.
    trafo_target_slope
        PTM tail behavior: transition to unit slope (``"identity"``) or continue
        with the boundary slope (``"continue_linearly"``).
    **kwargs
        Forwarded to :class:`liesel.model.Dist`, including arguments for
        :class:`~liesel_ptm.LocScaleTransformationDist`.

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
