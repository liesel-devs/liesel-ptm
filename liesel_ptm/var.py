from typing import Any, Self

import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.contrib.splines import pspline_penalty
from liesel_gam.constraint import penalty_to_unit_design

Array = Any
InferenceTypes = Any


class PTMCoef(lsl.Var):
    """Coefficient for PTM transformation splines.

    This class wraps latent spline coefficients with penalty handling and
    provides several factory constructors for common penalty/prior setups.

    Parameters
    ----------
    scale
        A scale variable used for the coefficient prior.
    penalty
        Penalty matrix applied to the spline coefficients.

    Attributes
    ----------
    Z
        Reparameterization matrix applied to latent coefficients.
    penalty
        The (possibly scaled) penalty matrix used for priors.
    latent_coef
        Underlying latent coefficient variable.
    nparam
        Number of spline parameters (after transformations).
    """

    def __init__(
        self,
        scale: lsl.Var,
        penalty: Array,
        inference: InferenceTypes = None,
        name: str = "",
        scale_penalty: bool = True,
        diagonalize_penalty: bool = True,
        Z: Array | None = None,
        role: str = "transformation_coef",
        noncentered: bool = False,
    ):
        nparam = jnp.shape(penalty)[-1]

        if Z is None:
            Z = jnp.eye(nparam)

        if scale_penalty:
            scale_pen = jnp.linalg.norm(penalty, ord=jnp.inf)
            penalty = penalty / scale_pen

        if diagonalize_penalty:
            rank = jnp.linalg.matrix_rank(penalty)
            Z2 = penalty_to_unit_design(penalty, rank=rank)
            penalty = jnp.eye(nparam)
            penalty = penalty.at[rank:, rank:].set(0.0)
        else:
            Z2 = jnp.eye(nparam)

        self.Z = Z @ Z2
        self.penalty = penalty

        if jnp.allclose(self.penalty, jnp.eye(nparam)):
            prior = lsl.Dist(
                tfd.Normal,
                loc=jnp.zeros(nparam),
                scale=scale,
            )
        else:
            prior = lsl.Dist(
                gam.MultivariateNormalSingular,
                loc=jnp.asarray(0.0),
                scale=scale,
                penalty=penalty,
                penalty_rank=jnp.asarray(jnp.linalg.matrix_rank(penalty)),
            )

        self.latent_coef = lsl.Var.new_param(
            jnp.zeros((nparam,)),
            distribution=prior,
            inference=inference,
            name=name + "_latent",
        )
        self.latent_coef.role = role
        self.scale = scale
        self.noncentered = noncentered

        if noncentered:
            assert self.latent_coef.dist_node is not None
            self.latent_coef.dist_node["scale"] = lsl.Value(1.0)

            def compute_coef1(latent_coef, scale):
                coef = jnp.einsum("pj,...j->...p", self.Z, scale * latent_coef)
                return jnp.expand_dims(coef, -2)

            coef_calc = lsl.Calc(compute_coef1, self.latent_coef, self.scale)
        else:

            def compute_coef2(latent_coef):
                coef = jnp.einsum("pj,...j->...p", self.Z, latent_coef)
                return jnp.expand_dims(coef, -2)

            coef_calc = lsl.Calc(compute_coef2, self.latent_coef)
        super().__init__(coef_calc, name=name)

        self.nparam = nparam
        self.update()

    @classmethod
    def new_ridge(
        cls,
        knots: Array,
        scale: lsl.Var,
        inference: InferenceTypes = None,
        name: str = "",
        role: str = "transformation_coef",
    ) -> Self:
        """Create a ridge-penalized coefficient set.

        Parameters
        ----------
        knots
            Knot vector for the spline basis. This method is compatible with knots
            :class:`.PTMKnots`, but not with :class:`.OnionKnots`.
        scale
            Scale variable or node used for the coefficient prior.
        inference
            Optional inference specification for the latent parameter, a
            :class:`liesel.goose.MCMCSpec` object.
        name
            Optional base name for created variables.
        role
            Role assigned to the latent coefficient variable.

        Returns
        -------
        PTMCoef
            Configured coefficient with ridge penalty.
        """
        nparam = len(knots) - 4 - 1
        penalty = jnp.eye(nparam)
        return cls(
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            scale_penalty=False,
            diagonalize_penalty=False,
            role=role,
        )

    @classmethod
    def new_rw1_sumzero(
        cls,
        knots: Array,
        scale: lsl.Var,
        inference: InferenceTypes = None,
        name: str = "",
        scale_penalty: bool = True,
        diagonalize_penalty: bool = True,
        role: str = "transformation_coef",
        noncentered: bool = False,
    ) -> Self:
        """Create RW1 (first-order random walk) coefficients with sum-to-zero.

        Parameters
        ----------
        knots
            Knot vector for the spline basis.
        scale
            Scale variable used for the coefficient prior.
        inference
            Optional inference specification for the latent parameter, a
            :class:`liesel.goose.MCMCSpec` object.
        name
            Optional base name for created variables.
        scale_penalty
            Whether to scale the penalty matrix to unit infinity norm.
        diagonalize_penalty
            Whether to diagonalize the penalty via a eigenvalue decomposition.
        role
            Role assigned to the latent coefficient variable.

        Returns
        -------
        PTMCoef
            Configured coefficient with RW1 sum-to-zero constraint.
        """

        nparam = len(knots) - 4 - 1
        penalty = pspline_penalty(d=nparam, diff=1)

        if scale_penalty:
            scale_pen = jnp.linalg.norm(penalty, ord=jnp.inf)
            penalty = penalty / scale_pen

        if diagonalize_penalty:
            rank = jnp.linalg.matrix_rank(penalty)
            Z = penalty_to_unit_design(penalty, rank=rank)[:, :-1]
            penalty = jnp.eye(nparam - 1)

        return cls(
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            scale_penalty=scale_penalty,
            diagonalize_penalty=diagonalize_penalty,
            Z=Z,
            role=role,
            noncentered=noncentered,
        )

    @classmethod
    def new_rw1_sumzero_ig(
        cls,
        knots: Array,
        ig_concentration: float,
        ig_scale: float,
        inference: InferenceTypes = None,
        name: str = "",
        scale_penalty: bool = True,
        diagonalize_penalty: bool = True,
        role: str = "transformation_coef",
    ) -> Self:
        """Create RW1 sum-to-zero coefficients with an inverse-gamma prior on the
        random walk variance.

        Parameters
        ----------
        knots
            Knot vector for the spline basis.
        ig_concentration
            Concentration parameter for the inverse-gamma prior.
        ig_scale
            Scale parameter for the inverse-gamma prior.
        inference
            Optional inference specification for the latent parameter, a
            :class:`liesel.goose.MCMCSpec` object. By default, uses a Gibbs sampler.
        name
            Optional base name for created variables.
        scale_penalty
            Whether to scale the penalty matrix to unit infinity norm.
        diagonalize_penalty
            Whether to diagonalize the penalty via a eigenvalue decomposition.
        role
            Role assigned to the latent coefficient variable.

        Returns
        -------
        PTMCoef
            Configured coefficient with an inverse-gamma variance parameter.
        """

        nparam = len(knots) - 4 - 1
        penalty = pspline_penalty(d=nparam, diff=1)

        if scale_penalty:
            scale_pen = jnp.linalg.norm(penalty, ord=jnp.inf)
            penalty = penalty / scale_pen

        if diagonalize_penalty:
            rank = jnp.linalg.matrix_rank(penalty)
            Z = penalty_to_unit_design(penalty, rank=rank)[:, :-1]
            penalty = jnp.eye(nparam - 1)

        scale = gam.ScaleIG(
            value=1.0,
            concentration=ig_concentration,
            scale=ig_scale,
            name=name + "_scale",
        )

        coef = cls(
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            scale_penalty=scale_penalty,
            diagonalize_penalty=diagonalize_penalty,
            Z=Z,
            role=role,
        )

        scale.setup_gibbs_inference(coef.latent_coef, penalty=coef.penalty)

        return coef

    @classmethod
    def new_rw1_sumzero_wb(
        cls,
        knots: Array,
        wb_scale: float,
        inference: InferenceTypes = None,
        scale_inference: InferenceTypes = None,
        name: str = "",
        scale_penalty: bool = True,
        diagonalize_penalty: bool = True,
        role: str = "transformation_coef",
        scale_bijector: tfb.Bijector = tfb.Exp(),
    ) -> Self:
        """Create RW1 sum-to-zero coefficients with a Weibull prior on the random
        walk variance.

        Parameters
        ----------
        knots
            Knot vector for the spline basis.
        wb_scale
            Scale parameter for the Weibull prior.
        inference
            Optional inference specification for the latent parameter.
        scale_inference
            Optional inference for the scale variable. Will be passed to the
            inference argument of :func:`liesel_gam.scale_wb`, thus acting on the
            level of the variance paramter (not the scale parameter).
        name
            Optional base name for created variables.
        scale_penalty
            Whether to scale the penalty matrix to unit infinity norm.
        diagonalize_penalty
            Whether to diagonalize the penalty via a eigenvalue decomposition.
        role
            Role assigned to the latent coefficient variable.
        scale_bijector
            Optional bijector applied to the scale variable. Will be passed to the
            bijector argument of :func:`liesel_gam.scale_wb`, thus acting on the
            level of the variance paramter (not the scale parameter).

        Returns
        -------
        PTMCoef
            Configured coefficient with a Weibull variance parameter.
        """

        nparam = len(knots) - 4 - 1
        penalty = pspline_penalty(d=nparam, diff=1)
        if scale_penalty:
            scale_pen = jnp.linalg.norm(penalty, ord=jnp.inf)
            penalty = penalty / scale_pen

        if diagonalize_penalty:
            rank = jnp.linalg.matrix_rank(penalty)
            Z = penalty_to_unit_design(penalty, rank=rank)[:, :-1]
            penalty = jnp.eye(nparam - 1)

        scale = gam.scale_wb(
            value=1.0,
            scale=wb_scale,
            name=name + "_scale",
            inference=scale_inference,
            bijector=scale_bijector,
        )

        coef = cls(
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            scale_penalty=scale_penalty,
            diagonalize_penalty=diagonalize_penalty,
            Z=Z,
            role=role,
        )

        return coef

    @classmethod
    def new_rw1_fromzero(
        cls,
        knots: Array,
        scale: lsl.Var,
        inference: InferenceTypes = None,
        name: str = "",
        scale_penalty: bool = True,
        diagonalize_penalty: bool = True,
        role: str = "transformation_coef",
        noncentered: bool = False,
    ) -> Self:
        """Create RW1 coefficients anchored at zero.

        Parameters
        ----------
        knots
            Knot vector for the spline basis.
        scale
            Scale variable used for the coefficient prior.
        inference
            Optional inference specification for the latent parameter.
        name
            Optional base name for created variables.
        scale_penalty
            Whether to scale the penalty matrix to unit infinity norm.
        diagonalize_penalty
            Whether to diagonalize the penalty via a eigenvalue decomposition.
        role
            Role assigned to the latent coefficient variable.

        Returns
        -------
        PTMCoef
            Configured coefficient with RW1 anchored at zero.
        """

        nparam = len(knots) - 11
        penalty = pspline_penalty(d=nparam + 1, diff=1)
        penalty = penalty[1:, 1:]

        return cls(
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            scale_penalty=scale_penalty,
            diagonalize_penalty=diagonalize_penalty,
            role=role,
            noncentered=noncentered,
        )

    @classmethod
    def new_rw1_fromzero_wb(
        cls,
        knots: Array,
        wb_scale: float,
        inference: InferenceTypes = None,
        scale_inference: InferenceTypes = None,
        name: str = "",
        scale_penalty: bool = True,
        diagonalize_penalty: bool = True,
        role: str = "transformation_coef",
        scale_bijector: tfb.Bijector = tfb.Exp(),
        noncentered: bool = False,
    ) -> Self:
        """Create RW1 sum-to-zero coefficients with a Weibull prior on the random
        walk variance.

        Parameters
        ----------
        knots
            Knot vector for the spline basis.
        wb_scale
            Scale parameter for the Weibull prior on variance.
        inference
            Optional inference specification for the latent parameter.
        scale_inference
            Optional inference for the scale variable. Will be passed to the
            inference argument of :func:`liesel_gam.scale_wb`, thus acting on the
            level of the variance paramter (not the scale parameter).
        name
            Optional base name for created variables.
        scale_penalty
            Whether to scale the penalty matrix to unit infinity norm.
        diagonalize_penalty
            Whether to diagonalize the penalty via a eigenvalue decomposition.
        role
            Role assigned to the latent coefficient variable.
        scale_bijector
            Optional bijector applied to the scale variable. Will be passed to the
            bijector argument of :func:`liesel_gam.scale_wb`, thus acting on the
            level of the variance paramter (not the scale parameter).

        Returns
        -------
        PTMCoef
            Configured coefficient container with Weibull variance parameter.
        """

        nparam = len(knots) - 11
        penalty = pspline_penalty(d=nparam + 1, diff=1)
        penalty = penalty[1:, 1:]

        scale = gam.scale_wb(
            value=1.0,
            scale=wb_scale,
            name=name + "_scale",
            inference=scale_inference,
            bijector=scale_bijector,
        )

        return cls(
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            scale_penalty=scale_penalty,
            diagonalize_penalty=diagonalize_penalty,
            role=role,
            noncentered=noncentered,
        )

    @classmethod
    def new_rw1_fromzero_ig(
        cls,
        knots: Array,
        ig_concentration: float,
        ig_scale: float,
        inference: InferenceTypes = None,
        name: str = "",
        scale_penalty: bool = True,
        diagonalize_penalty: bool = True,
        role: str = "transformation_coef",
        noncentered: bool = False,
    ) -> Self:
        """Create from-zero RW1 coefficients with an inverse-gamma prior on the
        variance parameter.

        Parameters
        ----------
        knots
            Knot vector for the spline basis.
        ig_concentration
            Concentration parameter for the inverse-gamma prior on scale.
        ig_scale
            Scale parameter for the inverse-gamma prior on scale.
        inference
            Optional inference specification for the latent parameter, a
            :class:`liesel.goose.MCMCSpec` object. By default, uses a Gibbs sampler.
        name
            Optional base name for created variables.
        scale_penalty
            Whether to scale the penalty matrix to unit infinity norm.
        diagonalize_penalty
            Whether to diagonalize the penalty via a eigenvalue decomposition.
        role
            Role assigned to the latent coefficient variable.

        Returns
        -------
        PTMCoef
            Configured coefficient with inverse-gamma prior on the variance parameter.
        """

        if noncentered:
            raise ValueError(
                "Inverse-gamma Gibbs updates require centered coefficients."
            )

        nparam = len(knots) - 11
        penalty = pspline_penalty(d=nparam + 1, diff=1)
        penalty = penalty[1:, 1:]

        scale = gam.ScaleIG(
            value=1.0,
            concentration=ig_concentration,
            scale=ig_scale,
            name=name + "_scale",
        )

        coef = cls(
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            scale_penalty=scale_penalty,
            diagonalize_penalty=diagonalize_penalty,
            role=role,
            noncentered=noncentered,
        )

        scale.setup_gibbs_inference(coef.latent_coef, penalty=coef.penalty)

        return coef
