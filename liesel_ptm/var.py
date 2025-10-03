from typing import Any, Literal, Self

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm.gam as gam

from .constraint import mixed_model
from .gam.kernel import init_star_ig_gibbs as init_star_ig_gibbs2
from .kernel import init_star_ig_gibbs
from .penalty import Penalty

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
            Z2 = mixed_model(penalty, rank=rank)
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
                return jnp.einsum("pj,...j->...p", self.Z, scale * latent_coef)

            coef_calc = lsl.Calc(compute_coef1, self.latent_coef, self.scale)
        else:

            def compute_coef2(latent_coef):
                return jnp.einsum("pj,...j->...p", self.Z, latent_coef)

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
        penalty = Penalty.pspline(nparam=nparam, random_walk_order=1)

        if scale_penalty:
            scale_pen = jnp.linalg.norm(penalty, ord=jnp.inf)
            penalty = penalty / scale_pen

        if diagonalize_penalty:
            rank = jnp.linalg.matrix_rank(penalty)
            Z = mixed_model(penalty, rank=rank)[:, :-1]
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
        penalty = Penalty.pspline(nparam=nparam, random_walk_order=1)

        if scale_penalty:
            scale_pen = jnp.linalg.norm(penalty, ord=jnp.inf)
            penalty = penalty / scale_pen

        if diagonalize_penalty:
            rank = jnp.linalg.matrix_rank(penalty)
            Z = mixed_model(penalty, rank=rank)[:, :-1]
            penalty = jnp.eye(nparam - 1)

        scale = ScaleInverseGamma(
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

        scale.variance_param.inference = gs.MCMCSpec(
            init_star_ig_gibbs, kernel_kwargs={"coef": coef}
        )

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
        scale_bijector: tfb.Bijector | None = tfb.Exp(),
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
            inference argument of :class:`.ScaleWeibull`, thus acting on the
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
            bijector argument of :class:`.ScaleWeibull`, thus acting on the
            level of the variance paramter (not the scale parameter).

        Returns
        -------
        PTMCoef
            Configured coefficient with a Weibull variance parameter.
        """

        nparam = len(knots) - 4 - 1
        penalty = Penalty.pspline(nparam=nparam, random_walk_order=1)
        if scale_penalty:
            scale_pen = jnp.linalg.norm(penalty, ord=jnp.inf)
            penalty = penalty / scale_pen

        if diagonalize_penalty:
            rank = jnp.linalg.matrix_rank(penalty)
            Z = mixed_model(penalty, rank=rank)[:, :-1]
            penalty = jnp.eye(nparam - 1)

        scale = ScaleWeibull(
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
        penalty = Penalty.pspline(nparam=nparam + 1, random_walk_order=1)
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
        scale_bijector: tfb.Bijector | None = tfb.Exp(),
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
            inference argument of :class:`.ScaleWeibull`, thus acting on the
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
            bijector argument of :class:`.ScaleWeibull`, thus acting on the
            level of the variance paramter (not the scale parameter).

        Returns
        -------
        PTMCoef
            Configured coefficient container with Weibull variance parameter.
        """

        nparam = len(knots) - 11
        penalty = Penalty.pspline(nparam=nparam + 1, random_walk_order=1)
        penalty = penalty[1:, 1:]

        scale = ScaleWeibull(
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

        nparam = len(knots) - 11
        penalty = Penalty.pspline(nparam=nparam + 1, random_walk_order=1)
        penalty = penalty[1:, 1:]

        scale = ScaleInverseGamma(
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

        scale.variance_param.inference = gs.MCMCSpec(
            init_star_ig_gibbs2,
            kernel_kwargs={"coef": coef, "scale": scale, "penalty": penalty},
        )

        return coef


class ScaleWeibull(lsl.Var):
    """
    A variable with a Weibull prior on its square.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter.
    scale
        Scale parameter.
    name
        Name of the variable.
    inference
        Inference type.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.
    clip_min
        Values very close to zero will be soft-clipped to this value to avoid numerical
        instability. For 32-bit floats, we use a default of ``jnp.sqrt(jnp.exp(-9.0))``;
        for 64-bit floats we use ``jnp.sqrt(jnp.exp(-11.0))``.

    Attributes
    ----------
    variance_param
        The internal variance parameter (square of the reported scale). This is
        an ``lsl.Var`` created as the latent variance parameter.
    bijector
        The bijector passed to the constructor (or ``None``). When present,
        the bijector was applied to the variance parameter and the public
        variable becomes a deterministic transform of that parameter.
    """

    def __init__(
        self,
        value: Array,
        scale: float | lsl.Var | lsl.Node,
        concentration: float | lsl.Var | lsl.Node = 0.5,
        name: str = "",
        inference: InferenceTypes = None,
        bijector: tfb.Bijector | None = None,
        role: str = "hyperparam",
        clip_min: Literal["default"] | float | Array = "default",
    ):
        value = jnp.asarray(value)
        if isinstance(scale, float):
            scale = jnp.asarray(scale, dtype=value.dtype)  # type: ignore
        if not isinstance(scale, lsl.Node | lsl.Var):
            scale = lsl.Value(scale, _name=f"{name}_scale")  # type: ignore

        if isinstance(concentration, float):
            concentration = jnp.asarray(concentration, dtype=value.dtype)  # type: ignore
        if not isinstance(concentration, lsl.Node | lsl.Var):
            concentration = lsl.Value(concentration, _name=f"{name}_concentration")  # type: ignore

        prior = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        self.variance_param = lsl.Var.new_param(
            jnp.square(value), prior, inference=inference, name=f"{name}_square"
        )
        self.variance_param.role = role

        if clip_min == "default":
            if value.dtype == jnp.dtype("float32"):
                clip_min_real = jnp.array(-9.0, dtype=value.dtype)
                clip_min_pos = jnp.exp(clip_min_real)
            elif value.dtype == jnp.dtype("float64"):
                clip_min_real = jnp.array(-11.0, dtype=value.dtype)
                clip_min_pos = jnp.exp(clip_min_real)
            else:
                raise TypeError(f"{value.dtype} not recognized.")
        else:
            clip_min_pos = jnp.square(jnp.asarray(clip_min))
            clip_min_real = (
                bijector.inverse(clip_min_pos) if bijector is not None else clip_min_pos
            )

        super().__init__(
            lsl.Calc(lambda x: jnp.sqrt(x) + clip_min_pos, self.variance_param),
            name=name,
        )

        self.variance_param.update()
        self.update()
        val = self.variance_param.value
        log_prob = self.variance_param.log_prob
        self.bijector = None

        if bijector is not None:
            bijectors = tfb.Chain(
                (bijector, tfb.SoftClip(low=clip_min_real, hinge_softness=0.3))
            )
            transformed = self.variance_param.transform(bijectors, inference="drop")
            transformed.inference = inference
            transformed.role = self.variance_param.role
            self.variance_param.role = ""
            transformed.update()
            self.variance_param.update()
            self.update()
            self.bijector = bijector
            self.value_node.function = jnp.sqrt  # type: ignore

            val = transformed.value
            log_prob = transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)


class ScaleInverseGamma(lsl.Var):
    """
    A variable with an Inverse Gamma prior on its square.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``b``.
    name
        Name of the variable.
    inference
        Inference type.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    Attributes
    ----------
    variance_param
        The internal variance parameter (square of the reported scale). This is
        an ``lsl.Var`` created as the latent variance parameter.
    bijector
        The bijector passed to the constructor (or ``None``). When present,
        the bijector was applied to the variance parameter and the public
        variable becomes a deterministic transform of that parameter.
    """

    def __init__(
        self,
        value: Array,
        concentration: float | lsl.Var | lsl.Node,
        scale: float | lsl.Var | lsl.Node,
        name: str = "",
        inference: InferenceTypes = None,
        bijector: tfb.Bijector | None = None,
        role: str = "hyperparam",
    ):
        value = jnp.asarray(value)

        if isinstance(scale, float):
            scale = jnp.asarray(scale)  # type: ignore
            scale = lsl.Data(scale, _name=f"{name}_scale")

        if isinstance(concentration, float):
            concentration = jnp.asarray(concentration)  # type: ignore
            concentration = lsl.Data(concentration, _name=f"{name}_concentration")

        prior = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)
        self.variance_param = lsl.Var.new_param(
            value, prior, inference=inference, name=f"{name}_square"
        )
        self.variance_param.role = role
        super().__init__(lsl.Calc(jnp.sqrt, self.variance_param), name=name)

        self.variance_param.update()
        self.update()
        val = self.variance_param.value
        log_prob = self.variance_param.log_prob
        self.bijector = None

        if bijector is not None:
            transformed = self.variance_param.transform(bijector, inference="drop")
            transformed.inference = inference
            transformed.role = self.variance_param.role

            self.variance_param.role = ""
            transformed.update()
            self.variance_param.update()
            self.update()
            self.bijector = bijector

            val = transformed.value
            log_prob = transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)
