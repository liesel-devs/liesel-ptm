from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, Self

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.typing import ArrayLike
from liesel.contrib.splines import basis_matrix, equidistant_knots

from ..constraint import LinearConstraintEVD, mixed_model
from ..penalty import Penalty
from .dist import MultivariateNormalSingular
from .kernel import init_star_ig_gibbs
from .roles import Roles

InferenceTypes = Any
Array = Any

logger = logging.getLogger(__name__)


class UserVar(lsl.Var):
    @classmethod
    def new_calc(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_obs(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_param(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_value(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )


class term(UserVar):
    """
    General structured additive term.

    A structured additive term represents a smooth or structured effect in a
    generalized additive model. The term wraps a design/basis matrix together
    with a prior/penalty and a set of coefficients. The object exposes the
    coefficient variable and evaluates the term as the matrix-vector product
    of the basis and the coefficients.
    The term evaluates to ``basis @ coef``.

    Parameters
    ----------
    basis
        A :class:`.Basis` instance that produces the design matrix for the \
        term. The basis must evaluate to a 2-D array with shape ``(n_obs, n_bases)``.
    penalty
        Penalty matrix or a variable/value wrapping the penalty \
        used to construct the multivariate normal prior for the coefficients.
    scale
        Scale parameter for the prior on the coefficients. This \
        is typically either a scalar or a per-coefficient scale variable.
    name
        Human-readable name for the term. Used for labelling variables and \
        building sensible default names for internal nodes.
    inference
        :class:`liesel.goose.MCMCSpec` inference specification forwarded to coefficient\
        creation.
    coef_name
        Name for the coefficient variable. If ``None``, a default name based \
        on ``name`` will be used.
    _update_on_init
        If ``True`` (default) the internal calculation/graph nodes are \
        evaluated during initialization. Set to ``False`` to delay \
        initial evaluation.

    Raises
    ------
    ValueError
        If ``basis.value`` does not have two dimensions.

    Attributes
    ----------
    scale
        The scale variable used by the prior on the coefficients.
    nbases
        Number of basis functions (number of columns in the basis matrix).
    basis
        The basis object provided to the constructor.
    coef
        The coefficient variable created for this term. It holds the prior
        (multivariate normal singular) and is used in the evaluation of the
        term.
    is_noncentered
        Whether the term has been reparameterized to the non-centered form.

    Examples
    --------
    Create a P-spline basis and wrap it into a term::

        # assume x to be a 1d array.
        b = ps(x, nbases=10)
        t = term(basis=b, penalty=b.penalty, name="f(x)")

    """

    def __init__(
        self,
        basis: Basis,
        penalty: lsl.Var | lsl.Value | Array,
        scale: lsl.Var | Array,
        name: str,
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        _update_on_init: bool = True,
    ):
        coef_name = f"{name}_coef" if coef_name is None else coef_name

        if not jnp.asarray(basis.value).ndim == 2:
            raise ValueError(f"basis must have 2 dimensions, got {basis.value.ndim}.")

        nbases = jnp.shape(basis.value)[-1]

        if isinstance(penalty, lsl.Var | lsl.Value):
            penalty_arr = penalty.value
        else:
            penalty_arr = penalty

        prior = lsl.Dist(
            MultivariateNormalSingular,
            loc=0.0,
            scale=scale,
            penalty=penalty,
            penalty_rank=jnp.linalg.matrix_rank(penalty_arr),
        )

        self.scale = prior["scale"]
        self.nbases = nbases
        self.basis = basis
        self.coef = lsl.Var.new_param(
            jnp.zeros(nbases), prior, inference=inference, name=coef_name
        )
        calc = lsl.Calc(
            lambda basis, coef: jnp.dot(basis, coef),
            basis=basis,
            coef=self.coef,
            _update_on_init=_update_on_init,
        )

        super().__init__(calc, name=name)
        if _update_on_init:
            self.coef.update()
        self.coef.role = Roles.coef_smooth
        self.role = Roles.term_smooth

        self.is_noncentered = False

    @property
    def includes_intercept(self) -> bool | None:
        return self.basis.includes_intercept

    def reparam_noncentered(self) -> Self:
        """
        Turns this term into noncentered form, which means the prior for
        the coefficient will be turned from ``coef ~ N(0, scale^2 * inv(penalty))`` into
        ``latent_coef ~ N(0, inv(penalty)); coef = scale * latent_coef``.
        This can sometimes be helpful when sampling with the No-U-Turn Sampler.
        """
        if self.is_noncentered:
            return self

        assert self.coef.dist_node is not None

        self.coef.dist_node["scale"] = lsl.Value(1.0)

        def scaled_dot(basis, coef, scale):
            return jnp.dot(basis, scale * coef)

        self.value_node = lsl.Calc(scaled_dot, self.basis, self.coef, self.scale)
        self.coef.update()
        self.update()
        self.is_noncentered = True
        return self

    @classmethod
    def f(
        cls,
        basis: Basis,
        fname: str = "f",
        scale: lsl.Var | Array | float = 1000.0,
        inference: InferenceTypes = None,
        noncentered: bool = False,
    ) -> term:
        """
        Construct a smooth term from a :class:`Basis`.

        This convenience constructor builds a named ``term`` using the
        provided basis. The penalty matrix is taken from ``basis.penalty`` and
        a coefficient variable with an appropriate multivariate-normal prior
        is created. The returned term evaluates to ``basis @ coef``.

        Parameters
        ----------
        basis
            Basis object that provides the design matrix and penalty for the \
            smooth term. The basis must have an associated input variable with \
            a meaningful name (used to compose the term name).
        fname
            Function-name prefix used when constructing the term name. Default \
            is ``'f'`` which results in names like ``f(x)`` when the basis \
            input is named ``x``.
        scale
            Scale parameter passed to the coefficient prior. \
            Defaults to ``1000.0`` for a weakly-informative prior.
        inference
            Inference specification forwarded to the coefficient variable \
            creation, a :class:`liesel.goose.MCMCSpec`.
        noncentered
            If ``True``, the term is reparameterized to the non-centered \
            form via :meth:`.reparam_noncentered` before being returned.

        Returns
        -------
        A :class:`.term` instance configured with the given basis and prior settings.

        Notes
        -----
        The default coefficient name is a LaTeX-like string ``"$\\beta_{f(x)}$"``
        to improve readability in printed summaries.

        See Also
        --------
        .ScaleWeibull : A scale parameter derived from a variance parameter with Weibull
            prior.

        Examples
        --------

        Create a P-spline basis and wrap it into a term, using a Weibull prior
        for the variance parameter::

            import liesel.goose as gs
            import tensorflow_probability.substrates.jax.bijectors as tfb

            # assume x to be a 1d array.
            b = ps(x, nbases=10)
            scale = ScaleWeibull(
                1.0,
                scale=0.5,
                inference=gs.MCMCSpec(gs.NUTSKernel),
                bijector=tfb.Exp(),
                name="fx_scale"
            )
            t = term.f(basis=b, scale=scale)
        """
        name = f"{fname}({basis.x.name})"

        coef_name = "$\\beta_{" + f"{name}" + "}$"

        term = cls(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
            name=name,
        )

        if noncentered:
            term.reparam_noncentered()

        return term

    @classmethod
    def f_ig(
        cls,
        basis: Basis,
        fname: str = "f",
        ig_concentration: float = 1.0,
        ig_scale: float = 0.005,
        inference: InferenceTypes = None,
        variance_value: float = 100.0,
        noncentered: bool = False,
    ) -> term:
        """
        Construct a smooth term with an inverse-gamma prior on the variance.

        This convenience constructor creates a term similar to :meth:`.f` but
        sets up an explicit variance parameter with an Inverse-Gamma prior.
        A scale variable is set up by taking the square-root, and the
        coefficient prior uses the derived ``scale`` together with the basis
        penalty. By default a Gibbs-style initialization is attached to the
        variance inference via an internal kernel; an optional jitter
        distribution can be provided for MCMC initialization.

        Parameters
        ----------
        basis
            Basis object providing the design matrix and penalty.
        fname
            Prefix used to build the term name (default: ``'f'``).
        ig_concentration
            Concentration (shape) parameter of the Inverse-Gamma prior for the \
            variance.
        ig_scale
            Scale parameter of the Inverse-Gamma prior for the variance.
        inference
            Inference specification forwarded to the coefficient variable \
            creation, a :class:`liesel.goose.MCMCSpec`.
        variance_value
            Initial value for the variance parameter.
        noncentered
            If ``True``, reparameterize the term to non-centered form \
            (see :meth:`.reparam_noncentered`).

        Returns
        -------
        A :class:`.term` instance configured with an inverse-gamma prior on
        the variance and an appropriate inference specification for
        variance updates.

        Notes
        -----
        - The variance parameter is named using a LaTeX-like representation
          ``"$\\tau^2_{...}$"`` for readability in summaries.

        Examples
        --------
        Create a P-spline basis and wrap it into a term::

            # assume x to be a 1d array.
            b = ps(x, nbases=10)
            t = term.f_ig(b)
        """
        name = f"{fname}({basis.x.name})"
        coef_name = "$\\beta_{" + f"{name}" + "}$"

        variance = lsl.Var.new_param(
            value=variance_value,
            distribution=lsl.Dist(
                tfd.InverseGamma,
                concentration=ig_concentration,
                scale=ig_scale,
            ),
            name="$\\tau^2_{" + f"{name}" + "}$",
        )
        variance.role = Roles.variance_smooth

        scale = lsl.Var.new_calc(jnp.sqrt, variance, name="$\\tau_{" + f"{name}" + "}$")
        scale.role = Roles.scale_smooth

        term = cls(
            basis=basis,
            scale=scale,
            penalty=basis.penalty,
            inference=inference,
            name=name,
            coef_name=coef_name,
        )

        variance.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={"coef": term.coef, "scale": scale},
        )

        if noncentered:
            term.reparam_noncentered()

        return term


SmoothTerm = term
Term = term


class LinearTerm(term):
    """Kept for backwards-compatibility of the interface."""

    def __init__(
        self,
        x: lsl.Var | Array,
        name: str,
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
        add_intercept: bool = False,
        coef_name: str | None = None,
        basis_name: str | None = None,
    ):
        if not isinstance(x, lsl.Var):
            x = lsl.Var.new_obs(x, name=f"{name}_input")

        if not x.name:
            # to ensure sensible basis name
            raise ValueError(f"{x=} must be named.")

        coef_name = coef_name or f"{name}_coef"
        basis_name = basis_name or f"B({name})"
        basis = Basis.new_linear(value=x, name=basis_name, add_intercept=add_intercept)

        nbases = jnp.shape(basis.value)[-1]
        penalty = jnp.eye(nbases)
        # just a temporary variable to satisfy the api of term
        scale = lsl.Var(1.0, name=f"_{name}_scale_tmp")

        super().__init__(
            basis=basis,
            penalty=penalty,
            scale=scale,
            name=name,
            inference=inference,
            coef_name=coef_name,
        )
        self.coef.dist_node = distribution
        self.coef.role = Roles.coef_linear
        self.role = Roles.term_linear


class LinearTerm2(term):
    """New version of LinearTerm, with interface consistent with the term base class."""

    def __init__(
        self,
        value: lsl.Var | lsl.Node | Array,
        name: str,
        scale: lsl.Var | Array = 1000.0,
        inference: InferenceTypes = None,
        add_intercept: bool = False,
        coef_name: str | None = None,
        basis_name: str | None = None,
    ):
        if not isinstance(value, lsl.Var | lsl.Node):
            x: lsl.Var | lsl.Node = lsl.Var.new_obs(value, name=f"{name}_input")
        else:
            x = value

        if not x.name:
            # to ensure sensible basis name
            raise ValueError(f"{value=} must be named.")

        coef_name = coef_name or f"{name}_coef"
        basis_name = basis_name or f"B({name})"
        basis = Basis.new_linear(value=x, name=basis_name, add_intercept=add_intercept)

        nbases = jnp.shape(basis.value)[-1]
        penalty = jnp.eye(nbases)
        super().__init__(
            basis=basis,
            penalty=penalty,
            coef_name=coef_name,
            name=name,
            inference=inference,
            scale=scale,
        )

        self.coef.role = Roles.coef_linear
        self.role = Roles.term_linear


class Intercept(lsl.Var):
    def __init__(
        self,
        name: str,
        value: Array | float = 0.0,
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
    ) -> None:
        super().__init__(
            value=value, distribution=distribution, name=name, inference=inference
        )
        self.parameter = True
        self.role = Roles.intercept


def make_callback(function, input_shape, dtype, *args, **kwargs):
    if len(input_shape):
        k = input_shape[-1]

    def fn(x):
        n = jnp.shape(jnp.atleast_1d(x))[0]
        if len(input_shape) == 2:
            shape = (n, k)
        elif len(input_shape) == 1:
            shape = (n,)
        elif not len(input_shape):
            shape = ()
        else:
            raise RuntimeError(
                "Return shape of 'basis_fn(value)' must"
                f" have <= dimensions, got {input_shape}"
            )
        result_shape = jax.ShapeDtypeStruct(shape, dtype)
        result = jax.pure_callback(
            function, result_shape, x, *args, vmap_method="sequential", **kwargs
        )
        return result

    return fn


class Basis(UserVar):
    """
    General basis for a structured additive term.

    The ``Basis`` class wraps either a provided observation variable or a raw
    array and a basis-generation function. It constructs an internal
    calculation node that produces the basis (design) matrix used by
    smooth terms. The basis function may be executed via a
    callback that does not need to be jax-compatible (the default, potentially slow)
    with a jax-compatible function that is included in just-in-time-compilation
    (when ``use_callback=False``).

    Parameters
    ----------
    value
        If a :class:`liesel.model.Var` or node is provided it is used as \
        the input variable for the basis. Otherwise a raw array-like \
        object may be supplied together with ``xname`` to create an \
        observed variable internally.
    basis_fn
        Function mapping the input variable's values to a basis matrix or \
        vector. It must accept the input array and any ``basis_kwargs`` \
        and return an array of shape ``(n_obs, n_bases)`` (or a scalar/1-d \
        array for simpler bases). By default this is the identity \
        function (``lambda x: x``).
    name
        Optional name for the basis object. If omitted, a sensible name \
        is constructed from the input variable's name (``B(<xname>)``).
    xname
        Required when ``value`` is a raw array: provides a name for the \
        observation variable that will be created.
    use_callback
        If ``True`` (default) the basis_fn is wrapped in a JAX \
        ``pure_callback`` via :func:`make_callback` to allow arbitrary \
        Python basis functions while preserving JAX tracing. If ``False`` \
        the function is used directly and must be jittable via JAX.
    cache_basis
        If ``True`` the computed basis is cached in a persistent \
        calculation node (``lsl.Calc``), which avoids re-computation \
        when not required, but uses memory. If ``False`` a transient \
        calculation node (``lsl.TransientCalc``) is used and the basis \
        will be recomputed with each evaluation of ``Basis.value``, \
        but not stored in memory.
    includes_intercept
        Explicit flag stating whether the basis includes an intercept \
        column. If ``None`` the property is left unspecified and may be \
        inferred (if possible) by downstream code.
    penalty
        Penalty matrix associated with the basis. If omitted, \
        a default identity penalty is created based on the number \
        of basis functions.
    **basis_kwargs
        Additional keyword arguments forwarded to ``basis_fn``.

    Raises
    ------
    ValueError
        If ``value`` is an array and ``xname`` is not provided, or if
        the created input variable has no name.

    Notes
    -----
    The basis is evaluated once during initialization (via
    ``self.update()``) to determine its shape and dtype. The internal
    callback wrapper inspects the return shape to build a compatible
    JAX ShapeDtypeStruct for the pure callback.

    Attributes
    ----------
    role
        The role assigned to this variable.
    observed
        Whether the basis is derived from an observed variable (always \
        ``True`` for bases created from input data).
    includes_intercept
        Flag indicating whether the basis contains an intercept column. If \
        ``None`` the information is unspecified.
    x
        The input variable (observations) used to construct the basis.
    nbases
        Number of basis functions (number of columns in the basis matrix).
    penalty
        Penalty matrix (wrapped as a :class:`liesel.model.Value`) associated \
        with the basis.
    is_constrained
        Whether a linear constraint has been applied to the basis.
    penalty_is_scaled
        Whether the penalty has been scaled (via :meth:`.reparam_scale_penalty`).
    penalty_is_diagonalized
        Whether the penalty has been diagonalized (via \
        :meth:`.reparam_diagonalize_penalty`).
    constraint_type
        If constrained, the type of constraint applied (for example \
        ``'sumzero_term'``).
    constraint_matrix
        The constraint matrix used when a ``'custom'`` constraint has been \
        applied.

    Examples
    --------
    Identity basis from a named variable::

        import liesel.model as lsl
        import jax.numpy as jnp
        xvar = lsl.Var.new_obs(jnp.array([1.,2.,3.]), name='x')
        b = Basis(value=xvar)
    """

    def __init__(
        self,
        value: lsl.Var | lsl.Node | Array,
        basis_fn: Callable[[Array], Array] | Callable[..., Array] = lambda x: x,
        name: str | None = None,
        xname: str | None = None,
        use_callback: bool = True,
        cache_basis: bool = True,
        includes_intercept: bool | None = None,
        penalty: ArrayLike | lsl.Value | None = None,
        **basis_kwargs,
    ) -> None:
        if isinstance(value, lsl.Var | lsl.Node):
            value_var = value
        else:
            if not xname:
                raise ValueError(
                    "When supplying an array to `value`, `xname` must be defined."
                )
            value_var = lsl.Var.new_obs(value, name=xname)

        if not value_var.name:
            # to ensure sensible basis name
            raise ValueError(f"{value=} must be named.")

        if use_callback:
            value_ar = jnp.asarray(value_var.value)
            basis_ar = basis_fn(value_ar, **basis_kwargs)
            dtype = basis_ar.dtype
            input_shape = jnp.shape(basis_ar)
            fn = make_callback(basis_fn, input_shape, dtype, **basis_kwargs)
        else:
            fn = basis_fn

        name_ = name or f"B({value_var.name})"

        if cache_basis:
            calc = lsl.Calc(fn, value_var, _name=name_ + "_calc")
        else:
            calc = lsl.TransientCalc(fn, value_var, _name=name_ + "_calc")

        super().__init__(calc, name=name_)
        self.update()
        self.role = Roles.basis
        self.observed = True
        self.includes_intercept = includes_intercept
        self.x = value_var
        basis_shape = jnp.shape(self.value)
        if len(basis_shape) >= 1:
            self.nbases = basis_shape[-1]
        else:
            self.nbases = 1  # scalar case

        if isinstance(penalty, lsl.Value):
            penalty_var = penalty
        elif penalty is None:
            penalty_arr = jnp.eye(self.nbases)
            penalty_var = lsl.Value(penalty_arr)
        else:
            penalty_arr = jnp.asarray(penalty)
            penalty_var = lsl.Value(penalty_arr)

        self._penalty = penalty_var

        self.is_constrained = False
        self.penalty_is_scaled = False
        self.penalty_is_diagonalized = False
        self.constraint_type: str | None = None
        self.constraint_matrix: ArrayLike | None = None

    @property
    def penalty(self) -> lsl.Value:
        """
        Return the penalty matrix wrapped as a :class:`liesel.model.Value`.

        Returns
        -------
        lsl.Value
            Value wrapper holding the penalty (precision) matrix for this
            basis.
        """
        return self._penalty

    def update_penalty(self, value: ArrayLike | lsl.Value):
        """
        Update the penalty matrix for this basis.

        Parameters
        ----------
        value
            New penalty matrix or an already-wrapped :class:`liesel.model.Value`.
        """
        if isinstance(value, lsl.Value):
            self._penalty.value = value.value
        else:
            penalty_arr = jnp.asarray(value)
            self._penalty = lsl.Value(penalty_arr)

    @classmethod
    def new_linear(
        cls,
        value: lsl.Var | lsl.Node | Array,
        name: str | None = None,
        xname: str | None = None,
        add_intercept: bool = False,
    ):
        """
        Create a linear basis (design matrix) from input values.

        Parameters
        ----------
        value
            Input variable or raw array used to construct the design matrix.
        name
            Optional name for the basis.
        xname
            Name for the observation variable when ``value`` is \
            a raw array.
        add_intercept
            If ``True``, adds an intercept column of ones as the first \
            column of the design matrix.

        Returns
        -------
        A :class:`.Basis` instance that produces a (n_obs, n_features)
        design matrix.
        """

        def as_matrix(x):
            x = jnp.atleast_1d(x)
            if len(jnp.shape(x)) == 1:
                x = jnp.expand_dims(x, -1)
            if add_intercept:
                ones = jnp.ones(x.shape[0])
                x = jnp.c_[ones, x]
            return x

        basis = cls(
            value=value,
            basis_fn=as_matrix,
            name=name,
            xname=xname,
            use_callback=False,
            cache_basis=False,
            includes_intercept=add_intercept,
        )

        return basis

    def reparam_diagonalize_penalty(self) -> Self:
        """
        Diagonalize the penalty via an eigenvalue decomposition.

        This method computes a transformation that diagonalizes
        the penalty matrix and updates the internal basis function such that
        subsequent evaluations use the accordingly transformed basis. The penalty is
        updated to the diagonalized version and the attribute
        ``penalty_is_diagonalized`` is set to True.

        Returns
        -------
        The modified basis instance (self).
        """

        if self.penalty_is_diagonalized:
            return self

        assert isinstance(self.value_node, lsl.Calc)
        basis_fn = self.value_node.function

        K = self.penalty.value
        Z = mixed_model(K)

        def reparam_basis(*args, **kwargs):
            return basis_fn(*args, **kwargs) @ Z

        self.value_node.function = reparam_basis
        self.update()
        penalty = Z.T @ K @ Z
        self.update_penalty(penalty)
        self.penalty_is_diagonalized = True

        return self

    def reparam_scale_penalty(self) -> Self:
        """
        Scale the penalty matrix by its infinite norm.

        The penalty matrix is divided by its infinity norm (max absolute row
        sum) so that its values are numerically well-conditioned for
        downstream use. The updated penalty replaces the previous one and
        ``penalty_is_scaled`` is set to True.

        Returns
        -------
        The modified basis instance (self).
        """

        if self.penalty_is_scaled:
            return self

        K = self.penalty.value

        scale = jnp.linalg.norm(K, ord=jnp.inf)
        penalty = K / scale
        self.update_penalty(penalty)
        return self

    def _apply_constraint(self, Z: Array) -> Self:
        """
        Apply a linear reparameterisation to the basis using matrix Z.

        This internal helper multiplies the basis functions by ``Z`` (i.e.
        right-multiplies the design matrix) and updates the penalty to
        reflect the change of basis: ``K_new = Z.T @ K @ Z``.

        Parameters
        ----------
        Z
            Transformation matrix applied to the basis functions.

        Returns
        -------
        The modified basis instance (self).
        """

        assert isinstance(self.value_node, lsl.Calc)
        basis_fn = self.value_node.function

        K = self.penalty.value

        def reparam_basis(*args, **kwargs):
            return basis_fn(*args, **kwargs) @ Z

        self.value_node.function = reparam_basis
        self.update()
        penalty = Z.T @ K @ Z
        self.update_penalty(penalty)
        return self

    def reparam_constraint(
        self,
        type: Literal["sumzero_coef", "sumzero_term", "constant_and_linear", "custom"],
        constraint_matrix: Array | None = None,
    ) -> Self:
        """
        Apply a linear constraint to the basis and corresponding penalty.

        Parameters
        ----------
        type
            Type of constraint to apply. If ``'custom'``, the \
            ``constraint_matrix`` argument must be provided and will be used \
            directly. Other values cause the method to compute a suitable \
            constraint matrix via internal helpers.
        constraint_matrix
            Custom constraint matrix to apply when ``type=='custom'``.

        Returns
        -------
        The modified basis instance (self).

        Raises
        ------
        ValueError
            If ``type=='custom'`` and ``constraint_matrix`` is not provided.
        """

        type_ = type

        if self.is_constrained:
            if type_ != self.constraint_type:
                logger.warning(
                    f"{self} is already constrained ('{self.constraint_type}' "
                    "constraint), so your "
                    f"additional constraint '{type_}' will not be applied."
                )
            return self

        if type_ == "custom" and constraint_matrix is None:
            raise ValueError(f"If {type_=}, `constraint_matrix` must be passed.")

        match type_:
            case "sumzero_coef":
                Z = LinearConstraintEVD.sumzero_coef(self.nbases)
                self._includes_intercept = False
            case "sumzero_term":
                Z = LinearConstraintEVD.sumzero_term(self.value)
                self._includes_intercept = False
            case "constant_and_linear":
                Z = LinearConstraintEVD.constant_and_linear(self.x.value, self.value)
                self._includes_intercept = False
            case "custom":
                Z = constraint_matrix

        self._apply_constraint(Z)
        self.is_constrained = True
        self.constraint_type = type_
        self.constraint_matrix = constraint_matrix

        return self


def ps(
    x: ArrayLike | pd.Series | lsl.Var,
    nbases: int,
    xname: str | None = None,
    knots: ArrayLike | None = None,
    order: int = 3,
    outer_ok: bool = False,
    penalty: ArrayLike | None = None,
    diagonalize_penalty: bool = True,
    scale_penalty: bool = True,
    constraint: Literal["sumzero_term", "sumzero_coef", "constant_and_linear", "custom"]
    | None = "sumzero_term",
    constraint_matrix: ArrayLike | None = None,
) -> Basis:
    """
    Create a P-spline basis.

    Convenience function to build a B-spline/P-spline basis for a numeric
    predictor. It constructs knots (if not provided), builds a basis function
    that evaluates the B-spline basis, attaches a difference-penalty (by
    default a second-order random-walk penalty), and returns a
    :class:`.Basis` instance. Optional post-processing steps like scaling the
    penalty, diagonalization and constraint application are supported and provided
    with sensible defaults.

    Parameters
    ----------
    x
        Predictor values or an observed variable. If an array or Series is \
        provided, ``xname`` should be set to produce a named observation \
        variable.
    nbases
        Number of basis functions (parameters) to use for the P-spline.
    xname
        Name for the created observation variable when ``x`` is not already \
        a :class:`liesel.model.Var`.
    knots
        Knot locations for the B-spline basis. If ``None`` equidistant knots \
        are generated based on the data and ``nbases``.
    order
        Order of the B-spline (degree + 1). Default is 3 (cubic splines).
    outer_ok
        Passed to the underlying basis generator to control behavior for observed \
        values outside the internal knot range.
    penalty
        Penalty matrix to use. If ``None`` a default P-spline penalty is \
        created using a second-order difference.
    diagonalize_penalty
        If ``True`` (default), diagonalize the penalty.
    scale_penalty
        If ``True`` (default), scale the penalty matrix to improve numeric \
        conditioning.
    constraint
        Optional constraint to apply to the basis (see \
        :meth:`.Basis.reparam_constraint`). If ``None`` no constraint is
        applied.
    constraint_matrix
        Custom constraint matrix used when ``constraint=='custom'``.

    Returns
    -------
    A configured :class:`Basis` instance representing the P-spline basis.
    """
    if isinstance(x, lsl.Var):
        xvar = x
    elif isinstance(x, pd.Series):
        xvar = lsl.Var.new_obs(jnp.asarray(x.to_numpy()), name=xname or "")
    else:
        xvar = lsl.Var.new_obs(jnp.asarray(x), name=xname or "")

    if not xvar.name:
        raise ValueError("Either x must be a named lsl.Var, or xname must be set.")

    if knots is None:
        knots = equidistant_knots(xvar.value, n_param=nbases, order=order)

    def basis_fun(x):
        return basis_matrix(x, knots, order, outer_ok)

    if penalty is None:
        penalty = Penalty.pspline(nparam=nbases, random_walk_order=2)
    else:
        penalty = jnp.asarray(penalty)

    knots = knots
    nbases = nbases
    basis = Basis(value=xvar, basis_fn=basis_fun, penalty=penalty)

    if scale_penalty:
        basis.reparam_scale_penalty()

    if diagonalize_penalty:
        basis.reparam_diagonalize_penalty()

    if constraint is not None:
        basis.reparam_constraint(type=constraint, constraint_matrix=constraint_matrix)

    return basis


def lin(
    x: ArrayLike | pd.Series | lsl.Var,
    xname: str | None = None,
    add_intercept: bool = False,
) -> Basis:
    """
    Create a linear basis (design matrix) for a predictor.

    This function wraps :meth:`.Basis.new_linear` and accepts either a raw
    array/Series or an existing :class:`liesel.model.Var`. When a raw array is
    supplied, ``xname`` is used to create a named observation variable.

    Parameters
    ----------
    x
        Predictor values or an observed variable.
    xname
        Name for the created observation variable when ``x`` is not already \
        a :class:`liesel.model.Var`.
    add_intercept
        If ``True``, include an intercept column of ones in the design \
        matrix.

    Returns
    -------
    A :class:`.Basis` instance that produces the design matrix for the
    linear predictor.
    """
    if isinstance(x, lsl.Var):
        xvar = x
    elif isinstance(x, pd.Series):
        xvar = lsl.Var.new_obs(jnp.asarray(x.to_numpy()), name=xname or "")
    else:
        xvar = lsl.Var.new_obs(jnp.asarray(x), name=xname or "")

    basis = Basis.new_linear(xvar, add_intercept=add_intercept, name=f"X({xvar.name})")
    return basis


class term_ri(UserVar):
    """Random-intercept (group-level) term.

    The :class:`term_ri` class implements a random intercept effect where the
    basis encodes group labels (integers) and each group receives a
    coefficient. The coefficients are centered and scaled by an associated
    scale parameter. This term is provided in a form convenient for mixed
    models and hierarchical specifications.

    Parameters
    ----------
    basis
        Basis object that must evaluate to a 1-D array of integer labels \
        (group indices). The length of the unique labels defines the \
        number of group-specific coefficients.
    scale
        Scale parameter (or variable) for the random-intercept \
        coefficients. Typically represents the standard deviation for the \
        group effects.
    name
        Human-readable name for the term. Used to construct internal \
        variable names when ``coef_name`` is not provided.
    inference
        Inference specification forwarded to coefficient creation.
    coef_name
        Name for the coefficient variable. If ``None`` a default name is \
        constructed from ``name``.

    Raises
    ------
    ValueError
        If ``basis.value`` is not one-dimensional (group index array).
    """

    def __init__(
        self,
        basis: Basis,
        scale: lsl.Var | Array,
        name: str,
        inference: InferenceTypes = None,
        coef_name: str | None = None,
    ):
        coef_name = f"{name}_coef" if coef_name is None else coef_name

        if not jnp.asarray(basis.value).ndim == 1:
            raise ValueError(f"basis must have 1 dimensions, got {basis.value.ndim}.")

        nbases = jnp.shape(basis.value)[-1]

        prior = lsl.Dist(
            MultivariateNormalSingular,
            loc=0.0,
            penalty=jnp.eye(nbases),
            scale=1.0,
            penalty_rank=nbases,
        )

        self.scale = scale
        self.nbases = nbases
        self.basis = basis
        self.coef = lsl.Var.new_param(
            jnp.zeros(nbases), prior, inference=inference, name=coef_name
        )

        self.scaled_coef = lsl.Var.new_calc(
            lambda coef, scale: scale * (coef - coef.mean()),
            self.coef,
            self.scale,
            name=f"{self.scale.name}{coef_name}",
        )

        def scaled_take(basis, scaled_coef):
            return jnp.take(scaled_coef, basis)

        calc = lsl.Calc(scaled_take, basis, self.scaled_coef)

        super().__init__(calc, name=name)
        self.coef.update()
        self.coef.role = Roles.coef_smooth
        self.role = Roles.term_smooth

        self.is_noncentered = True
        self.includes_intercept = False

    @classmethod
    def f_ig(
        cls,
        basis: Basis,
        fname: str = "ri",
        ig_concentration: float = 1.0,
        ig_scale: float = 0.005,
        inference: InferenceTypes = gs.MCMCSpec(gs.NUTSKernel),
        variance_value: float = 100.0,
        variance_jitter_dist: tfd.Distribution | None = None,
        coef_name: str | None = None,
    ) -> term_ri:
        """
        Construct a random-intercept term with an inverse-gamma variance prior.

        This convenience constructor creates a :class:`.term_ri` and attaches a
        variance parameter with an Inverse-Gamma prior. The variance is
        transformed into a scale variable and used to scale the group
        coefficients. An MCMC inference specification for the variance is set
        up using a Gibbs kernel by default.

        Parameters
        ----------
        basis
            Basis encoding integer group labels.
        fname
            Prefix used to build the term name (default ``'ri'``).
        ig_concentration
            Shape parameter for the Inverse-Gamma prior on the variance.
        ig_scale
            Scale parameter for the Inverse-Gamma prior on the variance.
        inference
            Inference specification for the coefficient variable.
        variance_value
            Initial value for the variance parameter.
        coef_name
            Optional human-readable name for the coefficient variable.

        Returns
        -------
        A configured random-intercept term with variance and scale
        variables attached.
        """

        name = f"{fname}({basis.x.name})"
        coef_name = "$\\beta_{" + f"{name}" + "}$"

        variance = lsl.Var.new_param(
            value=variance_value,
            distribution=lsl.Dist(
                tfd.InverseGamma,
                concentration=ig_concentration,
                scale=ig_scale,
            ),
            name="$\\tau^2_{" + f"{name}" + "}$",
        )
        variance.role = Roles.variance_smooth

        scale = lsl.Var.new_calc(jnp.sqrt, variance, name="$\\tau_{" + f"{name}" + "}$")
        scale.role = Roles.scale_smooth

        term = cls(
            basis=basis,
            scale=scale,
            inference=inference,
            name=name,
            coef_name=coef_name,
        )

        assert term.coef.dist_node is not None
        variance.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={
                "coef": term.scaled_coef,
                "scale": scale,
                "penalty": term.coef.dist_node["penalty"].value,
            },
            jitter_dist=variance_jitter_dist,
        )

        return term

    @classmethod
    def f(
        cls,
        basis: Basis,
        fname: str = "ri",
        scale: lsl.Var | Array | float = 1000.0,
        inference: InferenceTypes = gs.MCMCSpec(gs.NUTSKernel),
    ) -> term_ri:
        """
        Construct a random-intercept term with a custom scale.

        Parameters
        ----------
        basis
            Basis encoding integer group labels.
        fname
            Prefix used to build the term name (default ``'ri'``).
        scale
            Scale parameter (or variable) for the group coefficients. If a \
            scalar is provided, it is wrapped into a parameter/value as \
            needed.
        inference
            Inference specification forwarded to coefficient creation.

        Returns
        -------
        A configured random-intercept term.
        """

        name = f"{fname}({basis.x.name})"
        coef_name = "$\\beta_{" + f"{name}" + "}$"

        term = cls(
            basis=basis,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
            name=name,
        )

        return term


RandomIntercept = term_ri


def ri(
    x: ArrayLike | pd.Series | lsl.Var,
    xname: str | None = None,
) -> Basis:
    """
    Create a random-intercept basis from grouping labels.

    The returned :class:`Basis` maps arbitrary group labels (strings or
    integers) to contiguous integer indices suitable for indexing group
    coefficients. The produced basis yields, for each observation, the
    integer group index which can be used by :class:`term_ri` to select the
    corresponding group effect.

    Parameters
    ----------
    x
        Grouping labels for each observation. If ``x`` is not an \
        :class:`liesel.model.Var`, a named observation variable will be \
        created using ``xname``.
    xname
        Name for the created observation variable when ``x`` is not already \
        an :class:`liesel.model.Var`.

    Returns
    -------
    A :class:`Basis` whose evaluation yields integer group indices for
    each observation.
    """
    if isinstance(x, lsl.Var):
        xvar = x
    elif isinstance(x, pd.Series):
        xvar = lsl.Var.new_obs(x.to_numpy(), name=xname or "")
    else:
        xvar = lsl.Var.new_obs(x, name=xname or "")

    xval = xvar.value
    unique_labels = np.unique(xval)
    mapping = {val: i for i, val in enumerate(unique_labels)}
    lookup = np.frompyfunc(lambda x: mapping.get(x, -1), 1, 1)

    def basis_fun(x):
        return jnp.asarray(np.asarray(lookup(x), dtype=int))

    basis = Basis(
        xvar,
        basis_fn=basis_fun,
        includes_intercept=False,
        name=f"B({xvar.name})",
    )
    return basis
