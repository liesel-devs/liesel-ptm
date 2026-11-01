"""Legacy shared-shape PTMs on Liesel and liesel-gam components."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel_gam as gam
from jax.typing import ArrayLike
from liesel_gam.iwls_proposals import gaussian_iwls_spec_loc, gaussian_iwls_spec_scale

from .bspline import PTMKnots
from .model import (
    PTMDist,
    PTMPredictors,
    _check_intercept_free,
    _check_scale,
    _make_predictors,
    _make_response,
    _response_distribution,
    _validate_geometry,
)


class LocationIntercept(lsl.Var):
    """Computed location intercept that makes the training residual mean zero.

    Parameters
    ----------
    y
        Training response variable or data node.
    loc_model, log_scale_model
        Intercept-free location and log-scale predictors.
    name
        Name used to record the intercept in posterior samples.
    """

    def __init__(
        self,
        y: lsl.Var | lsl.Node,
        loc_model: lsl.Var,
        log_scale_model: lsl.Var,
        name: str = "",
    ) -> None:
        value_calc = lsl.Calc(self.compute_intercept, y, loc_model, log_scale_model)
        super().__init__(value_calc, name=name)
        self.update()

    @staticmethod
    def compute_intercept(y, loc_model, log_scale_model):
        nobs = jnp.shape(jnp.atleast_1d(log_scale_model))[0]

        log_mean_exp = jax.nn.logsumexp(-log_scale_model) - jnp.log(nobs)
        inv_scale_model_mean = jnp.exp(-log_mean_exp)

        residual_model_mean = jnp.mean((y - loc_model) * jnp.exp(-log_scale_model))

        return inv_scale_model_mean * residual_model_mean


class LogScaleIntercept(lsl.Var):
    """Computed log-scale intercept that makes training residual variance one.

    Parameters
    ----------
    y
        Training response variable or data node.
    loc
        Location predictor including its computed intercept.
    log_scale_model
        Intercept-free log-scale predictor. Uses the empirical variance with ``ddof=0``.
    name
        Name used to record the intercept in posterior samples.
    """

    def __init__(
        self,
        y: lsl.Var | lsl.Node,
        loc: lsl.Var,
        log_scale_model: lsl.Var,
        name: str = "",
    ) -> None:
        value_calc = lsl.Calc(
            self.compute_intercept,
            y,
            loc,
            log_scale_model,
        )

        super().__init__(
            value_calc,
            name=name,
        )

        self.update()

    @staticmethod
    def compute_intercept(y, loc, log_scale_model):
        scale_model = jnp.exp(log_scale_model)
        exp_scale_intercept = jnp.std((y - loc) / scale_model)
        return jnp.log(exp_scale_intercept)


def _computed_intercepts(
    model: lsl.Model,
) -> tuple[LocationIntercept, LogScaleIntercept]:
    result = []
    for cls in (LocationIntercept, LogScaleIntercept):
        found = [var for var in model.vars.values() if isinstance(var, cls)]
        if len(found) != 1 or not found[0].weak or found[0].parameter:
            raise ValueError(
                f"Expected exactly one computed, nonparameter {cls.__name__}."
            )
        result.append(found[0])
    return result[0], result[1]


def _full_data_response(y, loc, log_scale, *, nobs: int):
    expected = (nobs,)
    if jnp.shape(y) != expected or any(
        jnp.shape(value) not in ((), expected) for value in (loc, log_scale)
    ):
        raise ValueError(
            f"LegacyPTM requires full-data fitting with {nobs} rows; "
            "minibatches and partial data updates are unsupported."
        )
    return y


def _observation_replaced(
    model: lsl.Model, var: lsl.Var, newdata: gs.PositionInput
) -> bool:
    if var.name in newdata:
        return True
    if var.strong:
        return False
    inputs = model.node_parental_subgraph(var.value_node)
    sources = {
        name
        for name, source in model.observed.items()
        if source.strong and source.value_node in inputs
    }
    return bool(sources) and sources <= newdata.keys()


class LegacyPTM(lsl.Model):
    """Shared-shape PTM with computed location and log-scale intercepts.

    Parameters
    ----------
    y
        One-dimensional response values with at least two rows, or an unowned
        observed :class:`liesel.model.Var` without a distribution. An array becomes
        a writable observed input named ``y``.
        The likelihood is attached to :attr:`response`, an identity view of that
        input, so data updates also reach the computed intercepts.
    nshape
        Number of shape coefficients before constraints; an integer of at least three.
    bounds
        Flexible interval on the standardized residual scale. Cubic, equidistant
        :class:`~liesel_ptm.PTMKnots` are constructed internally.
    loc, scale, shape
        Optional native liesel-gam predictors, retained by identity. Location and
        log scale must be intercept-free. Scale must use ``inv_link=jnp.exp``;
        its ``linear_predictor`` supplies the log-scale effects. Shape must be a
        shared, ``sumzero_coef``-constrained predictor with ``nshape`` dimensions.
        Defaults use empty location/scale effects and the same shape construction
        as :class:`~liesel_ptm.PTM`. Names ``loc`` and ``scale`` are reserved for
        the complete predictors; name supplied effects separately.
    shape_scale
        Native scale variable, :class:`liesel_gam.VarIGPrior`, or fixed positive real
        for the automatic shape intercept. ``None`` uses the unclipped Weibull variance
        prior described in :class:`~liesel_ptm.PTM`. Configure this on the predictor
        itself when supplying ``shape``.

    Raises
    ------
    ValueError
        If the shape dimension or interval is invalid, location/scale contain
        intercepts, shape is not shared and constrained, or the response variable
        already belongs to a model or has a distribution.

    Notes
    -----
    Add terms through :attr:`pred` before creating inference objects. Computed
    intercepts are separate from these predictors and enter the likelihood as
    ``loc`` and ``scale``. :meth:`setup_mcmc` records both intercepts at every draw;
    :meth:`predict` uses those records on a copied prediction graph.

    Use :meth:`predict` on the model for targets depending on computed intercepts.
    :meth:`liesel.model.Var.predict` builds a plain Liesel submodel and recomputes
    weak intercepts; it does not use the recorded intercept draws.

    Fitting uses the complete observation count supplied at construction. Joint
    shuffling of response and covariates is supported. Minibatches, training/validation
    splits and partial data updates with fewer rows are rejected because the computed
    intercepts depend on the full data. New-data prediction uses recorded intercepts
    and accepts different observation counts.

    See Also
    --------
    LocationIntercept : Compute the location intercept from training residuals.
    LogScaleIntercept : Compute the corresponding log-scale intercept.
    """

    pred: PTMPredictors
    """Location/scale effects and shared shape, excluding computed intercepts."""
    response: lsl.Var
    """Observed likelihood view of the writable response input (``y`` by default)."""

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
        _validate_geometry(nshape, bounds, minimum=3)
        response = _make_response(y)
        response_shape = jnp.shape(response.value)
        if len(response_shape) != 1 or response_shape[0] < 2:
            raise ValueError(
                "LegacyPTM requires a one-dimensional response with at least two rows."
            )
        self.pred = _make_predictors(
            nshape, loc, scale, shape, shape_scale, legacy=True
        )
        for predictor in (self.pred.loc, self.pred.scale, self.pred.shape):
            if predictor.name in ("loc", "scale"):
                raise ValueError(
                    f"Predictor name {predictor.name!r} is reserved for the likelihood."
                )
        for predictor in (self.pred.loc, self.pred.scale):
            _check_intercept_free(predictor)
        if self.pred.shape.terms:
            raise ValueError("Legacy shape must be shared, without covariate terms.")
        # The strong response input feeds both intercepts and a weak likelihood
        # view, keeping Liesel's variable graph acyclic without copying the data.
        training_y = lsl.Calc(
            partial(_full_data_response, nobs=response_shape[0]),
            response,
            self.pred.loc,
            self.pred.scale.linear_predictor,
            _name="legacy_full_data_response",
        )
        beta = LocationIntercept(
            training_y, self.pred.loc, self.pred.scale.linear_predictor, name="beta0"
        )
        loc_full = lsl.Var.new_calc(lambda b, m: b + m, beta, self.pred.loc, name="loc")
        gamma = LogScaleIntercept(
            training_y, loc_full, self.pred.scale.linear_predictor, name="gamma0"
        )
        scale_full = lsl.Var.new_calc(
            lambda g, relative: jnp.exp(g) * relative,
            gamma,
            self.pred.scale,
            name="scale",
        )
        knots = PTMKnots(*bounds, nparam=nshape)
        distribution = PTMDist(
            knots.knots, loc_full, scale_full, self.pred.shape, bspline="ptm"
        )
        self._scale = scale_full
        self.response = lsl.Var.new_obs(
            response, name=f"{response.name or 'y'}_likelihood"
        )
        with _response_distribution(self, self.response, distribution):
            super().__init__(self.response)

    def setup_mcmc(
        self,
        *,
        seed: int,
        num_chains: int = 4,
        iwls_kwargs: Mapping[str, Any] | None = None,
        shape_spec: gs.MCMCSpec | None = None,
    ) -> gs.EngineBuilder:
        """Configure Gaussian-proposal IWLS and shape NUTS/HMC for a legacy PTM.

        Parameters
        ----------
        seed, num_chains
            Random seed and chain count for :class:`liesel.goose.EngineBuilder`.
        iwls_kwargs
            Arguments for :class:`liesel.goose.IWLSKernel`. Defaults to step size 1 and
            target acceptance 0.5. Proposal geometry comes from liesel-gam.
        shape_spec
            NUTS or HMC specification shared by shape coefficients and nonconjugate
            shape scales. Defaults to NUTS with target acceptance 0.9.

        Returns
        -------
        liesel.goose.EngineBuilder
            Builder with complete kernel coverage and both computed intercepts included
            in recorded positions. Add adaptation/posterior epochs before building.

        Raises
        ------
        ValueError
            If terms are unsupported, the shared shape is not sum-to-zero constrained,
            or parameter blocks do not cover the model without overlap.

        Notes
        -----
        Updates parameter inference specifications. Existing scale inference, including
        conjugate Gibbs, is retained; missing scalar scale updates use NUTS. The fitted
        model remains the posterior target, including its computed-intercept updates.
        Predictors are taken from :attr:`pred`. Location/scale require isotropic
        structured terms, such as ``TermBuilder.slin``; plain linear, anisotropic and
        scale-factored terms require manual Liesel configuration.

        See Also
        --------
        predict : Predict using the recorded intercept draws.
        """
        model = self
        loc_terms, log_scale_terms, shape = (
            self.pred.loc,
            self.pred.scale,
            self.pred.shape,
        )
        scale_name = self._scale.name
        intercepts = _computed_intercepts(model)
        for var in (loc_terms, log_scale_terms, shape):
            if var.model is not model:
                raise ValueError(
                    f"Predictor {var.name!r} in .pred must belong to this model."
                )
        _check_scale(log_scale_terms, legacy=True)
        if shape.terms or not isinstance(shape.intercept, gam.MultivariateIntercept):
            raise ValueError(
                "Shape must have a shared intercept and no covariate terms."
            )
        reparam = shape.intercept.dimension_reparam.value
        if reparam.shape[1] != reparam.shape[0] - 1 or not jnp.allclose(
            reparam.sum(axis=0), 0.0, atol=1e-6
        ):
            raise ValueError("Shared shape must use the sumzero_coef constraint.")
        blocks = [
            set(model.parental_submodel(var).parameters)
            for var in (loc_terms, log_scale_terms, shape)
        ]
        if any(blocks[i] & blocks[j] for i in range(3) for j in range(i)):
            raise ValueError("Location, log-scale and shape parameter blocks overlap.")
        unexplained = set(model.parameters) - set.union(*blocks)
        if unexplained:
            raise ValueError(f"Parameters outside .pred: {sorted(unexplained)}")

        kwargs: dict[str, Any] = {"initial_step_size": 1.0, "da_target_accept": 0.5}
        kwargs.update(iwls_kwargs or {})
        if "chol_info_fn" in kwargs:
            raise ValueError("Gaussian proposal geometry cannot be overridden here.")
        specs: dict[str, gs.MCMCSpec] = {}
        for predictor, factory in (
            (loc_terms, gaussian_iwls_spec_loc),
            (log_scale_terms, gaussian_iwls_spec_scale),
        ):
            _check_intercept_free(predictor)
            for term in predictor.terms.values():
                if not isinstance(term, gam.StrctTerm):
                    raise ValueError(
                        f"{term.name!r} needs a structured term for Gaussian IWLS; "
                        "use TermBuilder.slin or configure Liesel manually."
                    )
                if term.coef.name in specs:
                    raise ValueError(
                        f"Coefficient {term.coef.name!r} occurs in multiple terms."
                    )
                options = kwargs | (
                    {"scale_name": scale_name} if predictor is loc_terms else {}
                )
                specs[term.coef.name] = factory(term, **options)

        if shape_spec is None:
            shape_spec = gs.MCMCSpec(
                gs.NUTSKernel, kernel_kwargs={"da_target_accept": 0.9}
            )
        if shape_spec.kernel not in (gs.NUTSKernel, gs.HMCKernel):
            raise ValueError("shape_spec must use NUTSKernel or HMCKernel.")
        shape_spec = replace(
            shape_spec, kernel_group=shape_spec.kernel_group or "legacy_ptm_shape"
        )
        for name in set.union(*blocks) - specs.keys():
            var = model.parameters[name]
            existing = var.get_inference(None)
            if name in blocks[2] and not (
                isinstance(existing, gs.MCMCSpec)
                and existing.kernel is gam.init_star_ig_gibbs
            ):
                specs[name] = shape_spec
            elif existing is not None:
                if not isinstance(existing, gs.MCMCSpec):
                    raise ValueError(f"{name!r} must have a Liesel MCMCSpec.")
                specs[name] = existing
            elif jnp.size(var.value) == 1:
                specs[name] = gs.MCMCSpec(gs.NUTSKernel)
            else:
                raise ValueError(f"Missing inference for scale parameter {name!r}.")

        for name, spec in specs.items():
            model.parameters[name].inference = spec
        builder = gs.LieselMCMC(model).get_engine_builder(seed, num_chains)
        coverage = Counter(
            name for kernel in builder.kernels for name in kernel.position_keys
        )
        if coverage != Counter(model.parameters.keys()):
            raise ValueError(
                "Every model parameter must occur in exactly one MCMC kernel."
            )
        builder.positions_included = [var.name for var in intercepts]
        return builder

    def predict(
        self,
        samples: gs.PositionInput,
        predict: Sequence[str] | None = None,
        newdata: gs.PositionInput | None = None,
        chunk_size: int | None = 64,
    ) -> dict[str, Any]:
        """Predict from a legacy PTM using its recorded intercept draws.

        Parameters
        ----------
        samples
            Posterior samples, including both intercepts recorded by
            :meth:`setup_mcmc`, with the same leading sample axes.
        predict
            Variable or node names to predict. ``None`` returns all variables at
            fitted data. With ``newdata``, observed variables are omitted unless
            they or all their observed value inputs have replacement values.
        newdata, chunk_size
            Passed to :meth:`liesel.model.Model.predict` on a detached prediction graph.

        Returns
        -------
        dict
            Predictions with the sample axes retained, as in ``Model.predict``.

        Raises
        ------
        ValueError
            If recorded intercepts are missing, their sample axes differ, or ``predict``
            is empty.
        KeyError
            If a prediction target or new-data key is not part of the model.

        Notes
        -----
        The fitted model and samples are unchanged. Only copied intercept calculations
        become strong value nodes, so Liesel uses the recorded draws. For predictor-only
        targets, Liesel evaluates a submodel without training responses. Include new
        responses in ``newdata`` when predicting response-dependent quantities.
        Always call this model method for intercept-dependent predictions;
        :meth:`liesel.model.Var.predict` bypasses recorded-intercept handling.
        """
        model = self
        if predict is None:
            predict = [
                name
                for name, var in self.vars.items()
                if not var.observed
                or newdata is None
                or _observation_replaced(self, var, newdata)
            ]
        intercepts = _computed_intercepts(model)
        missing = {var.name for var in intercepts} - samples.keys()
        if missing:
            raise ValueError(f"Missing recorded intercept draws: {sorted(missing)}")
        if jnp.shape(samples[intercepts[0].name]) != jnp.shape(
            samples[intercepts[1].name]
        ):
            raise ValueError("Recorded intercept draws must have matching sample axes.")
        if isinstance(predict, str) or not predict:
            raise ValueError("predict must be a nonempty sequence of names.")
        variables = model.copy_vars()
        for var in intercepts:
            copied = variables[var.name]
            value = lsl.Value(jnp.asarray(samples[var.name]).reshape(-1)[0])
            value.name = copied.value_node.name
            copied.value_node = value
        prediction_model = lsl.Model(list(variables.values()))
        return prediction_model.predict(samples, predict, newdata, chunk_size)
