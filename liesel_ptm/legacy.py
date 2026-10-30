"""Legacy shared-shape PTMs on Liesel and liesel-gam components."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel_gam as gam
from liesel_gam.iwls_proposals import gaussian_iwls_spec_loc, gaussian_iwls_spec_scale


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


class LegacyPTM:
    """Sampling and prediction for a legacy shared-shape PTM.

    Parameters
    ----------
    model
        Built Liesel model with one computed location/log-scale intercept pair.
        The model graph and parameters stay in Liesel; this class holds a reference.

    See Also
    --------
    LocationIntercept : Compute the location intercept from training residuals.
    LogScaleIntercept : Compute the corresponding log-scale intercept.
    """

    model: lsl.Model
    """Liesel model used for sampling and copied for prediction."""

    def __init__(self, model: lsl.Model) -> None:
        self.model = model

    def setup_mcmc(
        self,
        *,
        loc_terms: gam.AdditivePredictor,
        log_scale_terms: gam.AdditivePredictor,
        shape: gam.MVAdditivePredictor,
        seed: int,
        num_chains: int = 4,
        scale_name: str = "scale",
        iwls_kwargs: Mapping[str, Any] | None = None,
        shape_spec: gs.MCMCSpec | None = None,
    ) -> gs.EngineBuilder:
        """Configure Gaussian-proposal IWLS and shape NUTS/HMC for a legacy PTM.

        Parameters
        ----------
        loc_terms, log_scale_terms
            Intercept-free predictors containing isotropic structured liesel-gam terms.
            Use ``TermBuilder.slin`` for linear terms. Plain linear, anisotropic and
            scale-factored terms require manual Liesel setup instead.
        shape
            Shared multivariate shape predictor with an intercept and no covariate
            terms.
        seed, num_chains
            Random seed and chain count for :class:`liesel.goose.EngineBuilder`.
        scale_name
            Name of the observation-scale variable used for location proposal weights.
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

        See Also
        --------
        predict : Predict using the recorded intercept draws.
        """
        model = self.model
        intercepts = _computed_intercepts(model)
        for var in (loc_terms, log_scale_terms, shape):
            if var.model is not model:
                raise ValueError(f"{var.name!r} must belong to the supplied model.")
        if scale_name not in model.vars:
            raise ValueError(f"Observation scale {scale_name!r} is not in the model.")
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
            raise ValueError(
                f"Parameters outside the supplied predictors: {sorted(unexplained)}"
            )

        kwargs: dict[str, Any] = {"initial_step_size": 1.0, "da_target_accept": 0.5}
        kwargs.update(iwls_kwargs or {})
        if "chol_info_fn" in kwargs:
            raise ValueError("Gaussian proposal geometry cannot be overridden here.")
        specs: dict[str, gs.MCMCSpec] = {}
        for predictor, factory in (
            (loc_terms, gaussian_iwls_spec_loc),
            (log_scale_terms, gaussian_iwls_spec_scale),
        ):
            if isinstance(predictor.intercept, lsl.Var) or jnp.any(
                predictor.intercept.value
            ):
                raise ValueError(f"{predictor.name!r} must be intercept-free.")
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
        *,
        predict: Sequence[str],
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
            Nonempty sequence of variable or node names to predict.
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

        Notes
        -----
        The fitted model and samples are unchanged. Only copied intercept calculations
        become strong value nodes, so Liesel uses the recorded draws. For predictor-only
        targets, training responses are absent from the prediction graph. Include new
        responses in ``newdata`` when predicting response-dependent quantities.
        """
        model = self.model
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
        nodes, variables = model.copy_nodes_and_vars()
        for var in intercepts:
            copied = variables[var.name]
            value = lsl.Value(jnp.asarray(samples[var.name]).reshape(-1)[0])
            value.name = copied.value_node.name
            copied.value_node = value
            nodes[value.name] = value
        objects = nodes | variables
        prediction_model = lsl.Model([objects[name] for name in predict])
        return prediction_model.predict(samples, predict, newdata, chunk_size)
