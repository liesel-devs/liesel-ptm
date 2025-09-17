from collections import ChainMap
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, Self, cast

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.contrib import splines

InferenceTypes = Any
Array = Any


class Sum(lsl.Var):
    def __init__(self, name: str) -> None:
        def _sum(*args, **kwargs):
            # the + 0. implicitly ensures correct dtype also for empty predictors
            return sum(args) + sum(kwargs.values()) + jnp.asarray(0.0)

        super().__init__(lsl.Calc(_sum), name=name)
        self.update()
        self.terms: dict[str, lsl.Var] = {}
        """Dictionary of terms in this predictor."""

    def update(self) -> Self:
        return cast(Self, super().update())

    def __add__(self, other: lsl.Var) -> Self:
        self.value_node.add_inputs(other)
        self.terms[other.name] = other
        return self.update()

    def __iadd__(self, other: lsl.Var | Sequence[lsl.Var]) -> Self:
        if isinstance(other, lsl.Var):
            self.append(other)
        else:
            self.extend(other)
        return self

    def append(self, term: lsl.Var) -> None:
        if not isinstance(term, lsl.Var):
            raise TypeError(f"{term} is of unsupported type {type(term)}.")

        if term.name in self.terms:
            raise RuntimeError(f"{self} already contains a term of name {term.name}.")

        self.value_node.add_inputs(term)
        self.terms[term.name] = term
        self.update()

    def extend(self, terms: Sequence[lsl.Var]) -> None:
        for term in terms:
            self.append(term)

    def __getitem__(self, name) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"


class LocationIntercept(lsl.Var):
    x = None  # to make PTM._set_kwargs work seamlessly

    def __init__(
        self,
        y: lsl.Value,
        loc_model: lsl.Var,
        log_scale_model: lsl.Var,
        name: str = "",
    ) -> None:
        value_calc = lsl.Calc(self.compute_intercept, y, loc_model, log_scale_model)
        super().__init__(value_calc, name=name)
        self.update()
        # self.parameter = True

    @staticmethod
    def compute_intercept(y, loc_model, log_scale_model):
        # this is computing the following:
        # scale_model = jnp.exp(log_scale_model)
        # inv_scale_model_mean = 1 / jnp.mean(1 / scale_model)
        # return inv_scale_model_mean * residual_model_mean

        # justification:
        # 1/ scale_model = 1 / exp(log_scale_model) = exp(-log_scale_model)
        # mean(exp(-log_scale_model)) = sum(exp(-log_scale_model)) / n
        # = exp( logsumexp(-log_scale_model) ) - n
        # 1 / exp(log_mean) = exp(-log_mean)

        nobs = jnp.shape(jnp.atleast_1d(log_scale_model))[0]

        log_mean_exp = jax.nn.logsumexp(-log_scale_model) - jnp.log(nobs)
        inv_scale_model_mean = jnp.exp(-log_mean_exp)

        residual_model_mean = jnp.mean((y - loc_model) * jnp.exp(-log_scale_model))

        return inv_scale_model_mean * residual_model_mean


class LogScaleIntercept(lsl.Var):
    def __init__(
        self,
        y: lsl.Value,
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
        # self.parameter = True

    @staticmethod
    def compute_intercept(y, loc, log_scale_model):
        scale_model = jnp.exp(log_scale_model)
        exp_scale_intercept = jnp.std((y - loc) / scale_model)
        return jnp.log(exp_scale_intercept)


def loc_intercept_pgibbs(
    position_keys: Sequence[str],
    model_var: lsl.Var,
    response_name: str = "response",
    loc_model_name: str = "$\\mu(x)$",
    log_scale_model_name: str = "$\\log(\\sigma)(x)$",
    subset: Array | None = None,
) -> gs.GibbsKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    model = model_var.model
    if model is None:
        raise ValueError("The model must be set in the model variable.")

    name = position_keys[0]

    def transition(prng_key, model_state):
        pos = model.extract_position(
            [loc_model_name, log_scale_model_name, response_name, name], model_state
        )

        y = pos[response_name]
        loc_model = pos[loc_model_name]
        log_scale_model = pos[log_scale_model_name]

        if subset is not None:
            if jnp.shape(y) and jnp.shape(y)[-1] > 1:
                y = y[subset]
            if jnp.shape(loc_model) and jnp.shape(loc_model)[-1] > 1:
                loc_model = loc_model[subset]
            if jnp.shape(log_scale_model) and jnp.shape(log_scale_model)[-1] > 1:
                log_scale_model = log_scale_model[subset]

        intercept_value = LocationIntercept.compute_intercept(
            y=y,
            loc_model=loc_model,
            log_scale_model=log_scale_model,
        )

        return {name: intercept_value}

    return gs.GibbsKernel([name], transition)


def loc_intercept_mh(
    position_keys: Sequence[str],
    model_var: lsl.Var,
    response_name: str = "response",
    loc_model_name: str = "$\\mu(x)$",
    log_scale_model_name: str = "$\\log(\\sigma)(x)$",
    subset: Array | None = None,
    **kwargs,
) -> gs.MHKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    model = model_var.model
    if model is None:
        raise ValueError("The model must be set in the model variable.")

    name = position_keys[0]

    all_keys = [loc_model_name, log_scale_model_name, response_name, name]

    def proposal(prng_key, model_state, step_size) -> gs.MHProposal:
        pos = model.extract_position(all_keys, model_state)
        step_size = jnp.clip(step_size, min=0.01)

        intercept_before = pos[name]

        y = pos[response_name]
        loc_model = pos[loc_model_name]
        log_scale_model = pos[log_scale_model_name]

        if subset is not None:
            if jnp.shape(y) and jnp.shape(y)[-1] > 1:
                y = y[subset]
            if jnp.shape(loc_model) and jnp.shape(loc_model)[-1] > 1:
                loc_model = loc_model[subset]
            if jnp.shape(log_scale_model) and jnp.shape(log_scale_model)[-1] > 1:
                log_scale_model = log_scale_model[subset]

        intercept_value = LocationIntercept.compute_intercept(
            y=y,
            loc_model=loc_model,
            log_scale_model=log_scale_model,
        )

        proposal = intercept_value + step_size * jax.random.normal(prng_key)
        position = {name: proposal}

        prop_dist = tfd.Normal(loc=intercept_value, scale=step_size)
        fwd_prob = prop_dist.log_prob(proposal)
        bwd_prob = prop_dist.log_prob(intercept_before)

        log_correction = bwd_prob - fwd_prob

        return gs.MHProposal(
            position=position,  # type: ignore
            log_correction=log_correction,
        )

    return gs.MHKernel(position_keys, proposal_fn=proposal, **kwargs)


def scale_intercept_pgibbs(
    position_keys: Sequence[str],
    model_var: lsl.Var,
    response_name: str = "response",
    loc_name: str = "$\\mu$",
    log_scale_model_name: str = "$\\log(\\sigma)(x)$",
    subset: Array | None = None,
) -> gs.GibbsKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    model = model_var.model
    if model is None:
        raise ValueError("The model must be set in the model variable.")

    name = position_keys[0]

    def transition(prng_key, model_state):
        pos = model.extract_position(
            [loc_name, log_scale_model_name, response_name], model_state
        )

        y = pos[response_name]
        loc = pos[loc_name]
        log_scale_model = pos[log_scale_model_name]

        if subset is not None:
            if jnp.shape(y) and jnp.shape(y)[-1] > 1:
                y = y[subset]
            if jnp.shape(loc) and jnp.shape(loc)[-1] > 1:
                loc = loc[subset]
            if jnp.shape(log_scale_model) and jnp.shape(log_scale_model)[-1] > 1:
                log_scale_model = log_scale_model[subset]

        intercept_value = LogScaleIntercept.compute_intercept(
            y=y,
            loc=loc,
            log_scale_model=log_scale_model,
        )

        return {name: intercept_value}

    return gs.GibbsKernel([name], transition)


def scale_intercept_mh(
    position_keys: Sequence[str],
    model_var: lsl.Var,
    response_name: str = "response",
    loc_name: str = "$\\mu$",
    log_scale_model_name: str = "$\\log(\\sigma)(x)$",
    subset: Array | None = None,
    **kwargs,
) -> gs.MHKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    model = model_var.model
    if model is None:
        raise ValueError("The model must be set in the model variable.")

    name = position_keys[0]

    def proposal(prng_key, model_state, step_size) -> gs.MHProposal:
        pos = model.extract_position(
            [loc_name, log_scale_model_name, response_name, name], model_state
        )
        step_size = jnp.clip(step_size, min=0.01)

        intercept_before = pos[name]
        y = pos[response_name]
        loc = pos[loc_name]
        log_scale_model = pos[log_scale_model_name]

        if subset is not None:
            if jnp.shape(y) and jnp.shape(y)[-1] > 1:
                y = y[subset]
            if jnp.shape(loc) and jnp.shape(loc)[-1] > 1:
                loc = loc[subset]
            if jnp.shape(log_scale_model) and jnp.shape(log_scale_model)[-1] > 1:
                log_scale_model = log_scale_model[subset]

        intercept_value = LogScaleIntercept.compute_intercept(
            y=y,
            loc=loc,
            log_scale_model=log_scale_model,
        )

        proposal = intercept_value + step_size * jax.random.normal(prng_key)
        position = {name: proposal}

        prop_dist = tfd.Normal(loc=intercept_value, scale=step_size)
        fwd_prob = prop_dist.log_prob(proposal)
        bwd_prob = prop_dist.log_prob(intercept_before)

        log_correction = bwd_prob - fwd_prob

        return gs.MHProposal(
            position=position,  # type: ignore
            log_correction=log_correction,
        )

    return gs.MHKernel(position_keys, proposal_fn=proposal, **kwargs)


def identity_pgibbs(
    position_keys: Sequence[str],
    model_var: lsl.Var,
) -> gs.GibbsKernel:
    model = model_var.model
    if model is None:
        raise ValueError("The model must be set in the model variable.")

    def transition(prng_key, model_state):
        pos = model.extract_position(position_keys, model_state)
        return pos

    return gs.GibbsKernel(position_keys, transition)


class LocPredictor(lsl.Var):
    def __init__(
        self,
        intercept: lsl.Var,
        loc_predictor: Sum,
        name: str,
    ) -> None:
        self.loc_intercept = intercept
        self.loc_predictor = loc_predictor

        def _sum(*args, **kwargs):
            # the + 0. implicitly ensures correct dtype also for empty predictors
            return sum(args) + sum(kwargs.values()) + jnp.asarray(0.0)

        super().__init__(lsl.Calc(_sum), name=name)

        self._terms = {self.loc_intercept.name: self.loc_intercept}

        self.value_node.add_inputs(self.loc_intercept)
        self.value_node.add_inputs(self.loc_predictor)

        self.update()

    @classmethod
    def new_compute_intercept(
        cls,
        loc_predictor: Sum,
        log_scale_predictor: Sum,
        response_value: lsl.Value,
        name: str,
    ) -> Self:
        intercept = LocationIntercept(
            y=response_value,
            loc_model=loc_predictor,
            log_scale_model=log_scale_predictor,
            name="$\\beta_0$",
        )
        return cls(intercept, loc_predictor, name=name)

    @classmethod
    def new_psample_intercept(
        cls,
        loc_predictor: Sum,
        log_scale_predictor: Sum,
        response_name: str,
        name: str,
        response_value: lsl.Value | None = None,
        subset: Array | None = None,
    ) -> Self:
        if response_value is not None:
            rval = (
                response_value.value if subset is None else response_value.value[subset]
            )

            if jnp.shape(loc_predictor.value):
                locval = (
                    loc_predictor.value
                    if subset is None
                    else loc_predictor.value[subset]
                )
            else:
                locval = loc_predictor.value

            if jnp.shape(log_scale_predictor.value):
                scaleval = (
                    log_scale_predictor.value
                    if subset is None
                    else log_scale_predictor.value[subset]
                )
            else:
                scaleval = log_scale_predictor.value

            intercept_value = LocationIntercept.compute_intercept(
                y=rval,
                loc_model=locval,
                log_scale_model=scaleval,
            )
        else:
            intercept_value = 0.0

        if not response_value:
            raise ValueError("Need response value.")
        if not response_value.name:
            raise ValueError("Response value must have name.")

        kernel_kwargs = {
            "response_name": response_value.name,
            "loc_model_name": loc_predictor.name,
            "log_scale_model_name": log_scale_predictor.name,
            "model_var": loc_predictor,
            "subset": subset,
        }

        intercept = lsl.Var.new_param(
            intercept_value,
            name="$\\beta_0$",
            inference=gs.MCMCSpec(loc_intercept_pgibbs, kernel_kwargs=kernel_kwargs),
        )

        return cls(intercept, loc_predictor, name=name)

    @classmethod
    def new_sample_mh_intercept(
        cls,
        loc_predictor: Sum,
        log_scale_predictor: Sum,
        response_name: str,
        name: str,
        response_value: lsl.Value | None = None,
        subset: Array | None = None,
        mh_kwargs: dict | None = None,
    ) -> Self:
        if response_value is not None:
            rval = (
                response_value.value if subset is None else response_value.value[subset]
            )

            if jnp.shape(loc_predictor.value):
                locval = (
                    loc_predictor.value
                    if subset is None
                    else loc_predictor.value[subset]
                )
            else:
                locval = loc_predictor.value

            if jnp.shape(log_scale_predictor.value):
                scaleval = (
                    log_scale_predictor.value
                    if subset is None
                    else log_scale_predictor.value[subset]
                )
            else:
                scaleval = log_scale_predictor.value

            intercept_value = LocationIntercept.compute_intercept(
                y=rval,
                loc_model=locval,
                log_scale_model=scaleval,
            )
        else:
            intercept_value = 0.0

        if not response_value:
            raise ValueError("Need response value.")
        if not response_value.name:
            raise ValueError("Response value must have name.")

        kernel_kwargs = {
            "response_name": response_value.name,
            "loc_model_name": loc_predictor.name,
            "log_scale_model_name": log_scale_predictor.name,
            "model_var": loc_predictor,
            "subset": subset,
        }

        if mh_kwargs is not None:
            kernel_kwargs |= mh_kwargs

        intercept = lsl.Var.new_param(
            intercept_value,
            name="$\\beta_0$",
            inference=gs.MCMCSpec(loc_intercept_mh, kernel_kwargs=kernel_kwargs),
        )

        return cls(intercept, loc_predictor, name=name)

    @classmethod
    def new_constant_intercept(
        cls,
        loc_predictor: Sum,
        name: str,
    ) -> Self:
        intercept_value = 0.0

        kernel_kwargs = {
            "model_var": loc_predictor,
        }

        intercept = lsl.Var.new_param(
            intercept_value,
            name="$\\beta_0$",
            inference=gs.MCMCSpec(identity_pgibbs, kernel_kwargs=kernel_kwargs),
        )

        return cls(intercept, loc_predictor, name=name)

    def update(self):
        self.loc_predictor.update()
        self.loc_intercept.update()
        return super().update()

    def __add__(self, other: lsl.Var) -> Self:
        self.loc_predictor += other
        self.loc_predictor.update()
        return self.update()

    def __iadd__(self, other: lsl.Var) -> Self:
        self.loc_predictor += other
        self.loc_predictor.update()
        return self.update()

    def __getitem__(self, name) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"

    @property
    def terms(self):
        return ChainMap(self._terms, self.loc_predictor.terms)


class ScalePredictor(lsl.Var):
    def __init__(
        self,
        intercept: lsl.Var,
        log_scale_predictor: Sum,
        name: str,
    ) -> None:
        self.log_scale_intercept = intercept

        self.log_scale_predictor = log_scale_predictor

        def _exp_sum(*args, **kwargs):
            # the + 0. implicitly ensures correct dtype also for empty predictors
            return jnp.exp(sum(args) + sum(kwargs.values()) + jnp.asarray(0.0))

        super().__init__(lsl.Calc(_exp_sum), name=name)

        self._terms = {self.log_scale_intercept.name: self.log_scale_intercept}

        self.value_node.add_inputs(self.log_scale_intercept)
        self.value_node.add_inputs(self.log_scale_predictor)

        self.update()

    @classmethod
    def new_compute_intercept(
        cls,
        loc: LocPredictor,
        log_scale_predictor: Sum,
        response_value: lsl.Value,
        name: str,
    ) -> Self:
        intercept = LogScaleIntercept(
            y=response_value,
            loc=loc,
            log_scale_model=log_scale_predictor,
            name="$\\gamma_0$",
        )
        return cls(
            intercept=intercept, log_scale_predictor=log_scale_predictor, name=name
        )

    @classmethod
    def new_psample_intercept(
        cls,
        loc: LocPredictor,
        log_scale_predictor: Sum,
        response_name: str,
        name: str,
        response_value: lsl.Value | None = None,
        subset: Array | None = None,
    ) -> Self:
        if response_value is not None:
            rval = (
                response_value.value if subset is None else response_value.value[subset]
            )
            if jnp.shape(loc.value):
                locval = loc.value if subset is None else loc.value[subset]
            else:
                locval = loc.value

            if jnp.shape(log_scale_predictor.value):
                scaleval = (
                    log_scale_predictor.value
                    if subset is None
                    else log_scale_predictor.value[subset]
                )
            else:
                scaleval = log_scale_predictor.value

            intercept_value = LogScaleIntercept.compute_intercept(
                y=rval,
                loc=locval,
                log_scale_model=scaleval,
            )
        else:
            intercept_value = 0.0

        if not response_value:
            raise ValueError("Need response value.")
        if not response_value.name:
            raise ValueError("Response value must have name.")

        kernel_kwargs = {
            "response_name": response_value.name,
            "loc_name": loc.name,
            "log_scale_model_name": log_scale_predictor.name,
            "model_var": loc,
            "subset": subset,
        }

        intercept = lsl.Var.new_param(
            intercept_value,
            name="$\\gamma_0$",
            inference=gs.MCMCSpec(scale_intercept_pgibbs, kernel_kwargs=kernel_kwargs),
        )

        return cls(
            intercept=intercept, log_scale_predictor=log_scale_predictor, name=name
        )

    @classmethod
    def new_sample_mh_intercept(
        cls,
        loc: LocPredictor,
        log_scale_predictor: Sum,
        response_name: str,
        name: str,
        response_value: lsl.Value | None = None,
        subset: Array | None = None,
        mh_kwargs: dict | None = None,
    ) -> Self:
        if response_value is not None:
            rval = (
                response_value.value if subset is None else response_value.value[subset]
            )
            if jnp.shape(loc.value):
                locval = loc.value if subset is None else loc.value[subset]
            else:
                locval = loc.value

            if jnp.shape(log_scale_predictor.value):
                scaleval = (
                    log_scale_predictor.value
                    if subset is None
                    else log_scale_predictor.value[subset]
                )
            else:
                scaleval = log_scale_predictor.value

            intercept_value = LogScaleIntercept.compute_intercept(
                y=rval,
                loc=locval,
                log_scale_model=scaleval,
            )
        else:
            intercept_value = 0.0

        if not response_value:
            raise ValueError("Need response value.")
        if not response_value.name:
            raise ValueError("Response value must have name.")

        kernel_kwargs = {
            "response_name": response_value.name,
            "loc_name": loc.name,
            "log_scale_model_name": log_scale_predictor.name,
            "model_var": loc,
            "subset": subset,
        }

        if mh_kwargs is not None:
            kernel_kwargs |= mh_kwargs

        intercept = lsl.Var.new_param(
            intercept_value,
            name="$\\gamma_0$",
            inference=gs.MCMCSpec(scale_intercept_mh, kernel_kwargs=kernel_kwargs),
        )

        return cls(
            intercept=intercept, log_scale_predictor=log_scale_predictor, name=name
        )

    @classmethod
    def new_constant_intercept(
        cls,
        loc: LocPredictor,
        log_scale_predictor: Sum,
        name: str,
    ) -> Self:
        intercept_value = 0.0

        kernel_kwargs = {
            "model_var": loc,
        }

        intercept = lsl.Var.new_param(
            intercept_value,
            name="$\\gamma_0$",
            inference=gs.MCMCSpec(identity_pgibbs, kernel_kwargs=kernel_kwargs),
        )

        return cls(
            intercept=intercept, log_scale_predictor=log_scale_predictor, name=name
        )

    def update(self):
        self.log_scale_predictor.update()
        self.log_scale_intercept.update()
        return super().update()

    def __add__(self, other: lsl.Var) -> Self:
        self.log_scale_predictor += other
        self.log_scale_predictor.update()
        return self.update()

    def __iadd__(self, other: lsl.Var) -> Self:
        self.log_scale_predictor += other
        self.log_scale_predictor.update()
        return self.update()

    def __getitem__(self, name) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"

    @property
    def terms(self):
        return ChainMap(self._terms, self.log_scale_predictor.terms)


def setup_loc_scale(
    loc_intercept: Literal[
        "compute", "pseudo_sample", "sample", "constant", "sample_mh"
    ],
    scale_intercept: Literal[
        "compute", "pseudo_sample", "sample", "constant", "sample_mh"
    ],
    response_value: lsl.Value | None = None,
    response_name: str = "response",
    loc_intercept_inference: InferenceTypes = None,
    scale_intercept_inference: InferenceTypes = None,
    loc_name: str = "$\\mu$",
    scale_name: str = "$\\sigma$",
    subset: Array | None = None,
    mh_kwargs: dict | None = None,
) -> tuple[LocPredictor, ScalePredictor]:
    loc_model = Sum(loc_name + "$(x)$")
    scale_model = Sum("$\\log($" + scale_name + "$)(x)$")

    if mh_kwargs is None:
        mh_kwargs = {
            "initial_step_size": 0.1,
            "da_tune_step_size": False,
            "da_target_accept": 0.8,
        }

    if loc_intercept == "compute" or scale_intercept == "compute":
        if response_value is None:
            raise ValueError(
                "To compute the intercept, a response value node must be provided."
            )

    if loc_intercept == "compute":
        loc = LocPredictor.new_compute_intercept(
            loc_predictor=loc_model,
            log_scale_predictor=scale_model,
            response_value=response_value,  # type: ignore
            name=loc_name,
        )
    elif loc_intercept == "pseudo_sample":
        if loc_intercept_inference is None:
            loc = LocPredictor.new_psample_intercept(
                loc_predictor=loc_model,
                log_scale_predictor=scale_model,
                response_name=response_name,
                name=loc_name,
                response_value=response_value,
                subset=subset,
            )
        else:
            raise ValueError("Inference defined, but pseudo-sampling selected.")

    elif loc_intercept == "sample_mh":
        if loc_intercept_inference is None:
            loc = LocPredictor.new_sample_mh_intercept(
                loc_predictor=loc_model,
                log_scale_predictor=scale_model,
                response_name=response_name,
                name=loc_name,
                response_value=response_value,
                subset=subset,
                mh_kwargs=mh_kwargs,
            )
        else:
            raise ValueError(
                "Inference defined, but sampling via Metropolis-Hastings selected."
            )

    elif loc_intercept == "sample":
        intercept = lsl.Var.new_param(
            0.0,
            name="$\\beta_0$",
            inference=loc_intercept_inference,
        )
        loc = LocPredictor(intercept, loc_model, name=loc_name)

    elif loc_intercept == "constant":
        if loc_intercept_inference is not None:
            raise ValueError(
                f"{loc_intercept=} and {loc_intercept_inference=} are incompatible."
            )

        loc = LocPredictor.new_constant_intercept(
            loc_predictor=loc_model,
            name=loc_name,
        )

    if scale_intercept == "compute":
        scale = ScalePredictor.new_compute_intercept(
            loc=loc,
            log_scale_predictor=scale_model,
            response_value=response_value,  # type: ignore
            name=scale_name,
        )
    elif scale_intercept == "pseudo_sample":
        if scale_intercept_inference is None:
            scale = ScalePredictor.new_psample_intercept(
                loc=loc,
                log_scale_predictor=scale_model,
                response_name=response_name,
                name=scale_name,
                response_value=response_value,
                subset=subset,
            )
        else:
            raise ValueError("Inference defined, but pseudo-sampling selected.")
    elif scale_intercept == "sample_mh":
        if scale_intercept_inference is None:
            scale = ScalePredictor.new_sample_mh_intercept(
                loc=loc,
                log_scale_predictor=scale_model,
                response_name=response_name,
                name=scale_name,
                response_value=response_value,
                subset=subset,
                mh_kwargs=mh_kwargs,
            )
        else:
            raise ValueError(
                "Inference defined, but sampling via Metropolis-Hastings selected."
            )
    elif scale_intercept == "sample":
        intercept = lsl.Var.new_param(
            0.0,
            name="$\\gamma_0$",
            inference=scale_intercept_inference,
        )
        scale = ScalePredictor(intercept, scale_model, name=scale_name)
    elif scale_intercept == "constant":
        if scale_intercept_inference is not None:
            raise ValueError(
                f"{scale_intercept=} and {scale_intercept_inference=} are incompatible."
            )

        scale = ScalePredictor.new_constant_intercept(
            loc=loc,
            log_scale_predictor=scale_model,
            name=scale_name,
        )

    return loc, scale


def sfn(exp_shape):
    order = 3
    p = jnp.shape(exp_shape)[-1] + 1

    outer_border = exp_shape[..., jnp.array([0, -1])] / 6
    inner_border = 5 * exp_shape[..., jnp.array([1, -2])] / 6
    middle = exp_shape[..., 2:-2]
    summed_exp_shape = (
        outer_border.sum(axis=-1, keepdims=True)
        + inner_border.sum(axis=-1, keepdims=True)
        + middle.sum(axis=-1, keepdims=True)
    )
    return (1 / (p - order)) * summed_exp_shape


def log_sfn(shape):
    order = 3
    J = jnp.shape(shape)[-1] + 1

    a = jnp.full((J - 1,), fill_value=1.0 / 6.0)
    a = a.at[-2:].set(0.0)

    b = jnp.full((J - 1,), fill_value=2.0 / 3.0)
    b = b.at[0].set(0.0)
    b = b.at[-1].set(0.0)

    c = jnp.full((J - 1,), fill_value=1.0 / 6.0)
    c = c.at[:2].set(0.0)

    log_w = jnp.log(a + b + c)

    log_T = jax.scipy.special.logsumexp(shape + log_w)

    return log_T - jnp.log(J - order)


def cumsum_leading_zero(exp_shape: Array) -> Array:
    """Cumulative sum with a leading zero."""
    zeros_shape = jnp.shape(exp_shape)[:-1] + (1,)
    exp_shape = jnp.concatenate((jnp.zeros(zeros_shape), exp_shape), axis=-1)
    return jnp.cumsum(exp_shape, axis=-1)


def normalization_coef(shape: Array, dknots: Array) -> Array:
    """
    Constructs the spline coefficients sucht that the average slope over the domain
    is one.
    """
    exp_shape = jnp.exp(shape)
    cumsum_exp_shape = cumsum_leading_zero(exp_shape)
    coef = (dknots / sfn(exp_shape)) * cumsum_exp_shape
    return coef


def normalization_coef_log(shape: Array, dknots: Array) -> Array:
    log_s = log_sfn(shape)
    corrected_coef = shape - log_s + jnp.log(dknots)
    return corrected_coef


class MISplineCoef:
    def __init__(self, knots: Array) -> None:
        self.knots = knots
        self.k1 = knots[3]
        B = splines.basis_matrix(jnp.atleast_1d(knots[3]), knots, 3)
        B0 = splines.basis_matrix(jnp.zeros(1), knots, 3)
        S = jnp.tril(jnp.ones((B.shape[-1], B.shape[-1])))  # for cumulative sum
        self.B = B @ S
        self.B0 = B0 @ S

        self.step = jnp.diff(knots).mean()

    def _add_intercept_and_exponentiate(self, log_increments, intercept, log_slope):
        exp_coef = jnp.exp(log_increments)
        prelim_coef = jnp.concatenate((jnp.zeros(1), exp_coef), axis=-1)
        offset = (self.B @ prelim_coef) - self.k1

        full_coef = jnp.concatenate((-offset + intercept, exp_coef), axis=-1)

        fx_at_zero = self.B0 @ full_coef

        coef2 = jnp.exp(log_increments + log_slope)
        full_coef2 = jnp.concatenate((-offset + intercept, coef2), axis=-1)

        fx_at_zero2 = self.B0 @ full_coef2

        diff_at_zero = (fx_at_zero2 - fx_at_zero).squeeze()
        coef3 = full_coef2.at[..., 0].set(full_coef2[..., 0] - diff_at_zero)

        return coef3

    def get_ptm_fn(self) -> Callable[[Array, Array, Array], Array]:
        nparam = len(self.knots) - 4 - 1
        zeros = jnp.zeros((nparam,))

        @jax.vmap
        def avg_slope_one(latent_log_increments):
            full_coef_including_leading_zero = normalization_coef(
                latent_log_increments, self.step
            )
            increments = jnp.diff(full_coef_including_leading_zero)

            return jnp.log(increments)

        add_intercept_and_exponentiate = jax.vmap(self._add_intercept_and_exponentiate)

        def compute_coef(log_increments, intercept, log_slope):
            log_increments = jnp.atleast_2d(log_increments + zeros)
            intercept = jnp.expand_dims(jnp.atleast_1d(intercept), -1)
            log_slope = jnp.expand_dims(jnp.atleast_1d(log_slope), -1)

            # log_increments_slope_one = avg_slope_one(log_increments)
            log_increments_slope_one = jax.vmap(normalization_coef_log, (0, None))(
                log_increments, self.step
            )
            full_coef = add_intercept_and_exponentiate(
                log_increments_slope_one, intercept, log_slope
            )
            return full_coef

        return compute_coef

    def get_ptm_fn_squeeze(self) -> Callable[[Array, Array, Array], Array]:
        fn = self.get_ptm_fn()

        def compute_coef(log_increments, intercept, log_slope):
            return fn(log_increments, intercept, log_slope).squeeze()

        return compute_coef

    def get_onion_fn(self) -> Callable[[Array, Array, Array], Array]:
        m = len(self.knots) - 6

        dk = self.step
        dk_array = jnp.full((3,), fill_value=dk)
        dk_array = jnp.log(dk_array)
        latent_correction_denominator = jnp.log((m - 5) * self.step)

        nparam = len(self.knots) - 11
        zeros = jnp.zeros((nparam,))

        @jax.vmap
        def onion_correction(latent_coef):
            latent_correction_numerator = jax.nn.logsumexp(latent_coef)
            latent_correction = (
                latent_correction_numerator - latent_correction_denominator
            )

            latent_coef = latent_coef - latent_correction
            full_latent_coef = jnp.concatenate((dk_array, latent_coef, dk_array))
            return full_latent_coef

        add_intercept_and_exponentiate = jax.vmap(self._add_intercept_and_exponentiate)

        def compute_coef(log_increments, intercept, log_slope):
            log_increments = jnp.atleast_2d(log_increments + zeros)
            intercept = jnp.expand_dims(jnp.atleast_1d(intercept), -1)
            log_slope = jnp.expand_dims(jnp.atleast_1d(log_slope), -1)

            log_increments_slope_one = onion_correction(log_increments)
            full_coef = add_intercept_and_exponentiate(
                log_increments_slope_one, intercept, log_slope
            )
            return full_coef

        return compute_coef

    def get_onion_fn_simple(self) -> Callable[[Array], Array]:
        m = len(self.knots) - 6

        dk = self.step
        dk_array = jnp.full((3,), fill_value=dk)
        dk_array = jnp.log(dk_array)
        latent_correction_denominator = jnp.log((m - 5) * self.step)

        nparam = len(self.knots) - 11
        zeros = jnp.zeros((nparam,))

        def onion_correction(latent_coef):
            latent_correction_numerator = jax.nn.logsumexp(latent_coef)
            latent_correction = (
                latent_correction_numerator - latent_correction_denominator
            )

            latent_coef = latent_coef - latent_correction
            full_latent_coef = jnp.concatenate((dk_array, latent_coef, dk_array))
            return full_latent_coef

        def compute_coef(log_increments):
            log_increments = log_increments + zeros

            log_increments_slope_one = onion_correction(log_increments)
            exp_coef = jnp.exp(log_increments_slope_one)

            full_coef = jnp.r_[(self.knots[2], exp_coef)]

            return full_coef

        return compute_coef


class SimplePTMPredictor(Sum):
    def __init__(self, name: str, fn: Callable[[Array], Array]) -> None:
        fn = jax.jit(fn)

        def _sum(*args, **kwargs):
            # the + 0. implicitly ensures correct dtype also for empty predictors
            return fn(sum(args) + sum(kwargs.values()) + 0.0)

        super(Sum, self).__init__(lsl.Calc(_sum), name=name)
        self.update()
        self.terms: dict[str, lsl.Var] = {}

    @classmethod
    def new_sum(cls, name: str) -> Self:
        def fn(x):
            return x

        return cls(name, fn=fn)

    @classmethod
    def new_ptm(cls, knots: Array, name: str) -> Self:
        mispline = MISplineCoef(knots)
        fn = partial(mispline.get_ptm_fn(), intercept=0.0, log_slope=0.0)  # type: ignore
        return cls(name, fn=fn)

    @classmethod
    def new_ptm_squeeze(cls, knots: Array, name: str) -> Self:
        mispline = MISplineCoef(knots)
        fn = partial(mispline.get_ptm_fn_squeeze(), intercept=0.0, log_slope=0.0)  # type: ignore
        return cls(name, fn=fn)

    @classmethod
    def new_onion(cls, knots: Array, name: str) -> Self:
        mispline = MISplineCoef(knots)
        fn = partial(mispline.get_onion_fn(), intercept=0.0, log_slope=0.0)  # type: ignore
        return cls(name, fn=fn)

    @classmethod
    def new_onion_simple(cls, knots: Array, name: str) -> Self:
        mispline = MISplineCoef(knots)
        fn = mispline.get_onion_fn_simple()  # type: ignore
        return cls(name, fn=fn)


class PTMPredictor(lsl.Var):
    def __init__(self, name: str, fn: Callable[[Array, Array, Array], Array]) -> None:
        self.intercept = Sum(name + "_intercept")
        self.log_slope = Sum(name + "_log_slipe")
        self.coef = Sum(name + "_coef")

        super().__init__(lsl.Calc(fn, self.coef, self.intercept, self.coef), name=name)

    @classmethod
    def new_ptm(cls, knots: Array, name: str) -> Self:
        mispline = MISplineCoef(knots)
        return cls(name, fn=mispline.get_ptm_fn())

    @classmethod
    def new_onion(cls, knots: Array, name: str) -> Self:
        mispline = MISplineCoef(knots)
        return cls(name, fn=mispline.get_onion_fn())
