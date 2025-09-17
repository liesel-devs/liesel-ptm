from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.epoch import EpochState, EpochType
from liesel.goose.kernel import TransitionOutcome
from liesel.goose.mh import mh_step
from liesel.goose.types import Kernel, ModelState, TKernelState, TTransitionInfo
from liesel.model.model import TemporaryModel

Array = Any
KeyArray = Any


class ConstantConditionalKernel:
    def __init__(
        self,
        position_keys: Sequence[str],
        kernel: type[Kernel],
        constant_position: dict[str, Array],
        **kernel_kwargs,
    ) -> None:
        self.kernel = kernel(position_keys, **kernel_kwargs)  # type: ignore
        self.constant_position = constant_position
        self.constant_position_keys = list(self.constant_position)
        self.position_keys = position_keys

        self.__class__ = _attach_class_attributes(type(self), kernel)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)

    def transition(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[TKernelState, TTransitionInfo]:
        is_adaptation = EpochType.is_adaptation(epoch.config.type)

        outcome: TransitionOutcome[TKernelState, TTransitionInfo] = jax.lax.cond(
            is_adaptation,
            self._adaptive_transition,
            self._standard_transition,
            prng_key,
            kernel_state,
            model_state,
            epoch,
        )

        return outcome

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[TKernelState, TTransitionInfo]:
        """
        Performs an MCMC transition *outside* an adaptation epoch. Must be jittable.
        """
        constant_position_before = self.model.extract_position(
            self.constant_position_keys, model_state
        )
        altered_model_state = self.model.update_state(
            self.constant_position, model_state
        )

        k0, k1 = jax.random.split(prng_key)

        if hasattr(self.kernel, "_standard_transition"):
            outcome = self.kernel._standard_transition(  # type: ignore
                k0, kernel_state, altered_model_state, epoch
            )
        else:
            outcome = self.kernel.transition(  # type: ignore
                k0, kernel_state, altered_model_state, epoch
            )

        # forward probability
        # log prob of the outcome given previous state
        fwd_prob = self.model.log_prob(outcome.model_state)

        # backward probability
        # log prob of previous state given outcome
        bwd_prob = self.model.log_prob(altered_model_state)

        log_correction = bwd_prob - fwd_prob

        model_state = self.model.update_state(
            constant_position_before, outcome.model_state
        )

        proposed_position = self.model.extract_position(self.position_keys, model_state)

        info, model_state = mh_step(
            prng_key=k1,
            model=self.model,
            proposal=proposed_position,
            model_state=model_state,
            log_correction=log_correction,
        )

        outcome.model_state = model_state
        outcome.info = info

        return outcome

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[TKernelState, TTransitionInfo]:
        """
        Performs an MCMC transition *in* an adaptation epoch. Must be jittable.
        """
        constant_position_before = self.model.extract_position(
            self.constant_position_keys, model_state
        )
        altered_model_state = self.model.update_state(
            self.constant_position, model_state
        )

        k0, k1 = jax.random.split(prng_key)

        if hasattr(self.kernel, "_adaptive_transition"):
            outcome = self.kernel._adaptive_transition(  # type: ignore
                k0, kernel_state, altered_model_state, epoch
            )
        elif hasattr(self.kernel, "_standard_transition"):
            outcome = self.kernel._standard_transition(  # type: ignore
                k0, kernel_state, altered_model_state, epoch
            )
        else:
            outcome = self.kernel.transition(  # type: ignore
                k0, kernel_state, altered_model_state, epoch
            )

        # forward probability
        # log prob of the outcome given previous state
        fwd_prob = self.model.log_prob(outcome.model_state)

        # backward probability
        # log prob of previous state given outcome
        bwd_prob = self.model.log_prob(altered_model_state)

        log_correction = bwd_prob - fwd_prob

        model_state = self.model.update_state(
            constant_position_before, outcome.model_state
        )

        proposed_position = self.model.extract_position(self.position_keys, model_state)

        info, model_state = mh_step(
            prng_key=k1,
            model=self.model,
            proposal=proposed_position,
            model_state=model_state,
            log_correction=log_correction,
        )

        outcome.model_state = model_state
        outcome.info = info

        return outcome


def _attach_class_attributes(wrapper_class: type, base_class: type) -> type:
    for key, val in base_class.__dict__.items():
        if not key.startswith("__") and not callable(val):
            setattr(wrapper_class, key, val)

    return wrapper_class


def freeze_conditional(
    inference: gs.MCMCSpec, constant_position: dict[str, Array]
) -> gs.MCMCSpec:
    kernel = inference.kernel

    if isinstance(kernel, type) and issubclass(kernel, ConstantConditionalKernel):
        return inference

    inference.kernel_kwargs["kernel"] = kernel
    inference.kernel_kwargs["constant_position"] = constant_position
    inference.kernel = ConstantConditionalKernel  # type: ignore
    return inference


def extract_parental_submodel_position(
    var: lsl.Var, exclude_roles: Sequence[str] | None
) -> dict[str, Array]:
    exclude_roles = exclude_roles if exclude_roles is not None else []

    if var.model:
        trafo_model = var.model.parental_submodel(var)
        params = {
            v.name: v.value
            for v in trafo_model.parameters.values()
            if v.role not in exclude_roles
        }
        return params

    try:
        to_float32 = not jax.config.jax_enable_x64  # type: ignore
    except Exception:  # just to be really sure in case anything changes
        # this is an implicit test of whether x64 flag is enabled
        to_float32 = jnp.array(1.0).dtype == jnp.dtype("float32")
    with TemporaryModel(var, to_float32=to_float32) as trafo_model:
        params = {
            v.name: v.value
            for v in trafo_model.parameters.values()
            if v.role not in exclude_roles
        }

    return params


def fix_position(position: dict[str, Array], to: float = 0.0) -> dict[str, Array]:
    return {n: jnp.full_like(v, fill_value=to) for n, v in position.items()}


def freeze_parental_submodel(
    inference: gs.MCMCSpec | Any,
    of: lsl.Var,
    to: float = 0.0,
    exclude_roles: Sequence[str] | None = None,
) -> gs.MCMCSpec:
    if not isinstance(inference, gs.MCMCSpec):
        raise TypeError("inference should be of type liesel.goose.MCMCSpec")
    position = extract_parental_submodel_position(var=of, exclude_roles=exclude_roles)
    fixed_position = fix_position(position, to=to)
    return freeze_conditional(inference, constant_position=fixed_position)
