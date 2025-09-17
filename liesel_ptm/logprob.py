from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.flatten_util
import liesel.goose as gs

Array = Any


class LogProb:
    """
    Interface for evaluating the unnormalized log probability of a Liesel model.

    Also provides access to the first and second derivatives.

    Parameters
    ----------
    model
        A Liesel model instance.
    component
        Which component of the model's log probability to evaluate.
    diff_mode
        Which auto-diff mode to use for the Hessian.
    """

    def __init__(
        self,
        model: gs.ModelInterface,
        component: Literal["log_prob", "log_lik", "log_prior"] = "log_prob",
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model
        # TODO diff mode is wrong for gradient;
        # we probably never want to use forward mode here.
        self._grad_fn = jax.grad(self.log_prob)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.component = component
        self.diff_mode = diff_mode

    def __call__(self, position: dict[str, Array | float], model_state) -> Array:
        return self.log_prob(position=position, model_state=model_state)

    def log_prob(self, position: dict[str, Array | float], model_state) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        updated_state = self.model.update_state(position, model_state)  # type: ignore
        return updated_state[f"_model_{self.component}"].value

    def grad(self, position: dict[str, Array | float], model_state) -> dict[str, Array]:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(position, model_state)

    def hessian(
        self, position: dict[str, Array | float], model_state
    ) -> dict[str, Array]:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(position, model_state)


class FlatLogProb:
    """
    Interface for evaluating the unnormalized log probability of a Liesel model.

    Also provides access to the first and second derivatives.
    The methods :meth:`.FlatLogProb.grad` and :meth:`.FlatLogProb.hessian` are
    flattened, which means the expect arrays as inputs and return arrays.

    Parameters
    ----------
    model
        A Liesel model instance.
    param_names
        Names of the variables at which to evaluate the log probability. Other \
        variables will be kept fixed at their current values in the model state. \
        If ``None`` (default), all *parameter* variables in the model will be used.
    component
        Which component of the model's log probability to evaluate.
    diff_mode
        Which auto-diff mode to use for the Hessian.
    """

    def __init__(
        self,
        model: gs.ModelInterface,
        model_state,
        param_names: Sequence[str],
        component: Literal["log_prob", "log_lik", "log_prior"] = "log_prob",
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model

        position = self.model.extract_position(param_names, model_state)
        _, unravel_fn = jax.flatten_util.ravel_pytree(position)
        self.unravel_fn = unravel_fn

        self._grad_fn = jax.grad(self)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.component = component
        self.diff_mode = diff_mode

    def __call__(self, flat_position: Array, model_state) -> Array:
        return self.log_prob(flat_position=flat_position, model_state=model_state)

    def log_prob(self, flat_position: Array, model_state) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        position = self.unravel_fn(flat_position)
        updated_state = self.model.update_state(position, model_state)
        return updated_state[f"_model_{self.component}"].value

    def grad(self, flat_position: Array, model_state) -> Array:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(flat_position, model_state)

    def hessian(self, flat_position: Array, model_state) -> Array:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(flat_position, model_state)
