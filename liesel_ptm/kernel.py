from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
from jax.flatten_util import ravel_pytree
from liesel.goose.types import ModelState

Array = Any


def star_ig_gibbs(coef: lsl.Var) -> gs.GibbsKernel:
    variance_var = coef.dist_node["scale"].value_node[0]  # type: ignore
    a_value = variance_var.dist_node["concentration"].value  # type: ignore
    b_value = variance_var.dist_node["scale"].value  # type: ignore

    try:
        penalty_value = coef.dist_node["penalty"].value  # type: ignore
    except KeyError:
        penalty_value = jnp.eye(coef.value.size)
    rank_value = jnp.linalg.matrix_rank(penalty_value)

    model = coef.model
    if model is None:
        raise ValueError("The model must be set in the coefficient variable.")

    name = variance_var.name

    def transition(prng_key, model_state):
        pos = model.extract_position([coef.name], model_state)

        coef_value = pos[coef.name].squeeze()

        a_gibbs = jnp.squeeze(a_value + 0.5 * rank_value)
        b_gibbs = jnp.squeeze(b_value + 0.5 * (coef_value @ penalty_value @ coef_value))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {name: draw}

    return gs.GibbsKernel([name], transition)


def init_star_ig_gibbs(position_keys: Sequence[str], coef: lsl.Var) -> gs.GibbsKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    try:
        coef = coef.value_node[0]  # type: ignore
    except IndexError:
        pass

    variance_var = coef.dist_node["scale"].value_node[0]  # type: ignore
    name = variance_var.name

    if position_keys[0] != name:
        raise ValueError(f"The position key must be {name}.")

    return star_ig_gibbs(coef)


class IWLSKernelDiag(gs.IWLSKernel):
    def __init__(
        self,
        position_keys: Sequence[str],
        scale_key: str | None = None,
        initial_step_size: float = 0.01,
        da_target_accept: float = 0.8,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
    ) -> None:
        super().__init__(
            position_keys=position_keys,
            initial_step_size=initial_step_size,
            da_target_accept=da_target_accept,
            da_gamma=da_gamma,
            da_kappa=da_kappa,
            da_t0=da_t0,
        )

        self.scale_key = scale_key

    def _chol_info(
        self, model_state: ModelState, flat_hessian_fn: Callable[[Array], Array]
    ) -> Array:
        """
        Computes the Cholesky decomposition of the Fisher information matrix via
        :attr:`.flat_hessian_fn`.

        The flat position is extracted from the :attr:`.model_state`. If the user
        provided a :attr:`.chol_info_fn` when initializing the kernel, this function is
        called instead.
        """
        flat_position, _ = ravel_pytree(self.position(model_state))
        if self.scale_key is not None:
            scale = self.model.extract_position([self.scale_key], model_state)[
                self.scale_key
            ]
            return jnp.eye(jnp.shape(flat_position)[-1]) * (1 / scale)
        else:
            return jnp.eye(jnp.shape(flat_position)[-1])
