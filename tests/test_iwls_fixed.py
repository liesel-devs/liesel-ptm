import jax
import jax.numpy as jnp
from liesel.goose.da import DualAvgState

from liesel_ptm.iwls_fixed import IWLSFixedKernelState


def test_iwls_fixed_state_round_trips_as_pytree() -> None:
    state = IWLSFixedKernelState(
        step_size=0.1,
        da_state=DualAvgState.from_step_size(0.1),
        chol_info=jnp.eye(2),
    )

    leaves, tree = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert restored.da_state is not None
    assert restored.chol_info.shape == (2, 2)
