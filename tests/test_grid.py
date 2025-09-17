import jax
import jax.numpy as jnp

from liesel_ptm.util.inverse_interpax import find_grid


def test_find_grid():
    fn = jnp.exp

    x = jnp.linspace(-5.0, 1.0, 100)
    y = fn(x)

    _, ygrid = find_grid(y=y, fn=fn, initial_xmin=0.5, initial_xmax=1.0, ngrid=100)

    assert jnp.all(ygrid.min() < y)
    assert jnp.all(ygrid.max() > y)


def test_find_grid_jit():
    fn = jnp.exp

    x = jnp.linspace(-5.0, 1.0, 100)
    y = fn(x)

    find_grid_jit = jax.jit(find_grid, static_argnums=(1, 2, 3, 4))

    _, ygrid = find_grid_jit(y=y, fn=fn, initial_xmin=0.5, initial_xmax=1.0, ngrid=100)
    assert jnp.all(ygrid.min() < y)
    assert jnp.all(ygrid.max() > y)
