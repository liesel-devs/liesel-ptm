import jax.numpy as jnp
import numpy as np
import pytest

from liesel_ptm.inverse_fn import (
    approximate_inverse,
    initial_inverse_guess,
    invert_fn,
    numerical_inverse_jax,
)


def test_initial_inverse_guess() -> None:
    y = np.linspace(-5, 2, 200)
    z = np.exp(y)

    ynew = initial_inverse_guess(y, z, z)
    assert y == pytest.approx(ynew)

    ynew0 = initial_inverse_guess(y, z, z[0])
    assert y[0] == pytest.approx(ynew0)


def test_inverse() -> None:
    y = np.linspace(-5, 2, 200)
    z = np.exp(y)

    iexp = invert_fn(np.exp)
    yhat = iexp(z, None).x

    assert yhat == pytest.approx(y, abs=1e-3)

    znew = np.random.uniform(0.1, 1, 10)
    yhat = iexp(znew, None).x
    assert yhat == pytest.approx(np.log(znew), abs=1e-3)


def test_inverse_jax() -> None:
    y = np.linspace(-5, 2, 200)
    z = jnp.exp(y)

    iexp = numerical_inverse_jax(jnp.exp)
    yhat = iexp(z, jnp.ones_like(y))

    assert yhat == pytest.approx(y, abs=1e-3)

    znew = np.random.uniform(0.1, 1, 10)
    yhat = iexp(znew, jnp.ones_like(znew))
    assert yhat == pytest.approx(np.log(znew), abs=1e-3)


def test_approx_inverse() -> None:
    y = np.random.normal(size=(10,))
    z = np.exp(y)

    ygrid = np.linspace(-5.0, 5.0, 200)
    zgrid = np.exp(ygrid)

    y_approx = approximate_inverse(ygrid, zgrid, z)

    assert y_approx == pytest.approx(y, abs=1e-3)


def test_approx_inverse_batched() -> None:
    y = np.random.normal(size=(4, 9, 20))
    z = np.exp(y)

    ygrid = np.linspace(-5.0, 5.0, 500)
    zgrid = np.exp(ygrid)

    y_approx = approximate_inverse(ygrid, zgrid, z)

    assert y_approx.shape == (4, 9, 20)
    assert y_approx == pytest.approx(y, abs=1e-3)
