import jax.numpy as jnp
import pytest
from jax.random import key, normal, uniform

from liesel_ptm import constraint as co
from liesel_ptm.bspline.approx import bspline_basis, equidistant_knots
from liesel_ptm.penalty import Penalty

x = uniform(key(1), (15,))
knots = equidistant_knots(x, n_param=7, order=3)
basis = bspline_basis(x, knots, 3)
nparam = basis.shape[-1]
K = Penalty.pspline(nparam)


class TestLinearConstraintEVD:
    @pytest.mark.parametrize("seed", range(10))
    def test_sumzero_coef(self, seed) -> None:
        Cbar = co.LinearConstraintEVD.sumzero_coef(nparam)
        a = normal(key(seed), Cbar.shape[-1])

        b = Cbar @ a

        assert b.sum() == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("seed", range(10))
    def test_sumzero_term(self, seed) -> None:
        Cbar = co.LinearConstraintEVD.sumzero_term(basis)
        a = normal(key(seed), Cbar.shape[-1])

        fx = basis @ Cbar @ a

        assert fx.sum() == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("seed", range(10))
    def test_sumzero_term2(self, seed) -> None:
        Cbar = co.LinearConstraintEVD.sumzero_term2(basis)
        a = normal(key(seed), Cbar.shape[-1])

        fx = basis @ Cbar @ a

        assert fx.sum() == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("seed", range(10))
    def test_constant_and_linear(self, seed) -> None:
        Cbar = co.LinearConstraintEVD.constant_and_linear(x, basis)
        a = normal(key(seed), Cbar.shape[-1])

        fx = basis @ Cbar @ a

        assert fx.sum() == pytest.approx(0.0, abs=1e-4)

        nobs = jnp.shape(x)[0]
        j = jnp.ones(shape=nobs)
        X = jnp.c_[j, x]
        A = jnp.linalg.inv(X.T @ X) @ X.T

        g = A @ fx
        assert g.shape == (2,)
        assert jnp.allclose(g, 0.0, atol=1e-4)

    def test_nullspace(self) -> None:
        Cbar = co.LinearConstraintEVD._nullspace(K)

        Kz = Cbar.T @ K @ Cbar
        assert Kz.shape == (nparam - 2, nparam - 2)
        assert jnp.allclose(Kz, jnp.eye(nparam - 2), atol=1e-4)
