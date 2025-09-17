import jax.numpy as jnp

import liesel_ptm as ptm


class TestPTMCoef:
    def test_init_ridge(self) -> None:
        nparam = 10
        knots = ptm.LogIncKnots(a=-4.0, b=4.0, nparam=nparam)
        scale = ptm.ScaleWeibull(1.0, scale=1.0, name="scale")
        coef = ptm.PTMCoef.new_ridge(knots=knots.knots, scale=scale, name="coef")

        assert coef.value.shape == (nparam,)
        assert jnp.allclose(coef.penalty, jnp.eye(nparam))

    def test_init_rw1(self) -> None:
        nparam = 10
        knots = ptm.LogIncKnots(a=-4.0, b=4.0, nparam=nparam)
        scale = ptm.ScaleWeibull(1.0, scale=1.0, name="scale")
        coef = ptm.PTMCoef.new_rw1_sumzero(knots=knots.knots, scale=scale, name="coef")

        assert coef.value.shape == (nparam,)
        assert jnp.allclose(coef.penalty, jnp.eye(nparam - 1))
