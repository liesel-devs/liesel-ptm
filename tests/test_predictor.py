import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest

import liesel_ptm as ptm

nparam = 10
ptm_knots = ptm.LogIncKnots(-4.0, 4.0, nparam=nparam)
onion_knots = ptm.bspline.OnionKnots(-4.0, 4.0, nparam=nparam)


class TestSimplePTMPredictor:
    def test_init_ptm(self) -> None:
        knots = ptm_knots
        trafo = ptm.predictor.SimplePTMPredictor.new_ptm(ptm_knots.knots, name="trafo")

        assert trafo is not None
        assert trafo.value[0, 0] == pytest.approx(knots.knots[2], abs=1e-6)
        assert jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)

        assert trafo.value.shape[-1] == nparam + 1

    def test_add_constant_term(self) -> None:
        knots = ptm_knots
        trafo = ptm.predictor.SimplePTMPredictor.new_ptm(knots.knots, name="trafo")
        trafo += lsl.Var(jnp.zeros(nparam), name="a")
        assert trafo is not None
        assert trafo.value[0, 0] == pytest.approx(knots.knots[2], abs=1e-6)
        assert jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)

        trafo += lsl.Var(jnp.ones(nparam), name="b")

        assert trafo.value[0, 0] == pytest.approx(knots.knots[2], abs=1e-6)
        assert jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)

    def test_add_varying_term(self) -> None:
        knots = ptm_knots
        trafo = ptm.predictor.SimplePTMPredictor.new_ptm(knots.knots, name="trafo")
        trafo += lsl.Var(jax.random.normal(jax.random.key(0), (nparam,)))
        assert trafo is not None

        assert trafo.value[0, 0] != pytest.approx(knots.knots[2], abs=1e-6)
        assert not jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)


class TestSimpleOnionPredictor:
    def test_init_onion(self) -> None:
        knots = onion_knots
        trafo = ptm.predictor.SimplePTMPredictor.new_onion(knots.knots, name="trafo")
        assert trafo is not None
        assert trafo.value[0, 0] == pytest.approx(knots.knots[2], abs=1e-6)
        assert jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)

        assert trafo.value.shape[-1] == nparam + 7

    def test_add_constant_term(self) -> None:
        knots = onion_knots
        trafo = ptm.predictor.SimplePTMPredictor.new_onion(knots.knots, name="trafo")
        trafo += lsl.Var(jnp.zeros(nparam), name="a")
        assert trafo is not None
        assert trafo.value[0, 0] == pytest.approx(knots.knots[2], abs=1e-6)
        assert jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)

        trafo += lsl.Var(jnp.ones(nparam), name="b")

        assert trafo.value[0, 0] == pytest.approx(knots.knots[2], abs=1e-6)
        assert jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)

    def test_add_varying_term(self) -> None:
        knots = onion_knots
        trafo = ptm.predictor.SimplePTMPredictor.new_onion(knots.knots, name="trafo")
        trafo += lsl.Var(jax.random.normal(jax.random.key(0), (nparam,)))
        assert trafo is not None

        assert trafo.value[0, 0] == pytest.approx(knots.knots[2], abs=1e-6)
        assert not jnp.allclose(trafo.value[0, 1:], knots.step, atol=1e-6)
