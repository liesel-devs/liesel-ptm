import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_ptm.swap_dists import TemporarilySwapDists, spec_bounded


class TestSpec:
    def test_spec_bounded(self):
        v = lsl.Var.new_param(
            1.0,
            distribution=lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.001),
            name="v",
        )
        spec = spec_bounded(v)

        model = lsl.Model([v])

        with TemporarilySwapDists(model, pairs=[spec]) as m:
            v2 = m.vars[v.name]
            v2t = m.vars[v.name + "_transformed"]

            assert v2.weak
            assert v2t.strong

            v2t.value = -10.0
            assert v2t.value == pytest.approx(-10.0)
            assert v2.value > 0.0

            v2t.value = -100.0
            assert v2t.value == pytest.approx(-100.0)
            assert v2.value > 0.0

            v2t.value = -1000.0
            assert v2t.value == pytest.approx(-1000.0)
            assert v2.value > 0.0

            v2val = v2.value

        assert v.value == pytest.approx(v2val)
