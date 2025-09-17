import jax

import liesel_ptm as ptm
from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline

knots = PTMKnots(-4.0, 4.0, nparam=10)
coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
bs = PTMSpline(knots.knots)


class TestDistOneCoef:
    def test_vector_x(self):
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        x = jax.random.normal(jax.random.key(0), (30,))
        lp = dist.log_prob(x)

        # shape fits
        assert lp.shape == (30,)

    def test_scalar_x(self):
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        lp = dist.log_prob(1.0)

        # shape fits
        assert lp.shape == ()

    def test_batched_x(self):
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        x = jax.random.normal(
            jax.random.key(0),
            (
                4,
                30,
            ),
        )
        lp = dist.log_prob(x)

        # shape fits
        assert lp.shape == (4, 30)


class TestDistBatchedCoef:
    def test_vector_x(self):
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        x = jax.random.normal(jax.random.key(0), (30,))
        lp = dist.log_prob(x)

        # shape fits
        assert lp.shape == (2, 30)

    def test_scalar_x(self):
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        lp = dist.log_prob(1.0)

        # shape fits
        assert lp.shape == (2,)

    def test_batched_x(self):
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        x = jax.random.normal(
            jax.random.key(0),
            (
                2,
                30,
            ),
        )
        lp = dist.log_prob(x)

        # shape fits
        assert lp.shape == (2, 30)
