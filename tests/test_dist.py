import jax
import jax.numpy as jnp

import liesel_ptm as ptm
from liesel_ptm.bspline.onion import OnionKnots, OnionSpline
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


class TestDistGPTM:
    def test_vector_x(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots, subscripts="...nj,...nj->...n")
        n = 17

        k = jax.random.key(1)

        coef = jax.random.normal(k, (n, knots.nparam))

        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=False
        )

        x = jax.random.normal(k, (n,))
        lp = dist.log_prob(x)

        # shape fits
        assert lp.shape == (n,)

        y = dist.inverse_transformation(x)
        assert y.shape == (n,)

        b = 13
        coef = jax.random.normal(k, (b, n, knots.nparam))

        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=False
        )

        lp = dist.log_prob(x)

        # shape fits
        assert lp.shape == (b, n)

        y = dist.inverse_transformation(x)
        assert y.shape == (b, n)

        y = dist.sample(seed=jax.random.key(1))
        assert y.shape == (b, n)

        y = dist.sample(2, seed=jax.random.key(1))
        assert y.shape == (2, b, n)

        y = dist.quantile(jnp.full((b, n), fill_value=0.1))
        assert y.shape == (b, n)

        q01 = jnp.full((b, n), fill_value=0.1)
        q06 = jnp.full((b, n), fill_value=0.6)
        q = jnp.stack((q01, q06), axis=0)
        assert q.shape == (2, b, n)

        y = dist.quantile(q)
        assert y.shape == (2, b, n)
