import timeit  # noqa
from functools import partial

import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pytest
from plotnine import aes, geom_line, geom_vline, ggplot

from liesel_ptm import bsplines as bs
from liesel_ptm import nodes as nd
from liesel_ptm.bsplines import OnionCoef, OnionKnots, kn
from liesel_ptm.custom_types import Array
from liesel_ptm.datagen import sample_shape
from liesel_ptm.liesel_internal import splines

nparam = 7
x = np.random.uniform(size=20)
knots = kn(x, order=3, n_params=20)
X = lsl.obs(np.random.uniform(-2, 2, size=(10, 3)), name="x1")
coef = lsl.param(np.random.uniform(0, 1, size=(3,)), name="b")


def test_extrapolating_basis_dot_slope1():
    x = np.linspace(-3, 3, 100)
    knots = kn(np.array([-2, 2]), order=3, n_params=20)
    coef = np.random.uniform(0, 1, size=(19,))
    coef = nd.cumsum_leading_zero(np.exp(coef))
    smooth = bs._extrapolating_basis_dot_fixed_slope(x, knots, coef)
    ggplot() + geom_line(aes(x, smooth))
    assert True


def test_extrapolating_basis_dot_slope_segment():
    x = np.linspace(-3, 3, 100)
    knots = kn(np.array([-2, 2]), order=3, n_params=20)
    coef = np.random.uniform(0, 1, size=(20,))
    # coef = lstm._cumsum_leading_zero(np.exp(coef))

    smooth = bs._extrapolating_basis_dot_continue_average_slope(x, knots, coef)
    ggplot() + geom_line(aes(x, smooth))
    assert True


def test_extrapolating_basis_dot_slope_point():
    x = np.linspace(-3, 3, 100)
    knots = kn(np.array([-2, 2]), order=3, n_params=20)
    coef = np.random.uniform(0, 1, size=(20,))
    # coef = lstm._cumsum_leading_zero(np.exp(coef))

    smooth = bs._extrapolating_basis_dot_continue_point_slope(x, knots, coef)
    ggplot() + geom_line(aes(x, smooth))
    assert True


def test_extrapolating_basis_dot_and_grad_slope_segment():
    x = np.linspace(-3, 3, 100)
    knots = kn(np.array([-2, 2]), order=3, n_params=20)
    coef = np.random.uniform(0, 1, size=(20,))
    # coef = lstm._cumsum_leading_zero(np.exp(coef))

    smooth = bs._extrapolating_basis_dot_continue_average_slope(x, knots, coef)
    smooth_grad = bs._extrapolating_basis_dot_grad_continue_average_slope(
        x, knots, coef
    )
    ggplot() + geom_line(aes(x, smooth)) + geom_line(aes(x, smooth_grad), color="red")

    slope_left = bs._average_slope_in_segment(coef[:4], knots)
    slope_right = bs._average_slope_in_segment(coef[-4:], knots)
    assert np.allclose(smooth_grad[x < knots[3]], slope_left)
    assert np.allclose(smooth_grad[x > knots[-4]], slope_right)


def test_extrapolating_basis_dot_and_grad_slope_fixed():
    x = np.linspace(-3, 3, 100)
    knots = kn(np.array([-2, 2]), order=3, n_params=20)
    coef = np.random.uniform(0, 1, size=(20,))
    # coef = lstm._cumsum_leading_zero(np.exp(coef))

    smooth = bs._extrapolating_basis_dot_continue_average_slope(x, knots, coef)
    smooth_grad = bs._extrapolating_basis_dot_grad_fixed_slope(
        x, knots, coef, slope_left=0.7, slope_right=0.7
    )
    ggplot() + geom_line(aes(x, smooth)) + geom_line(aes(x, smooth_grad), color="red")
    assert np.allclose(smooth_grad[x < knots[3]], 0.7)
    assert np.allclose(smooth_grad[x > knots[-4]], 0.7)


def _basis_dot_grad(
    x: Array,
    knots: Array,
    coef: Array,
):
    x = jnp.atleast_1d(x)

    basis = bs.bspline_basis_deriv(x, knots, 3)
    smooth = jnp.einsum("...ip,...p->...i", basis, coef)
    return smooth


def _basis_dot_grad_fast(
    x: Array,
    knots: Array,
    coef: Array,
):
    x = jnp.atleast_1d(x)
    dcoef = jnp.diff(coef) / jnp.diff(knots).mean()
    dcoef = jnp.concatenate((jnp.zeros(1), dcoef, jnp.zeros(1)))
    basis = bs.bspline_basis(x, knots, 2)
    smooth = jnp.einsum("...ip,...p->...i", basis, dcoef)
    return smooth


def test_basis_dot_grad_fast():
    x = np.linspace(-3.0, 3.0, 100)
    knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
    shape = sample_shape(jax.random.PRNGKey(42), 19).sample
    dknots = np.diff(knots).mean()
    coef = nd.normalization_coef(shape, dknots)

    grad1 = _basis_dot_grad(x, knots, coef)
    grad2 = bs._basis_dot_grad(x, knots, coef)
    grad3 = jnp.diag(jax.jacfwd(bs._basis_dot)(x, knots, coef))
    assert np.allclose(grad1, grad2, atol=1e-4)
    assert np.allclose(grad1, grad3, atol=1e-4)


def _basis_dot_grad2(
    x: Array,
    knots: Array,
    coef: Array,
):
    x = jnp.atleast_1d(x)

    basis = bs.bspline_basis_deriv2(x, knots, 3)
    smooth = jnp.einsum("...ip,...p->...i", basis, coef)
    return smooth


def _basis_dot_grad2_fast(
    x: Array,
    knots: Array,
    coef: Array,
):
    x = jnp.atleast_1d(x)
    dcoef = jnp.diff(coef) / jnp.diff(knots).mean()
    dcoef = jnp.diff(dcoef) / jnp.diff(knots).mean()

    basis = bs.bspline_basis(x, knots, 1)
    smooth = jnp.einsum("...ip,...p->...i", basis, bs.pad0(dcoef, 2, 2))
    return smooth


def test_basis_dot_grad2_fast():
    x = np.linspace(-3.0, 3.0, 100)
    knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
    shape = sample_shape(jax.random.PRNGKey(42), 19).sample
    dknots = np.diff(knots).mean()
    coef = nd.normalization_coef(shape, dknots)

    grad1 = _basis_dot_grad2(x, knots, coef)
    grad2 = bs._basis_dot_grad2(x, knots, coef)
    grad3 = jnp.diag(jax.jacfwd(bs._basis_dot_grad)(x, knots, coef))
    assert np.allclose(grad1, grad2, atol=1e-4)
    assert np.allclose(grad1, grad3, atol=1e-4)


@partial(jax.jit, static_argnames="order")
def bspline_basis_grad(x, knots, order):
    min_knot = knots[order]
    max_knot = knots[-(order + 1)]

    basis = splines.build_design_matrix_b_spline(x, knots[1:-1], order - 1)
    # basis_grad = splines.build_design_matrix_b_spline_derivative(x, knots, order)

    dknots = jnp.diff(knots).mean()
    D = jnp.diff(jnp.identity(jnp.shape(knots)[-1] - order - 1)).T
    basis_grad = (basis @ D) / dknots

    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis_grad)


def test_bspline_basis_grad():
    x = np.linspace(-3.0, 3.0, 100)
    knots = kn(np.array([-3.0, 3.0]), order=3, n_params=20)

    basis_grad_manual = bspline_basis_grad(x, knots, 3)
    basis_grad_auto = splines.build_design_matrix_b_spline_derivative(x, knots, 3)

    assert np.allclose(basis_grad_manual, basis_grad_auto, atol=1e-4)


def test_compare_speed_basis_dot():
    """
    Turns out, calling one of the "fast" doesn't really cut it.

    Using the seemingly slow version in _basis_dot_grad is actually somewhat faster.
    """
    x = np.linspace(-3.0, 3.0, 100)
    knots = kn(np.array([-3.0, 3.0]), order=3, n_params=20)
    shape = sample_shape(jax.random.PRNGKey(42), 19).sample
    dknots = np.diff(knots).mean()
    coef = nd.normalization_coef(shape, dknots)

    def manual():
        return _basis_dot_grad_fast(x, knots, coef)

    def auto():
        return _basis_dot_grad(x, knots, coef)

    # time_manual = timeit.timeit(manual, number=10000)
    # time_auto = timeit.timeit(auto, number=10000)


def test_compare_speed_basis_grad():
    """
    Turns out, the jax derivative is actually much faster.

    Update: Nope, it was just jitted. Manual derivative is faster.
    """

    x = np.linspace(-3.0, 3.0, 100)
    knots = kn(np.array([-3.0, 3.0]), order=3, n_params=20)

    def manual():
        return bspline_basis_grad(x, knots, 3)

    def auto():
        return splines.build_design_matrix_b_spline_derivative(x, knots, 3)

    # time_manual = timeit.timeit(manual, number=1000)
    # time_auto = timeit.timeit(auto, number=1000)


def test_basis_lookup():
    x = np.linspace(-3.0, 3.0, 100)
    knots = kn(np.array([-3.0, 3.0]), order=3, n_params=20)

    basis = bs.BSplineApprox(knots, 3)
    basis_val = basis.get_basis(x)

    assert basis_val.shape == (100, 20)


def test_basis_lookup_speed():
    """
    Now this is ridiculous. The lookup is SLOWER? MUCH SLOWER?
    This must be due to jitting.

    Ok, after switching to a much more efficient search for the closest
    value, the lookup implementation is now much faster. That's nice.
    """
    x = np.linspace(-3.0, 3.0, 1000)
    knots = kn(np.array([-3.0, 3.0]), order=3, n_params=20)
    basis = bs.BSplineApprox(knots, 3)

    # @jax.jit
    def _lookup(x):
        return basis.get_basis(x)

    def lookup():
        return _lookup(x)

    def calc():
        return splines.build_design_matrix_b_spline(x, knots, 3)

    # time_lookup = timeit.timeit(lookup, number=10000)
    # time_calc = timeit.timeit(calc, number=10000)


def test_basis_dot_and_grad():
    x = np.linspace(-3.0, 3.0, 1000)
    knots = kn(np.array([-3.0, 3.0]), order=3, n_params=20)
    basis = bs.BSplineApprox(knots, 3)
    shape = sample_shape(jax.random.PRNGKey(42), 19).sample
    dknots = np.diff(knots).mean()
    coef = nd.normalization_coef(shape, dknots)

    bdot_and_grad_fn = basis.get_basis_dot_and_deriv_fn()

    bdot, grad = bdot_and_grad_fn(x, coef)

    bdot_baseline = bs._basis_dot(x, knots, coef)
    grad_baseline = bs._basis_dot_grad(x, knots, coef)
    grad2_baseline = bs._basis_dot_grad2(x, knots, coef)

    assert np.allclose(bdot, bdot_baseline, atol=0.1)
    assert np.allclose(grad, grad_baseline, atol=0.1)

    # gradient with respect to data
    bdot_grad, grad2 = jax.jacobian(bdot_and_grad_fn, argnums=0)(x, coef)
    assert np.allclose(grad, jnp.diag(bdot_grad), atol=0.1)
    assert np.allclose(jnp.diag(grad2), grad2_baseline, atol=0.15)

    # gradient with respect to data using jacobian on baseline
    # same comparison as above, but I don't need to take the diagonal
    # here.
    bdot_baseline = jax.jacobian(bs._basis_dot, argnums=0)(x, knots, coef)
    grad_baseline = jax.jacobian(bs._basis_dot_grad, argnums=0)(x, knots, coef)
    assert np.allclose(bdot_grad, bdot_baseline, atol=0.1)
    assert np.allclose(grad2, grad_baseline, atol=0.15)

    # gradient with respect to coef
    bdot_grad_coef_baseline = jax.jacobian(bs._basis_dot, argnums=2)(x, knots, coef)
    bdot_grad2_coef_baseline = jax.jacobian(bs._basis_dot_grad, argnums=2)(
        x, knots, coef
    )
    bdot_grad, grad2 = jax.jacobian(bdot_and_grad_fn, argnums=1)(x, coef)

    assert np.allclose(bdot_grad_coef_baseline, bdot_grad, atol=0.1)
    assert np.allclose(bdot_grad2_coef_baseline, grad2, atol=0.15)


def test_basis_grad2():
    x = np.linspace(-3.0, 3.0, 100)
    knots = kn(np.array([-3.0, 3.0]), order=3, n_params=20)

    basis_grad_manual = bs.bspline_basis_deriv2(x, knots, 3)
    basis_grad_auto = splines.build_design_matrix_b_spline_derivative2(x, knots, 3)

    assert np.allclose(basis_grad_manual, basis_grad_auto, atol=1e-4)


def test_single_basis_dot() -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
    shape = sample_shape(jax.random.PRNGKey(42), 19).sample
    dknots = np.diff(knots).mean()
    coef = nd.normalization_coef(shape, dknots)

    basis = bs.BSplineApprox(knots, 3)
    basis_dot_and_grad_fn = basis.get_basis_dot_and_deriv_fn()
    basis_dot_fn = basis.get_basis_dot_fn()

    smooth, _ = basis_dot_and_grad_fn(x, coef)
    smooth2 = basis_dot_fn(x, coef)

    assert np.allclose(smooth, smooth2, atol=1e-6)


class TestExtrapBSplineApprox:
    # @pytest.mark.skip(reason="Manual test")
    def test_visual_inspection(self):
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(42), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis = bs.ExtrapBSplineApprox(knots, 3, eps=0.1)
        basis_dot_and_grad_fn = basis.get_extrap_basis_dot_and_deriv_fn()
        smooth, smooth_grad = basis_dot_and_grad_fn(x, coef)

        p = (
            ggplot()
            + geom_line(aes(x, smooth))
            + geom_vline(aes(xintercept=float(basis.min_knot)), linetype="dotted")
            + geom_vline(
                aes(xintercept=float(basis.min_knot - basis.eps)), linetype="dotted"
            )
            + geom_vline(aes(xintercept=float(basis.max_knot)), linetype="dotted")
            + geom_vline(
                aes(xintercept=float(basis.max_knot + basis.eps)), linetype="dotted"
            )
        )

        p
        ggplot() + aes(x, smooth_grad) + geom_line()

        assert True

    @pytest.mark.skip(reason="Manual test")
    def test_visual_inspection_3eps(self):
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(44), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis = bs.ExtrapBSplineApprox(knots, 3, eps=3.0)
        basis_dot_and_grad_fn = basis.get_extrap_basis_dot_and_deriv_fn()
        smooth, smooth_grad = basis_dot_and_grad_fn(x, coef)

        p = (
            ggplot()
            + geom_line(aes(x, smooth))
            + geom_vline(aes(xintercept=float(basis.min_knot)), linetype="dotted")
            + geom_vline(
                aes(xintercept=float(basis.min_knot - basis.eps)), linetype="dotted"
            )
            + geom_vline(aes(xintercept=float(basis.max_knot)), linetype="dotted")
            + geom_vline(
                aes(xintercept=float(basis.max_knot + basis.eps)), linetype="dotted"
            )
        )

        p
        ggplot() + aes(x, smooth_grad) + geom_line()

        assert True

    def test_function_runs(self) -> None:
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(42), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis = bs.ExtrapBSplineApprox(knots, 3)
        basis_dot_and_grad_fn = basis.get_extrap_basis_dot_and_deriv_fn()
        smooth, smooth_grad = basis_dot_and_grad_fn(x, coef)

        assert smooth.shape == x.shape
        assert smooth_grad.shape == x.shape
        assert not all(jnp.isnan(smooth))
        assert not all(jnp.isnan(smooth_grad))

    def test_autodiff_data(self) -> None:
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(42), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis = bs.ExtrapBSplineApprox(knots, 3)
        basis_dot_and_grad_fn = basis.get_extrap_basis_dot_and_deriv_fn(
            target_slope=1.0
        )
        _, smooth_grad = basis_dot_and_grad_fn(x, coef)

        fn_grad = jax.jacobian(basis_dot_and_grad_fn, argnums=0)

        smooth_grad_auto, smooth_grad2 = fn_grad(x, coef)

        assert np.allclose(smooth_grad, jnp.diag(smooth_grad_auto))
        assert not jnp.isnan(smooth_grad2).sum()

    def test_boundary_up(self) -> None:
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(42), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis = bs.ExtrapBSplineApprox(knots, 3)
        basis_dot_and_grad_fn = basis.get_extrap_basis_dot_and_deriv_fn()
        smooth, smooth_grad = basis_dot_and_grad_fn(x, coef)

        j = jnp.searchsorted(x, basis.grid[-1])

        s2 = smooth[j - 2]
        s1 = smooth[j - 1]
        s0 = smooth[j]

        assert s2 < s1 < s0

    def test_boundary_lo(self) -> None:
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(42), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis = bs.ExtrapBSplineApprox(knots, 3)
        basis_dot_and_grad_fn = basis.get_extrap_basis_dot_and_deriv_fn()
        smooth, smooth_grad = basis_dot_and_grad_fn(x, coef)

        j = jnp.searchsorted(x, basis.grid[0])

        s2 = smooth[j - 2]
        s1 = smooth[j - 1]
        s0 = smooth[j]
        sp1 = smooth[j + 1]

        assert s2 < s1 < s0 < sp1

    def test_extra_basis_dot(self) -> None:
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(42), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis = bs.ExtrapBSplineApprox(knots, 3)
        basis_dot_and_grad_fn = basis.get_extrap_basis_dot_and_deriv_fn(
            target_slope=1.0
        )
        basis_dot_fn = basis.get_extrap_basis_dot_fn()

        smooth, _ = basis_dot_and_grad_fn(x, coef)
        smooth2 = basis_dot_fn(x, coef)

        assert np.allclose(smooth, smooth2, atol=1e-6)

    @pytest.mark.skip(reason="Manual Benchmark")
    def test_benchmark(self) -> None:
        x = np.linspace(-3.0, 3.0, 1000)
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=20)
        shape = sample_shape(jax.random.PRNGKey(42), 19).sample
        dknots = np.diff(knots).mean()
        coef = nd.normalization_coef(shape, dknots)

        basis_extrap = bs.ExtrapBSplineApprox(knots, 3)
        basis_dot_and_grad_fn_extrap = basis_extrap.get_extrap_basis_dot_and_deriv_fn(
            target_slope=1.0
        )
        basis_dot_and_grad_fn_extrap_avg = (
            basis_extrap.get_extrap_basis_dot_and_deriv_fn(target_slope=None)
        )

        basis = bs.BSplineApprox(knots, 3)
        basis_dot_and_grad_fn = basis.get_basis_dot_and_deriv_fn()

        def extrap():
            return basis_dot_and_grad_fn_extrap(x, coef)

        def extrap_avg():
            return basis_dot_and_grad_fn_extrap_avg(x, coef)

        def no_extrap():
            return basis_dot_and_grad_fn(x, coef)

        time_extrap = timeit.timeit(extrap, number=1000)  # noqa
        time_extrap_avg = timeit.timeit(extrap, number=1000)  # noqa
        time_no_extrap = timeit.timeit(no_extrap, number=1000)  # noqa


def test_searchsorted():
    grid = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
    x = 0.21

    i = jnp.searchsorted(grid, x, side="right") - 1

    assert i == 2
    assert grid[i] <= x < grid[i + 1]


class TestOnionKnots:
    def test_default_nparam(self):
        knotsa = OnionKnots(-3.0, 3.0, nparam=10)
        knotsb = kn(jnp.array([-3.0, 3.0]), n_params=10)

        assert knotsa.knots.shape[0] == (knotsb.shape[0] + 6 + 1)

    def test_values_extension(self):
        knots = OnionKnots(-3.0, 3.0, nparam=10)

        assert jnp.allclose(jnp.diff(knots.knots), knots.step)

    def test_length(self):
        knots = OnionKnots(-3.0, 3.0, nparam=20)

        nparam = 20
        nleft = 3
        nright = 3
        order = 3
        boundary_knots = order + 1
        intercept = 1

        length_knots = nparam + nleft + nright + boundary_knots + intercept
        assert knots.knots.shape[0] == length_knots

    def test_ab(self):
        knots = OnionKnots(-3.0, 3.0, nparam=10)
        assert knots.left == pytest.approx(-3.0 - 2 * knots.step, abs=1e-4)
        assert knots.right == pytest.approx(3.0 + 2 * knots.step, abs=1e-4)

    @pytest.mark.parametrize(
        "left,right,nparam",
        [(-4.0, 4.0, 10), (-4.0, 4.0, 15), (-4.0, 5.0, 10), (-5.0, 4.0, 20)],
    )
    def test_new_from_lr(self, left, right, nparam):
        knots = OnionKnots.new_from_lr(left, right, nparam=nparam)
        assert knots.left == pytest.approx(left, abs=1e-4)
        assert knots.right == pytest.approx(right, abs=1e-4)


class TestOnionCoef:
    def test_compute_coef_runs(self):
        knots = OnionKnots(-3.0, 3.0, nparam=20)
        coef = OnionCoef(knots)

        latent_params = jnp.zeros(knots.nparam)

        coef(latent_params)

    def test_compute_coef_with_optimal_start(self):
        knots = OnionKnots(-3.0, 3.0, nparam=20)
        coef = OnionCoef(knots)

        latent_params = jnp.full((knots.nparam,), fill_value=coef.target_log_increment)

        coef_out = coef(latent_params)
        assert jnp.allclose(coef_out, coef.target_coef, atol=1e-5)

    def test_broadcasting(self):
        knots = OnionKnots(-3.0, 3.0, nparam=20)
        coef = OnionCoef(knots)

        latent_params = jnp.full((knots.nparam,), fill_value=coef.target_log_increment)
        latent_params = jnp.stack((latent_params, jnp.zeros(knots.nparam)))

        coef_out = coef(latent_params)
        assert jnp.allclose(coef_out[0, :], coef.target_coef, atol=1e-5)
        assert jnp.allclose(coef_out[1, :], coef.target_coef, atol=1e-5)

    def test_jit(self):
        knots = OnionKnots(-3.0, 3.0, nparam=20)
        coef = OnionCoef(knots)

        latent_params = jnp.full((knots.nparam,), fill_value=coef.target_log_increment)
        latent_params = jnp.stack((latent_params, jnp.zeros(knots.nparam)))

        coef_out = jax.jit(coef)(latent_params)
        assert jnp.allclose(coef_out[0, :], coef.target_coef, atol=1e-5)
        assert jnp.allclose(coef_out[1, :], coef.target_coef, atol=1e-5)
