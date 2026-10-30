"""Contracts relied on when delegating inference to Liesel."""

import jax
import jax.numpy as jnp
import liesel.goose as gs
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.flatten_util import ravel_pytree

import liesel_ptm as ptm
from demo import reference_legacy


def test_gaussian_iwls_geometry_still_targets_the_ptm_posterior():
    model = reference_legacy.build_model()
    shape = model.vars["delta"].intercept.coef
    model.state = model.update_state(
        {shape.name: jnp.linspace(-0.6, 0.6, shape.value.size)}
    )
    term = model.vars["loc_terms"].terms["m.x"]
    name = term.coef.name
    interface = gs.LieselInterface(model)
    flat, unravel = ravel_pytree(model.extract_position([name]))
    builder = ptm.LegacyPTM(model).setup_mcmc(
        loc_terms=model.vars["loc_terms"],
        log_scale_terms=model.vars["log_scale_terms"],
        shape=model.vars["delta"],
        seed=3,
        num_chains=1,
        iwls_kwargs={"initial_step_size": 0.7},
    )
    builder.add_posterior(1)
    builder.build()  # Attach the helper's actual target interface to its kernels.
    kernel = next(ker for ker in builder.kernels if name in ker.position_keys)
    assert isinstance(kernel, gs.IWLSKernel)
    chol_info_fn = kernel.chol_info_fn
    assert chol_info_fn is not None
    logprob = gs.FlatInterfaceLogProb(interface, model.state, [name])
    key = jax.random.key(3)
    proposal_key, _ = jax.random.split(key)

    def precision_at(value):
        state = interface.update_state(unravel(value), model.state)
        factor = jnp.asarray(chol_info_fn(state))
        return factor @ factor.T

    def proposal_mean(value):
        precision = precision_at(value)
        return value + 0.7**2 / 2 * jnp.linalg.solve(precision, logprob.grad(value))

    chol = jnp.asarray(chol_info_fn(model.state))
    proposed = proposal_mean(flat) + 0.7 * jnp.linalg.solve(
        chol.T, jax.random.normal(proposal_key, flat.shape)
    )

    def proposal_distribution(value):
        covariance_chol = jnp.linalg.cholesky(
            0.7**2 * jnp.linalg.inv(precision_at(value))
        )
        return tfd.MultivariateNormalTriL(proposal_mean(value), covariance_chol)

    forward = proposal_distribution(flat)
    backward = proposal_distribution(proposed)
    expected = jnp.minimum(
        1.0,
        jnp.exp(
            logprob(proposed)
            - logprob(flat)
            + backward.log_prob(flat)
            - forward.log_prob(proposed)
        ),
    )
    assert 0.0 < expected < 0.99  # Keep the target-density check nontrivial.
    epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
    outcome = jax.jit(kernel.transition)(
        key, kernel.init_state(key, model.state), model.state, epoch
    )
    np.testing.assert_allclose(outcome.info.acceptance_prob, expected, atol=2e-5)
    retained = interface.extract_position([name], outcome.model_state)[name]
    np.testing.assert_allclose(
        retained,
        unravel(proposed)[name] if outcome.info.position_moved else term.coef.value,
    )
