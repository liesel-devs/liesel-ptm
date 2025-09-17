import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd


def mock_samples(graph, shape, seed: int = 0):
    pos = graph.extract_position(list(graph.parameters))
    k = jax.random.key(seed)
    mock_samples = {}
    for name, value in pos.items():
        k, subkey = jax.random.split(k)
        try:
            dist = graph.vars[name].dist_node.init_dist()
        except AttributeError:
            dist = tfd.Normal(loc=jnp.zeros_like(value), scale=1.0)
        mock_samples[name] = dist.sample(shape, subkey)

    return mock_samples
