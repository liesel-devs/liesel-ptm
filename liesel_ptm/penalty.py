from typing import Any

import jax.numpy as jnp

Array = Any


class Penalty:
    @staticmethod
    def pspline(nparam: int, random_walk_order: int = 2) -> Array:
        """
        Builds an (nparam x nparam) P-spline penalty matrix.
        """
        D = jnp.diff(jnp.identity(nparam), random_walk_order, axis=0)
        return D.T @ D

    @staticmethod
    def pspline_normed(nparam: int, random_walk_order: int = 2) -> tuple[Array, Array]:
        """
        Builds an (nparam x nparam) P-spline penalty matrix.
        """
        D = jnp.diff(jnp.identity(nparam), random_walk_order, axis=0)
        K = D.T @ D
        scale = jnp.linalg.norm(K, ord=jnp.inf)
        return K / scale, scale
