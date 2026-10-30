from typing import Any

import jax.numpy as jnp

Array = Any


def mixed_model(penalty: Array, rank: Array | int | None = None) -> Array:
    if rank is None:
        rank = jnp.linalg.matrix_rank(penalty)
    evalues, evectors = jnp.linalg.eigh(penalty)
    evalues = evalues[::-1]  # put in decreasing order
    evectors = evectors[:, ::-1]  # make order correspond to eigenvalues
    rank = jnp.linalg.matrix_rank(penalty)

    if evectors[0, 0] < 0:
        evectors = -evectors

    U = evectors
    D = 1 / jnp.sqrt(jnp.ones_like(evalues).at[:rank].set(evalues[:rank]))
    Z = (U.T * jnp.expand_dims(D, 1)).T
    return Z


class LinearConstraintEVD:
    @staticmethod
    def general(constraint: Array) -> Array:
        A = constraint
        nconstraints, _ = A.shape

        AtA = A.T @ A
        evals, evecs = jnp.linalg.eigh(AtA)

        signs = jnp.sign(evecs[0, :])
        signs = jnp.where(signs == 0, 1.0, signs)
        evecs = evecs * signs

        rank = jnp.linalg.matrix_rank(AtA)
        Abar = evecs[:, :-rank].T

        A_stacked = jnp.r_[A, Abar]
        C_stacked = jnp.linalg.inv(A_stacked)
        Cbar = C_stacked[:, nconstraints:]
        return Cbar

    @classmethod
    def _nullspace(cls, penalty: Array, rank: float | Array | None = None) -> Array:
        if rank is None:
            rank = jnp.linalg.matrix_rank(penalty)
        evals, evecs = jnp.linalg.eigh(penalty)
        evals = evals[::-1]  # put in decreasing order
        evecs = evecs[:, ::-1]  # make order correspond to eigenvalues
        rank = jnp.sum(evals > 1e-6)

        if evecs[0, 0] < 0:
            evecs = -evecs

        U = evecs
        D = 1 / jnp.sqrt(jnp.ones_like(evals).at[:rank].set(evals[:rank]))
        Z = (U.T * jnp.expand_dims(D, 1)).T
        Abar = Z[:, :rank]

        return Abar

    @classmethod
    def constant_and_linear(cls, x: Array, basis: Array) -> Array:
        nobs = jnp.shape(x)[0]
        j = jnp.ones(shape=nobs)
        X = jnp.c_[j, x]
        A = jnp.linalg.inv(X.T @ X) @ X.T @ basis
        return cls.general(constraint=A)

    @classmethod
    def sumzero_coef(cls, ncoef: int) -> Array:
        j = jnp.ones(shape=(1, ncoef))
        return cls.general(constraint=j)

    @classmethod
    def sumzero_term(cls, basis: Array) -> Array:
        nobs = jnp.shape(basis)[0]
        j = jnp.ones(shape=nobs)
        A = jnp.expand_dims(j @ basis, 0)
        return cls.general(constraint=A)

    @classmethod
    def sumzero_term2(cls, basis: Array) -> Array:
        A = jnp.mean(basis, axis=0, keepdims=True)
        return cls.general(constraint=A)
