import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

Array = Any

logger = logging.getLogger(__name__)


@dataclass
class SwapSpec:
    from_dist: type[tfd.Distribution] | None = None
    from_name: str | None = None
    from_role: str | None = None
    to_dist: type[tfd.Distribution] | None = None
    to_dist_kwargs: dict[str, Array] | None = None
    bijector: tfb.Bijector | None = None


weibull_to_bounded_gamma = SwapSpec(
    from_dist=tfd.Weibull,
    to_dist=tfd.Gamma,
    to_dist_kwargs={"concentration": 2.0, "rate": 0.5},
    bijector=tfb.SoftClip(low=0.0025),
)

ig_to_bounded_gamma = SwapSpec(
    from_dist=tfd.InverseGamma,
    to_dist=tfd.Gamma,
    to_dist_kwargs={"concentration": 2.0, "rate": 0.5},
    bijector=tfb.SoftClip(low=0.0025),
)


def spec_lower_bound(
    var: lsl.Var,
    lower_bound: float | Array = 0.05**2,
    bijector: tfb.Bijector | None = None,
):
    lower_bound = jnp.asarray(lower_bound, jnp.asarray(var.value).dtype)

    if var.strong:
        spec = SwapSpec(from_name=var.name, bijector=tfb.SoftClip(low=lower_bound))
        return spec

    # if var is weak
    fn = var.value_node.function  # type: ignore
    try:
        bijector = fn.__self__
    except AttributeError as e:
        msg1 = f"{var=} is weak. Tried to obtain a bijector from its value function, "
        msg2 = "but this failed, because the value function does not seem to be a "
        msg3 = "bound method."
        raise AttributeError(msg1 + msg2 + msg3) from e

    low = bijector.inverse(lower_bound)
    parent_var = var.value_node[0]
    if parent_var.weak:  # type: ignore
        raise ValueError(
            f"{var} is weak, and parent variable {parent_var} is weak, too."
        )

    spec = SwapSpec(from_name=parent_var.name, bijector=tfb.SoftClip(low=low))
    return spec


def _guess_bijector(dist: tfd.Distribution):
    """
    Guesses the bijector being used.

    If the dist's domain is strictly positive, then the guess is that no bijector
    is being used.

    If the dist's domain includes negative values, then the guess that the exponential
    bijector is being used.
    """
    p = 0.001
    small = dist.quantile(p)
    i = 0
    while jnp.isnan(small):
        p += 0.001
        small = dist.quantile(p)
        i += 1

        if i >= 200:
            raise RuntimeError("Failed to guess bijector.")

    if small > 0.0:
        return tfb.Identity()
    elif small < 0.0:
        return tfb.Exp()


def spec_bounded(
    var: lsl.Var,
    lower_bound: float | Array = 0.05**2,
    upper_bound: float | Array = 10_000.0,
    bijector: tfb.Bijector | None = None,
):
    lower_bound = jnp.asarray(lower_bound, jnp.asarray(var.value).dtype)
    upper_bound = jnp.asarray(upper_bound, jnp.asarray(var.value).dtype)

    if var.strong and var.dist_node:
        bijector = _guess_bijector(var.dist_node.init_dist())  # type: ignore
        name = var.name

        # for example, if the bijector is tfb.Exp(), this will be the log
        fn = bijector.inverse
    elif var.strong:
        raise NotImplementedError
    else:
        # if var is weak, it is a function of another variable
        fn = var.value_node.function  # type: ignore
        try:
            bijector = fn.__self__
        except AttributeError as e:
            msg1 = (
                f"{var=} is weak. Tried to obtain a bijector from its value function, "
            )
            msg2 = "but this failed, because the value function does not seem to be a "
            msg3 = "bound method."
            raise AttributeError(msg1 + msg2 + msg3) from e

        parent_var = var.value_node[0]
        if parent_var.weak:  # type: ignore
            raise ValueError(
                f"{var} is weak, and parent variable {parent_var} is weak, too."
            )

        name = parent_var.name
        fn = bijector.inverse

    low = fn(lower_bound)
    high = fn(upper_bound)

    spec = SwapSpec(
        from_name=name,
        to_dist=tfd.Uniform,
        to_dist_kwargs={"low": low, "high": high},
        bijector=tfb.SoftClip(low=low, high=high),
    )
    return spec


class TemporarilySwapDists:
    def __init__(
        self, model: lsl.Model, pairs: Sequence[SwapSpec], to_float32: bool = True
    ):
        self.model = model
        self.to_float32 = to_float32
        self.model_copy = None
        self.pairs = pairs

        self.old_dists: dict[str, lsl.Dist] = {}
        self.old_values: dict[str, lsl.Node] = {}
        self.position: dict[str, Array] = {}

        self.names_to_replace = [pair.from_name for pair in self.pairs]
        self.roles_to_replace = [pair.from_role for pair in self.pairs]
        self.dists_to_replace = [pair.from_dist for pair in self.pairs]
        self.replacements = [pair.to_dist for pair in self.pairs]

        names_str = [n for n in self.names_to_replace if isinstance(n, str)]
        roles_str = [n for n in self.roles_to_replace if isinstance(n, str)]

        if len(set(names_str)) < len(names_str):
            raise ValueError("Names must be unique")

        if len(set(roles_str)) < len(roles_str):
            raise ValueError("Roles must be unique")

    def __enter__(self):
        _, vars_ = self.model.copy_nodes_and_vars()

        for name, var in vars_.items():
            if var.dist_node is None:
                continue
            dist_found = var.dist_node.distribution in self.dists_to_replace
            name_found = var.name in self.names_to_replace
            role_found = var.role in self.roles_to_replace
            if dist_found or name_found or role_found:
                self.old_dists[name] = var.dist_node
                self.old_values[name] = var.value_node

                if dist_found:
                    i = self.dists_to_replace.index(var.dist_node.distribution)
                elif name_found:
                    i = self.names_to_replace.index(var.name)
                elif role_found:
                    i = self.roles_to_replace.index(var.role)

                replacement_dist = self.pairs[i].to_dist
                replacement_kwargs = self.pairs[i].to_dist_kwargs
                bijector = self.pairs[i].bijector

                if replacement_dist is not None:
                    uid = str(uuid4())
                    kwargs = {
                        name: lsl.Value(value, _name=f"__{name}-{uid}__")
                        for name, value in replacement_kwargs.items()
                    }
                    temp_dist_node = lsl.Dist(replacement_dist, **kwargs)
                    var.dist_node = temp_dist_node
                if bijector is not None:
                    transformed = var.transform(bijector, inference="drop")
                    if not transformed.role:
                        transformed.role = var.role

                else:
                    var.dist_node = None

        self.model_copy = lsl.Model(list(vars_.values()), to_float32=self.to_float32)

        return self.model_copy

    def __exit__(self, exc_type, exc_value, traceback):
        _, vars_ = self.model_copy.pop_nodes_and_vars()

        for name, dist in self.old_dists.items():
            vars_[name].dist_node = dist

        model_copy = lsl.Model(list(vars_.values()), to_float32=self.to_float32)
        cleaned_state = {}
        for name, val in model_copy.state.items():
            if name in self.model.state:
                cleaned_state[name] = val
        self.model.state = cleaned_state

        return False  # Returning False means exceptions are not suppressed
