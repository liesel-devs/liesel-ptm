import logging
from itertools import chain

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl

from . import nodes as nd
from .custom_types import Array
from .dist import LocScaleTransformationDist
from .ptm_ls import LocationIntercept, ScaleInterceptExp, _product, _sum

logger = logging.getLogger(__name__)


class OnionPTMLocScale:
    def __init__(
        self,
        y: Array,
        nparam: int,
        tau2: nd.TransformedVar,
        a: float = -4.0,
        b: float = 4.0,
    ) -> None:
        self.tau2 = tau2
        self.knots = nd.OnionKnots(a=a, b=b, nparam=nparam, order=3)
        self.coef = nd.OnionCoefParam(knots=self.knots, tau2=tau2, name="shape_coef")

        self.loc_model: nd.Predictor = nd.Predictor("loc_model").update()
        """Predictor for the location model part. Does not include an intercept."""
        self.log_scale_model: nd.Predictor = nd.Predictor("log_scale_model").update()
        """Predictor for the log scale model part. Does not include an intercept."""

        self.scale_model = lsl.Calc(jnp.exp, self.log_scale_model).update()
        """The exponential of :attr:`.log_scale_model`."""

        self.loc_intercept = lsl.Var(
            LocationIntercept(y, self.loc_model, self.scale_model).update(),
            name="loc_intercept",
        )
        """Intercept :math:`\\beta_0` of the location model part."""

        self.scale_intercept_exp = lsl.Var(
            ScaleInterceptExp(
                y, self.loc_intercept, self.loc_model, self.scale_model
            ).update(),
            name="scale_intercept_exp",
        )
        r"""
        Exponential of the intercept :math:`\exp(\gamma_0)` of the scale model part.
        """

        self.loc = lsl.Var(
            lsl.Calc(_sum, self.loc_model, self.loc_intercept).update(), name="loc"
        )
        """The response's location, including an intercept."""

        self.scale = lsl.Var(
            lsl.Calc(_product, self.scale_model, self.scale_intercept_exp).update(),
            name="scale",
        )
        """The response's scale, including an intercept."""

        response_dist = lsl.Dist(
            LocScaleTransformationDist,
            knots=self.knots.knots,
            coef=self.coef,
            loc=self.loc,
            scale=self.scale,
        )

        self.response = lsl.obs(y, response_dist, name="response")

        self.intercept_names = [self.loc_intercept.name, self.scale_intercept_exp.name]
        """Names of the intercept nodes."""

    def build_graph(self) -> lsl.Model:
        gb = lsl.GraphBuilder().add(self.response)
        graph = gb.build_model()
        return graph

    def optimize_start_values(
        self, graph: lsl.Model, max_iter: int = 1000, patience: int = 10
    ) -> None:
        params = []  # type: ignore

        params = self.loc_model.parameters + self.log_scale_model.parameters

        logger.info("Optimizing location and scale parameters.")
        stopper = gs.Stopper(max_iter=max_iter, patience=patience)
        result = gs.optim_flat(model_train=graph, params=params, stopper=stopper)
        logger.info(
            "Optimizing location and scale parameters finished after"
            f" {result.iteration} iterations."
        )
        graph.state = result.model_state

        params = [self.coef.log_increments.transformed.name]  # type: ignore

        logger.info("Optimizing transformation parameters.")
        result = gs.optim_flat(model_train=graph, params=params, stopper=stopper)
        logger.info(
            "Optimizing transformation parameters finished after"
            f" {result.iteration} iterations."
        )
        graph.state = result.model_state

        return graph

    def setup_engine_builder(
        self, eb: gs.EngineBuilder, graph: lsl.Model, sample_transformation: bool = True
    ) -> gs.EngineBuilder:
        eb.set_model(gs.LieselInterface(graph))
        eb.set_initial_values(graph.state)

        loc_terms = self.loc_model.terms.values()
        scale_terms = self.log_scale_model.terms.values()

        for term in chain(loc_terms, scale_terms):
            for kernel in term.mcmc_kernels:
                eb.add_kernel(kernel)

        if sample_transformation:
            eb.add_kernel(gs.NUTSKernel([self.coef.log_increments.transformed.name]))
            eb.add_kernel(gs.NUTSKernel([nd.find_param(self.tau2).name]))  # type: ignore

        eb.positions_included = self.intercept_names

        return eb
