import logging

from .bsplines import BSplineApprox as BSplineApprox
from .bsplines import ExtrapBSplineApprox as ExtrapBSplineApprox
from .bsplines import avg_slope_bspline as avg_slope_bspline
from .bsplines import bspline_basis as bspline_basis
from .bsplines import bspline_basis_deriv as bspline_basis_deriv
from .bsplines import bspline_basis_deriv2 as bspline_basis_deriv2
from .bsplines import kn as kn
from .cache import cache as cache
from .datagen import PTMLocScaleDataGen as PTMLocScaleDataGen
from .datagen import TAMLocScaleDataGen as TAMLocScaleDataGen
from .datagen import example_data as example_data
from .datagen import sample_shape as sample_shape
from .nodes import BasisDot as BasisDot
from .nodes import BSplineBasis as BSplineBasis
from .nodes import ConstantPriorScalingFactor as ConstantPriorScalingFactor
from .nodes import Dot as Dot
from .nodes import ExpParam as ExpParam
from .nodes import Intercept as Intercept
from .nodes import LinearSmooth as LinearSmooth
from .nodes import LinearTerm as LinearTerm
from .nodes import MISpline as MISpline
from .nodes import NonlinearPSpline as NonlinearPSpline
from .nodes import Predictor as Predictor
from .nodes import PSpline as PSpline
from .nodes import RandomIntercept as RandomIntercept
from .nodes import RandomInterceptSumZero as RandomInterceptSumZero
from .nodes import S as S
from .nodes import ScaledBasisDot as ScaledBasisDot
from .nodes import ScaledDot as ScaledDot
from .nodes import ScaleHalfCauchy as ScaleHalfCauchy
from .nodes import ScaleInverseGamma as ScaleInverseGamma
from .nodes import ScaleWeibull as ScaleWeibull
from .nodes import StrAT as StrAT
from .nodes import StructuredAdditiveTerm as StructuredAdditiveTerm
from .nodes import SymmetricallyBoundedScalar as SymmetricallyBoundedScalar
from .nodes import TransformationDist as TransformationDist
from .nodes import TransformedVar as TransformedVar
from .nodes import TruncatedNormalOmega as TruncatedNormalOmega
from .nodes import VarHalfCauchy as VarHalfCauchy
from .nodes import VarHalfNormal as VarHalfNormal
from .nodes import VarInverseGamma as VarInverseGamma
from .nodes import VarWeibull as VarWeibull
from .nodes import bs as bs
from .nodes import cholesky_ltinv as cholesky_ltinv
from .nodes import diffpen as diffpen
from .nodes import model_matrix as model_matrix
from .nodes import normalization_coef as normalization_coef
from .nodes import nullspace_remover as nullspace_remover
from .nodes import sumzero_coef as sumzero_coef
from .nodes import sumzero_term as sumzero_term
from .optim import OptimResult as OptimResult
from .optim import Stopper as Stopper
from .optim import history_to_df as history_to_df
from .optim import optim_flat as optim_flat
from .ptm_ls import PTMLocScale as PTMLocScale
from .ptm_ls import PTMLocScalePredictions as PTMLocScalePredictions
from .ptm_ls import ShapePrior as ShapePrior
from .ptm_ls import state_to_samples as state_to_samples
from .sampling import cache_results as cache_results
from .sampling import get_log_lik_fn as get_log_lik_fn
from .sampling import get_log_prob_fn as get_log_prob_fn
from .sampling import kwargs_full as kwargs_full
from .sampling import kwargs_lin as kwargs_lin
from .sampling import kwargs_loc as kwargs_loc
from .sampling import kwargs_loc_lin as kwargs_loc_lin
from .sampling import kwargs_scale as kwargs_scale
from .sampling import kwargs_scale_lin as kwargs_scale_lin
from .sampling import optimize_parameters as optimize_parameters
from .sampling import sample_means as sample_means
from .sampling import sample_quantiles as sample_quantiles
from .sampling import summarise_by_quantiles as summarise_by_quantiles
from .sampling import summarise_by_samples as summarise_by_samples
from .tam import Normalization as Normalization
from .tam import ShapeParam as ShapeParam
from .tam import TAMLocScale as TAMLocScale
from .tam import normalization_fn as normalization_fn
from .var import Var as Var


def setup_logger() -> None:
    """
    Sets up a basic ``StreamHandler`` that prints log messages to the terminal.
    The default log level of the ``StreamHandler`` is set to "info".

    The global log level for Liesel can be adjusted like this::

        import logging
        logger = logging.getLogger("liesel")
        logger.level = logging.WARNING

    This will set the log level to "warning".
    """

    # We adjust only our library's logger
    logger = logging.getLogger("liesel_ptm")

    # This is the level that will in principle be handled by the logger.
    # If it is set, for example, to logging.WARNING, this logger will never
    # emit messages of a level below warning
    logger.setLevel(logging.DEBUG)

    # By setting this to False, we prevent the Liesel log messages from being passed on
    # to the root logger. This prevents duplication of the log messages
    logger.propagate = False

    # This is the default handler that we set for our log messages
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # We define the format of log messages for this handler
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)


setup_logger()
