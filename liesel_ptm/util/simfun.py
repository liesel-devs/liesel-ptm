from typing import Any

import numpy as np
from scipy import stats

Array = Any


class SimFunctions:
    @staticmethod
    def f1_linear(x: Array, b: float = 1.0) -> Array:
        """Linear function."""
        return b * x

    @staticmethod
    def f2_ushaped(x: Array) -> Array:
        """
        Roughly u-shaped function with an overall increasing trend.
        """
        return x + ((2 * x) ** 2) / 5.5

    @staticmethod
    def f3_oscillating(x: Array) -> Array:
        """Oscillating function with an overall decreasing trend."""
        return -x + np.pi * np.sin(np.pi * x)

    @staticmethod
    def f4_bell(x: Array) -> Array:
        """
        Function that is based on the normal PDF, but a bit twisted.
        """
        return 0.5 * x + 15 * stats.norm.pdf(2 * (x - 0.2)) - stats.norm.pdf(x + 0.4)
