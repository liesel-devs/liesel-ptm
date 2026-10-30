import numpy as np
import pytest

from liesel_ptm.util.summary import grouped_summary


@pytest.mark.parametrize("axis", [(0, 1), [0, 1]])
def test_grouped_summary_accepts_axis_sequences(axis) -> None:
    values = np.arange(24.0).reshape(2, 3, 4)

    summary = grouped_summary(values, axis=axis)

    np.testing.assert_allclose(summary["mean"], [10.0, 11.0, 12.0, 13.0])
