"""cenn-forecasting: cellular neural networks for multivariate time-series forecasting.

Version 1.0.0 ships AMS-CeNN, the auditable, stability-certified cellular residual network of the
accompanying paper, as the main model (``ams_cenn``, ``AMSCeNNForecaster``, ``CeNNModel1D`` with the C1/C2/C3
options). The conference-era forecaster of the 0.x releases is kept unchanged in ``cenn_forecasting.legacy``.
"""
from .model import (
    AdaptiveTauGate,
    CeNNCell1D,
    CeNNLayer1D,
    CeNNBlock1D,
    CeNNStack1D,
    CeNNModel1D,
    ResidualNorm1D,
)
from .ams import AMS_CENN_CONFIG, AMSCeNNForecaster, ams_cenn, count_parameters, make_windows

__version__ = "1.0.0"
__all__ = [
    "AdaptiveTauGate",
    "CeNNCell1D",
    "CeNNLayer1D",
    "CeNNBlock1D",
    "CeNNStack1D",
    "CeNNModel1D",
    "ResidualNorm1D",
    "AMS_CENN_CONFIG",
    "AMSCeNNForecaster",
    "ams_cenn",
    "count_parameters",
    "make_windows",
]
