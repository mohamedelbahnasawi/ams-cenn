import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cenn_forecasting import AMSCeNNForecaster, CeNNModel1D, ams_cenn, count_parameters, make_windows


def test_parameter_count_matches_paper_profile():
    # ETTh1 profile of the paper: seven series, lookback 512, horizon 96.
    assert count_parameters(ams_cenn(7, 512, 96)) == 106_197


def test_forward_shape_and_finiteness():
    m = ams_cenn(3, 64, 24).eval()
    x = torch.randn(5, 64, 3)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (5, 24, 3)
    assert torch.isfinite(y).all()


def test_last_value_normalization_is_level_equivariant():
    # LVN subtracts each channel's last observed value and adds it back: shifting a channel by a constant
    # shifts its forecast by the same constant.
    torch.manual_seed(0)
    m = ams_cenn(2, 48, 12).eval()
    x = torch.randn(4, 48, 2)
    c = torch.tensor([3.0, -2.0])
    with torch.no_grad():
        y0 = m(x)
        y1 = m(x + c)
    assert torch.allclose(y1 - c, y0, atol=1e-4)


def test_without_lvn_option_builds():
    m = ams_cenn(4, 64, 64, revin=False)
    assert isinstance(m, CeNNModel1D)
    assert count_parameters(m) > 0


def test_windows():
    X, Y = make_windows(np.arange(20, dtype=np.float32)[:, None], 5, 3)
    assert X.shape == (13, 5, 1) and Y.shape == (13, 3, 1)
    assert X[0, -1, 0] == 4 and Y[0, 0, 0] == 5


def test_forecaster_fit_predict_smoke():
    t = np.arange(600, dtype=np.float32)
    y = np.stack([np.sin(t / 10), np.cos(t / 7)], axis=1)
    f = AMSCeNNForecaster(input_size=32, h=8, max_steps=30, batch_size=64, val_check_steps=10, device="cpu")
    f.fit(y)
    out = f.predict(y)
    assert out.shape == (8, 2) and np.isfinite(out).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "neuralforecast" / "cenn" / "model.py").exists(),
    reason="fork core not available next to the package",
)
def test_matches_fork_core_bit_for_bit():
    # The packaged core is the fork's vendored core: same weights must give the same forecast.
    # Load the fork's core module file directly (it only depends on torch), bypassing the package __init__.
    core_file = Path(__file__).resolve().parents[2] / "neuralforecast" / "cenn" / "model.py"
    spec = importlib.util.spec_from_file_location("fork_cenn_core", core_file)
    fork = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fork)
    torch.manual_seed(1)
    ours = ams_cenn(5, 96, 24).eval()
    theirs = fork.CeNNModel1D(n_features=5, seq_length=96, pred_length=24, **__import__("cenn_forecasting").AMS_CENN_CONFIG).eval()
    theirs.load_state_dict(ours.state_dict())
    x = torch.randn(3, 96, 5)
    with torch.no_grad():
        assert torch.equal(ours(x), theirs(x))
