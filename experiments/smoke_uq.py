#!/usr/bin/env python
"""UQ data-capture smoke gate: forward_branches + checkpoint round-trip + artifact format.

Guards the two UQ sources:
  - scale-disagreement: CeNNModel1D.forward_branches (per-scale forecasts) — eval-guarded.
  - seed-ensemble: runner._save_artifacts writes a self-describing .npz per seed.
Plus the checkpoint save/load round-trip (post-hoc UQ + tau analysis depend on it).

Run: .venv/Scripts/python.exe experiments/smoke_uq.py
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from experiments.runner import build_cenn, load_dataset, _save_artifacts
from experiments.config import ARTIFACTS_DIR


def main():
    results = []

    # 1) forward_branches: eval-guard, shape, non-multiscale raises, no_grad
    try:
        m, _, _ = build_cenn("CeNN-Full", 96, 7, 1, 5); inner = m.model
        x = torch.randn(2, 512, 7)
        inner.train()
        try:
            inner.forward_branches(x); ok_guard = False
        except RuntimeError:
            ok_guard = True
        inner.eval()
        br = inner.forward_branches(x)
        ok = (ok_guard and tuple(br.shape) == (4, 2, 96, 7) and not br.requires_grad)
        results.append((ok, "forward_branches", f"train-raises={ok_guard} shape={tuple(br.shape)} no_grad={not br.requires_grad}"))
    except Exception as e:
        results.append((False, "forward_branches", f"CRASH {type(e).__name__}: {e}"))

    # 2) non-multiscale variant must raise
    try:
        m2, _, _ = build_cenn("C1-BoundedTau", 96, 7, 1, 5); m2.model.eval()
        try:
            m2.model.forward_branches(torch.randn(1, 512, 7)); ok = False
        except ValueError:
            ok = True
        results.append((ok, "non-multiscale-raises", "C1 forward_branches raises ValueError" if ok else "did NOT raise"))
    except Exception as e:
        results.append((False, "non-multiscale-raises", f"CRASH {type(e).__name__}: {e}"))

    # 3) _save_artifacts writes a self-describing .npz (pred,y,unique_id,ds).
    # Must use the HEADLINE model name: _save_artifacts now no-ops for any other variant
    # (the size guard of commit 64a9d4c) — a fake name would silently write nothing.
    try:
        from experiments.config import CENN_MAIN_VARIANT
        hm = f"CeNN_{CENN_MAIN_VARIANT}"
        df = pd.DataFrame({"unique_id": ["a", "a"], "ds": [1, 2], hm: [0.1, 0.2], "y": [0.3, 0.4]})
        _save_artifacts(None, hm, "SMOKEDS", 96, 1, df, hm)
        p = ARTIFACTS_DIR / "predictions" / f"{hm}__SMOKEDS__H96__seed1.npz"
        with np.load(p, allow_pickle=True) as d:   # context-manager closes the handle (Windows)
            files = sorted(d.files)
            ok = {"pred", "y", "unique_id", "ds"}.issubset(set(files))
        results.append((ok, "artifact-npz", f"keys={files}"))
        try:
            p.unlink()  # cleanup the smoke artifact
        except OSError:
            pass
    except Exception as e:
        results.append((False, "artifact-npz", f"CRASH {type(e).__name__}: {e}"))

    # 4) checkpoint save -> load -> predict round-trip (CeNN registered in MODEL_FILENAME_DICT)
    try:
        Y = load_dataset("ETTh2"); Y = pd.concat([g.tail(900) for _, g in Y.groupby("unique_id")], ignore_index=True)
        mm, _, _ = build_cenn("CeNN-Full", 96, 7, 1, 5)
        nf = NeuralForecast(models=[mm], freq="h"); nf.fit(Y, val_size=96)
        d = tempfile.mkdtemp(); nf.save(path=d, save_dataset=False, overwrite=True)
        nf2 = NeuralForecast.load(path=d); pr = nf2.predict(df=Y)
        ok = nf2.models[0].__class__.__name__ == "CeNN" and len(pr) > 0
        results.append((ok, "checkpoint-roundtrip", f"reloaded={nf2.models[0].__class__.__name__} pred_rows={len(pr)}"))
    except Exception as e:
        results.append((False, "checkpoint-roundtrip", f"CRASH {type(e).__name__}: {e}"))

    nfail = 0
    for ok, name, msg in results:
        if not ok: nfail += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} {msg}")
    print(f"\n{len(results) - nfail}/{len(results)} checks PASS")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
