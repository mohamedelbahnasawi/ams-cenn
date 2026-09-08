"""Regenerate the per-run figure artifacts (predictions, gate values, operator norms) for an
already-trained variant WITHOUT retraining: load the canonical checkpoint, run the clean stride-1
test evaluation (exactly as the robustness harness does), then call the runner's own artifact savers.

Why: figure inputs (per-window predictions, gate values, operator norms) are large and are not
saved by default; this recomputes them from the saved checkpoint. Training is deterministic
(verified < 1e-8 between repeats), so retraining would only reproduce the same checkpoints.

Usage:
  python -m experiments.analysis.regen_artifacts --model CeNN_AMS-Anc \
      --cells ETTh1:96 ETTh1:720 ETTm2:96 ETTm2:720 --seeds 1,42,123,7,2026 --what pred,tau,aeff
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import SPLITS, INPUT_SIZE, ARTIFACTS_DIR, CENN_MAIN_VARIANT   # noqa: E402
from experiments.runner import load_dataset, _save_artifacts, _save_tau, _save_aeff  # noqa: E402
from experiments.run_robustness import _cv                                                    # noqa: E402
from neuralforecast import NeuralForecast                                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"CeNN_{CENN_MAIN_VARIANT}")
    ap.add_argument("--cells", nargs="+", required=True, help="DATASET:H pairs")
    ap.add_argument("--seeds", default="1,42,123,7,2026")
    ap.add_argument("--what", default="pred,tau,aeff")
    ap.add_argument("--ckpt-root", default="experiments/checkpoints")
    a = ap.parse_args()
    what = set(a.what.split(","))
    for cell in a.cells:
        ds, h = cell.split(":"); h = int(h)
        val_size, test_size = SPLITS[ds]
        Y = load_dataset(ds)
        for seed in (int(s) for s in a.seeds.split(",")):
            ckpt = Path(a.ckpt_root) / f"{a.model}__{ds}__H{h}__seed{seed}"
            if not ckpt.exists():
                print(f"[missing] {ckpt.name}", flush=True); continue
            t0 = time.time()
            nf = NeuralForecast.load(str(ckpt))
            for attr in ("dataset", "uids", "last_dates", "ds"):
                if not hasattr(nf, attr):
                    setattr(nf, attr, None)
            alias = nf.models[0].alias
            cv = _cv(nf, Y, val_size, test_size)          # clean pass; also sets last_tau in the cells
            done = []
            if "pred" in what:
                _save_artifacts(nf, a.model, ds, h, seed, cv, alias, force_any=True); done.append("pred")
            if "tau" in what:
                _save_tau(nf, a.model, ds, h, seed); done.append("tau")
            if "aeff" in what:
                _save_aeff(nf, a.model, ds, h, seed); done.append("aeff")
            print(f"{a.model} {ds} H{h} seed{seed}: {','.join(done)} ({time.time()-t0:.0f}s)", flush=True)
    print("artifacts under", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
