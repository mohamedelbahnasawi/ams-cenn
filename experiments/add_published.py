"""Add published baseline results from original papers.

These are used for models where we don't re-run experiments
(PatchTST, iTransformer, DLinear, NHITS, TiDE).

Usage:
    python experiments/add_published.py          # Add all known published results
    python experiments/add_published.py --list   # Show what's available
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.config import PUBLISHED_DIR, HORIZONS, DATASETS_ALL


def save_published(model: str, dataset: str, horizon: int,
                   mse: float, mae: float, citation: str,
                   params: int | None = None):
    """Save a single published result as JSON."""
    result = {
        "model": model,
        "dataset": dataset,
        "horizon": horizon,
        "seed": 0,  # Published results are aggregated
        "input_size": 512,  # Most papers use 512
        "mse": mse,
        "mae": mae,
        "params": params,
        "source": "published",
        "citation": citation,
    }
    filename = f"{model}__{dataset}__H{horizon}__published.json"
    out_path = PUBLISHED_DIR / filename
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Published results from original papers
# All use L=512 (or as reported), MSE/MAE on standard 7:1:2 splits
#
# Sources:
#   PatchTST: Nie et al. 2023 (Table 2, channel-independent, L=512)
#   iTransformer: Liu et al. 2024 (Table 1, L=512)
#   DLinear: Zeng et al. 2023 (Table 2, L=512)
#   NHITS: Challu et al. 2023 (Appendix, L=512)
#   TiDE: Das et al. 2023 (Table 1, L=512)
#
# NOTE: Add/update these numbers as you collect them from papers.
#       Run `python experiments/add_published.py` after editing.
# ---------------------------------------------------------------------------

_CITE = "Nie et al. ICLR 2023, Table 3 (PatchTST/64, lookback L=512)"
PUBLISHED_RESULTS = {
    # PatchTST/64 (= native lookback L=512, matching our protocol) transcribed from Nie et al.
    # ICLR'23 Table 3 (supervised multivariate). Format: (model, dataset, horizon): (mse, mae, cite)
    ("PatchTST", "ETTh1", 96):  (0.370, 0.400, _CITE),
    ("PatchTST", "ETTh1", 192): (0.413, 0.429, _CITE),
    ("PatchTST", "ETTh1", 336): (0.422, 0.440, _CITE),
    ("PatchTST", "ETTh1", 720): (0.447, 0.468, _CITE),
    ("PatchTST", "ETTh2", 96):  (0.274, 0.337, _CITE),
    ("PatchTST", "ETTh2", 192): (0.341, 0.382, _CITE),
    ("PatchTST", "ETTh2", 336): (0.329, 0.384, _CITE),
    ("PatchTST", "ETTh2", 720): (0.379, 0.422, _CITE),
    ("PatchTST", "ETTm1", 96):  (0.293, 0.346, _CITE),
    ("PatchTST", "ETTm1", 192): (0.333, 0.370, _CITE),
    ("PatchTST", "ETTm1", 336): (0.369, 0.392, _CITE),
    ("PatchTST", "ETTm1", 720): (0.416, 0.420, _CITE),
    ("PatchTST", "ETTm2", 96):  (0.166, 0.256, _CITE),
    ("PatchTST", "ETTm2", 192): (0.223, 0.296, _CITE),
    ("PatchTST", "ETTm2", 336): (0.274, 0.329, _CITE),
    ("PatchTST", "ETTm2", 720): (0.362, 0.385, _CITE),
    ("PatchTST", "Weather", 96):  (0.149, 0.198, _CITE),
    ("PatchTST", "Weather", 192): (0.194, 0.241, _CITE),
    ("PatchTST", "Weather", 336): (0.245, 0.282, _CITE),
    ("PatchTST", "Weather", 720): (0.314, 0.334, _CITE),
    ("PatchTST", "Electricity", 96):  (0.129, 0.222, _CITE),
    ("PatchTST", "Electricity", 192): (0.147, 0.240, _CITE),
    ("PatchTST", "Electricity", 336): (0.163, 0.259, _CITE),
    ("PatchTST", "Electricity", 720): (0.197, 0.290, _CITE),
    ("PatchTST", "Traffic", 96):  (0.360, 0.249, _CITE),
    ("PatchTST", "Traffic", 192): (0.379, 0.256, _CITE),
    ("PatchTST", "Traffic", 336): (0.392, 0.264, _CITE),
    ("PatchTST", "Traffic", 720): (0.432, 0.286, _CITE),
}

# Placeholder — this dict will be populated by you or students
# by reading the original papers and adding entries.
# Each entry is: (model, dataset, horizon) -> (mse, mae, citation)


def add_all_published():
    """Save all known published results to experiments/published/."""
    count = 0
    for (model, dataset, horizon), (mse, mae, citation) in PUBLISHED_RESULTS.items():
        save_published(model, dataset, horizon, mse, mae, citation)
        count += 1
    print(f"Saved {count} published results to {PUBLISHED_DIR}")


def show_coverage():
    """Show which published results are available."""
    existing = list(PUBLISHED_DIR.glob("*.json"))
    print(f"\nPublished results available: {len(existing)}")

    if existing:
        for f in sorted(existing):
            print(f"  {f.stem}")

    # Show what's defined but not yet saved
    print(f"\nDefined in PUBLISHED_RESULTS dict: {len(PUBLISHED_RESULTS)} entries")

    # Show gaps
    models = ["PatchTST", "iTransformer", "DLinear", "NHITS", "TiDE"]
    print(f"\nCoverage (defined entries):")
    for model in models:
        for ds in DATASETS_ALL:
            entries = [(m, d, h) for (m, d, h) in PUBLISHED_RESULTS
                       if m == model and d == ds]
            if entries:
                horizons = [h for (_, _, h) in entries]
                print(f"  {model:>15} | {ds:<12} | H={horizons}")


def main():
    parser = argparse.ArgumentParser(description="Manage published baseline results")
    parser.add_argument("--list", action="store_true", help="Show coverage")
    args = parser.parse_args()

    if args.list:
        show_coverage()
    else:
        add_all_published()
        show_coverage()


if __name__ == "__main__":
    main()
