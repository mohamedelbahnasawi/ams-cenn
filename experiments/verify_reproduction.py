"""Clean-environment reproduction check for the paper's tables (REQ-11).

Regenerates the aggregates, the significance test, and every manuscript-facing table from the
per-seed result JSONs tracked in this repository, then compares the outputs with the committed
copies under experiments/expected/. Run by the `reproduce-tables` CI job on every push; run
locally with

    python experiments/verify_reproduction.py            # regenerate + compare
    python experiments/verify_reproduction.py --update   # refresh experiments/expected/

Comparison ignores generator comment lines (they carry timestamps) and compares the significance
statistics numerically (relative tolerance 1e-6) because BLAS/platform differences can change the
last printed digit of a p-value without changing any table entry.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
AGG = EXP / "aggregated"
EXPECTED = EXP / "expected"
TABLES = ["table_main_results_anchored.tex", "table_main_results_std_anchored.tex", "table_profile_anchored.tex",
          "table_regret28_anchored.tex", "table_generic_trunk_anchored.tex", "table_scaler_ams_anchored.tex",
          "table_crosschan_anchored.tex"]
PY = sys.executable


def run(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1")
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=str(ROOT), env=env, check=True)


def regenerate():
    run(str(EXP / "aggregate.py"), "--status")
    run(str(EXP / "aggregate.py"), "--latex")
    run(str(EXP / "make_tables.py"))
    run(str(EXP / "analysis" / "make_paper_tables.py"))


def norm_tex(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%")).strip()


def sig_view(d: dict) -> dict:
    """The parts of significance.json a reader relies on: ranks, statistic, p-values, CD, pairwise p."""
    out = {"mean_ranks": d.get("mean_ranks"), "friedman_stat": d.get("friedman_stat"), "friedman_p": d.get("friedman_p"),
           "critical_difference": d.get("critical_difference"), "n_complete_blocks": d.get("n_complete_blocks"),
           "k_methods": d.get("k_methods"), "methods": d.get("methods")}
    for key in ("nemenyi_p", "pairwise_p", "nemenyi_vs_main"):
        if key in d:
            out[key] = d[key]
    return out


def close(a, b, rtol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], rtol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))
    return a == b


def main():
    update = "--update" in sys.argv
    regenerate()
    produced = {name: AGG / "paper_tables" / name for name in TABLES}
    produced["significance.json"] = AGG / "tables" / "significance.json"
    if update:
        EXPECTED.mkdir(exist_ok=True)
        for name, path in produced.items():
            if name.endswith(".json"):
                (EXPECTED / name).write_text(json.dumps(sig_view(json.loads(path.read_text(encoding="utf-8"))), indent=2), encoding="utf-8")
            else:
                (EXPECTED / name).write_text(norm_tex(path.read_text(encoding="utf-8")) + "\n", encoding="utf-8")
        print("expected outputs refreshed under", EXPECTED)
        return 0
    failures = []
    for name, path in produced.items():
        exp = EXPECTED / name
        if not exp.exists():
            failures.append(f"{name}: no expected copy (run with --update)"); continue
        if name.endswith(".json"):
            got = sig_view(json.loads(path.read_text(encoding="utf-8")))
            want = json.loads(exp.read_text(encoding="utf-8"))
            ok = close(got, want)
        else:
            ok = norm_tex(path.read_text(encoding="utf-8")) == norm_tex(exp.read_text(encoding="utf-8"))
        print(f"{'OK  ' if ok else 'DIFF'} {name}")
        if not ok:
            failures.append(name)
    if failures:
        print("\nREPRODUCTION FAILED for:", ", ".join(failures))
        return 1
    print("\nAll manuscript tables and the significance statistics reproduce from the tracked result files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
