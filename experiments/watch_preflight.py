#!/usr/bin/env python
"""Live watcher for the preflight run. Reads experiments/preflight_status.json (rewritten
after every cell) and prints progress until done or Ctrl-C.

    python experiments/watch_preflight.py          # refresh every 5s
    python experiments/watch_preflight.py --once    # single snapshot
"""
import json, sys, time
from pathlib import Path

P = Path(__file__).resolve().parent / "preflight_status.json"

# TOTAL must track the actual preflight grid: a hardcoded count silently excluded
# variants added later (the two APP-MultiScale-* + K2-Euler), so the watcher reported PASS before
# they were ever validated. Derive it from the same grid() the preflight uses.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preflight import grid
    TOTAL = len(grid(quick=False))
except Exception:
    TOTAL = 0  # unknown; the watcher still shows the live count


def snapshot():
    if not P.exists():
        print("no preflight_status.json yet (preflight not started?)")
        return False
    try:
        d = json.load(open(P))
    except Exception:
        return False  # mid-write; try again next tick
    n = len(d)
    p = sum(v.get("status") == "PASS" for v in d.values())
    f = sum(v.get("status") == "FAIL" for v in d.values())
    last = list(d)[-1] if d else "-"
    print(f"{n:3d}/{TOTAL}  PASS={p}  FAIL={f}   last: {last}")
    for k, v in d.items():
        if v.get("status") == "FAIL":
            print(f"    FAIL {k}: {v.get('error', '')[:110]}")
    return TOTAL > 0 and n >= TOTAL and f == 0   # True == fully done & clean (TOTAL>0 guards import-fail)


def main():
    once = "--once" in sys.argv
    try:
        while True:
            done = snapshot()
            if once or done:
                if done:
                    print("preflight complete and clean.")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n(stopped watching; the preflight run is unaffected)")


if __name__ == "__main__":
    main()
