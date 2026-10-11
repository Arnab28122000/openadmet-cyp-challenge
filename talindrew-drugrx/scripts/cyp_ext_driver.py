"""Run every (seed, arm) in its own process, then aggregate."""
from __future__ import annotations
import argparse, json, subprocess, sys, os
from pathlib import Path
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="/content/out/cyp/ext_runs")
    ap.add_argument("--summary", default="/content/out/cyp/external_result.json")
    args = ap.parse_args()

    for seed in range(args.seeds):
        for arm in ("without", "with"):
            tgt = Path(args.out_dir) / f"{arm}_{seed}.json"
            if tgt.exists():
                print(f"[drv] {arm} seed {seed}: cached", flush=True)
                continue
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_ext_one.py",
                 "--seed", str(seed), "--arm", arm, "--out-dir", args.out_dir],
                cwd="/content", env={**os.environ, "PYTHONPATH": "/content"},
                capture_output=True, text=True)
            tail = [l for l in r.stdout.splitlines() if l.startswith("[one]")]
            print(tail[-1] if tail else f"[drv] {arm} seed {seed} FAILED rc={r.returncode}: "
                  + r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else "", flush=True)

    rows = {"with": [], "without": []}
    for f in sorted(Path(args.out_dir).glob("*.json")):
        d = json.loads(f.read_text())
        rows[d["arm"]].append(d["macro_st_rae"])
    a, b = np.array(rows["without"]), np.array(rows["with"])
    n = min(len(a), len(b))
    print(f"\nwithout external: {a.mean():.4f} +/- {a.std():.4f}  (n={len(a)})")
    print(f"with external:    {b.mean():.4f} +/- {b.std():.4f}  (n={len(b)})")
    if n:
        paired = a[:n] - b[:n]
        print(f"paired gain {paired.mean():+.4f} +/- {paired.std():.4f} "
              f"({int((paired>0).sum())}/{n} seeds better with external)")
    Path(args.summary).write_text(json.dumps(
        {"without": a.tolist(), "with": b.tolist(),
         "paired_gain_mean": float((a[:n]-b[:n]).mean()) if n else None,
         "paired_gain_sd": float((a[:n]-b[:n]).std()) if n else None,
         "seeds_better": int(((a[:n]-b[:n])>0).sum()) if n else 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
