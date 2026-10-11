"""Capacity/schedule sweep for the CYP net, one process per config.

The owner asked for more epochs and larger models. Worth testing rather than
assuming, because on this programme's other tasks capacity has never been the
constraint — H4's regularisation sweep moved test score by 0.0004 and every
feature arm sat inside its seed spread. But that was TDC ADMET, not this task,
and the previous CYP runs stopped at epoch ~35 with patience 40, which means
they were **early-stopped, not epoch-limited**: more epochs alone would change
nothing without also changing capacity or the schedule.

So each config varies something that could plausibly move the stopping point:
width, depth, dropout, ensemble size, and a cosine schedule over a much longer
horizon (1200 epochs, patience 150).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

CONFIGS = [
    # tag,            d_model, layers, dropout, members
    ("base",             1024, 3, 0.15, 32),
    ("wide",             2048, 3, 0.15, 32),
    ("deep",             1024, 5, 0.15, 32),
    ("wide-deep",        2048, 5, 0.20, 24),
    ("low-drop",         1024, 3, 0.05, 32),
    ("high-drop",        1024, 3, 0.30, 32),
    ("narrow",            512, 3, 0.10, 48),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/out/cyp_net")
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--summary", default="/content/out/cyp/sweep.json")
    args = ap.parse_args()

    for tag, d, l, dr, mem in CONFIGS:
        for seed in range(args.seeds):
            done = Path(args.out_dir) / f"{tag}_s{seed}.json"
            if done.exists():
                print(f"[sweep] {tag} s{seed}: cached", flush=True); continue
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(seed), "--d-model", str(d),
                 "--n-layers", str(l), "--dropout", str(dr), "--members", str(mem),
                 "--epochs", str(args.epochs), "--patience", str(args.patience),
                 "--out-dir", args.out_dir],
                cwd="/content", env={**os.environ, "PYTHONPATH": "/content"},
                capture_output=True, text=True)
            tail = [x for x in r.stdout.splitlines() if x.startswith("[net] " + tag)]
            if tail:
                print(tail[-1], flush=True)
            else:
                err = r.stderr.strip().splitlines()
                print(f"[sweep] {tag} s{seed} FAILED rc={r.returncode}: "
                      f"{err[-1][:160] if err else 'no stderr'}", flush=True)

    rows = []
    for f in sorted(Path(args.out_dir).glob("*.json")):
        rows.append(json.loads(f.read_text()))
    rows.sort(key=lambda r: r["valid"])
    print(f"\n{'tag':12s} {'valid':>8s} {'ep':>6s} {'epochs':>7s} {'min':>6s} {'GB':>6s}")
    for r in rows:
        print(f"{r['tag']:12s} {r['valid']:8.4f} {r['best_epoch']:6d} {r['epochs_run']:7d} "
              f"{r['minutes']:6.1f} {r['peak_gb']:6.1f}")
    Path(args.summary).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
