"""Does training keep improving past epoch 100 — and does the peak keep moving?

Measured so far: base peaks at epoch ~19, deep at ~135, wide at ~200, and all
three land on the same score (depth p=0.723, width +0.0000). So capacity moves
the *optimum* without moving the *destination*.

Two open questions that a best-point summary cannot answer:

1. **Is the model still improving at epoch 100?** For base, no — it peaked 80
   epochs earlier and is overfitting. For deep and wide, yes. The full validation
   curve is now recorded so this is readable rather than inferred.
2. **Does the pattern continue?** If capacity only buys a longer climb, then a
   much larger model trained much longer should peak later still and arrive at
   the same place. If instead the score finally improves at some capacity, the
   data-bound conclusion needs qualifying.

These configs run to 3,000 epochs with patience 500 — far past any observed
optimum — so the curve shape is visible rather than truncated by early stopping.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/content")
# Both of these were used below before they were imported: the metric guard was
# added here on 2026-10-05 and `METRIC_ID` was never brought in, so the branch
# meant to stop a stale-metric comparison would have raised NameError instead --
# and only on the cached path, which is the path a *resumed* run takes.
from drugrx.cyp.metric import METRIC_ID  # noqa: E402
from drugrx.cyp.publish import push_result  # noqa: E402

# Deliberately matched at 16 members so capacity is the only variable; members
# buy decorrelation and would otherwise confound the comparison.
CONFIGS = [
    ("long-base",  1024, 3, 16),
    ("long-deep",  1024, 6, 16),
    ("long-wide",  2048, 4, 16),
    ("long-huge",  2048, 6, 16),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=500)
    ap.add_argument("--out-dir", default="/content/out/cyp_long")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    for tag, d, l, mem in CONFIGS:
        if args.only and tag not in args.only:
            continue
        for s in range(args.seeds):
            done = Path(args.out_dir) / f"{tag}_s{s}.json"
            if done.exists():
                # Existence is not enough: a JSON written under a different metric
                # must be re-run, not reused. Pooling across a metric change is how
                # five old-metric depth runs were averaged with three official ones.
                try:
                    prev = json.loads(done.read_text()).get("metric_id")
                except Exception:
                    prev = None
                if prev == METRIC_ID:
                    print(f"[long] {tag} s{s}: cached", flush=True); continue
                print(f"[long] {tag} s{s}: stale metric ({prev}) -- re-running",
                      flush=True)
                done.unlink()
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(s), "--d-model", str(d),
                 "--n-layers", str(l), "--members", str(mem), "--dropout", "0.15",
                 "--epochs", str(args.epochs), "--patience", str(args.patience),
                 "--out-dir", args.out_dir],
                cwd="/content", env={**os.environ, "PYTHONPATH": "/content"},
                capture_output=True, text=True)
            tail = [x for x in r.stdout.splitlines() if x.startswith(f"[net] {tag}")]
            print(tail[-1] if tail else
                  f"[long] {tag} s{s} FAILED rc={r.returncode}: "
                  f"{(r.stderr.strip().splitlines() or ['no stderr'])[-1][:150]}", flush=True)
            # Push immediately: Pro+ runtimes die mid-experiment and took results
            # with them twice already. Shared with the ablation drivers, which
            # lacked it and lost two whole experiments on 2026-10-05.
            push_result(done, "cyp_long")

    # Report the curve at fixed checkpoints so "improvement at 100" is explicit.
    marks = [20, 50, 100, 200, 400, 800, 1500]
    print(f"\n{'config':12s} {'best':>8s} {'at ep':>6s} " +
          " ".join(f"{'ep'+str(m):>8s}" for m in marks))
    for f in sorted(Path(args.out_dir).glob("*.json")):
        d = json.loads(f.read_text())
        cur = dict(d.get("curve", []))
        eps = sorted(cur)
        def at(m):
            prior = [e for e in eps if e <= m]
            return f"{cur[prior[-1]]:.4f}" if prior else "   -  "
        print(f"{d['tag']+'_s'+str(d['seed']):12s} {d['valid']:8.4f} {d['best_epoch']:6d} " +
              " ".join(f"{at(m):>8s}" for m in marks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
