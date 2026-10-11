"""Does matching the loss to the metric help? The Huber delta sweep.

The scored metric is a **pure L1 hinge**: distance to the nearer credible-interval
bound, zero inside. Our loss Huber-smooths it at delta=1.0, and measurement on
this dataset says that smoothing is not a detail:

| endpoint | median interval width | molecules missing by < 1.0 |
|---|---|---|
| CYP1A2 | 0.328 | 83.3% |
| CYP2C9 | 0.526 | 93.5% |
| CYP2D6 | 0.272 | 87.0% |
| CYP3A4 | 0.379 | 83.0% |

So the model is trained almost entirely in the quadratic region. Two things
follow, and they are claims about the *estimator*, not the step size:

1. A squared hinge is **mean-seeking**. The L1 hinge is minimised where
   `P(lo > p) == P(hi < p)` -- a balance point between "interval entirely above
   the prediction" and "entirely below". These are different points whenever the
   interval distribution is asymmetric, and here it is: low-activity compounds
   carry much wider intervals.
2. A molecule missing the band by 0.35 contributes gradient 0.35 under the
   quadratic but is charged 1.0 per unit by the metric. Near-misses are exactly
   the molecules where a small shift converts error into an exact zero, and they
   are the ones the current loss under-weights.

This also offers an explanation for the one gain this programme has verified: the
per-endpoint additive offset fitted post-hoc (`cyp_optimise.py`) may have been
partially hand-correcting this mismatch. If so, aligning the loss should shrink
the offset's remaining benefit -- which is a falsifiable prediction, recorded here
before the run.

Arms are matched in everything but delta: same split, same seeds, same epochs,
same requested batch size. `cyp_net_one.py` records the *final* batch size, and
this script refuses to compare arms whose batch sizes diverged, because OOM
backoff silently breaking a control has already happened once here.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

# delta -> 0 is the metric itself; 1.0 is what every run so far used.
DELTAS = [1.0, 0.3, 0.1, 0.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-dir", default="/content/out/cyp_delta")
    ap.add_argument("--only", nargs="*", type=float, default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for d in (args.only if args.only is not None else DELTAS):
        for s in range(args.seeds):
            tag = f"delta{d:g}".replace(".", "p")
            if (out / f"{tag}_s{s}.json").exists():
                print(f"[delta] {tag} s{s}: cached", flush=True); continue
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(s),
                 "--d-model", str(args.d_model), "--n-layers", str(args.n_layers),
                 "--members", str(args.members), "--dropout", "0.15",
                 "--epochs", str(args.epochs), "--patience", str(args.patience),
                 "--batch-size", str(args.batch_size),
                 "--huber-delta", str(d),
                 "--out-dir", args.out_dir],
                capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if ln.startswith("[net]") and tag in ln:
                    print(ln, flush=True)
            if r.returncode != 0:
                print(f"[delta] {tag} s{s} FAILED rc={r.returncode}", flush=True)
                print((r.stdout or "")[-1200:], (r.stderr or "")[-1200:], flush=True)

    # ── paired comparison against delta=1.0, the incumbent ───────────
    runs = {}
    for f in sorted(out.glob("*.json")):
        d = json.loads(f.read_text())
        runs.setdefault(d["huber_delta"], {})[d["seed"]] = d

    bs = {(dl, s): r["batch_size_final"] for dl, v in runs.items() for s, r in v.items()}
    if len(set(bs.values())) > 1:
        print(f"\n[delta] REFUSING to compare: batch sizes diverged {bs}", flush=True)
        print("[delta] an arm that silently halved its batch is not a matched arm.", flush=True)
        return 1

    print("\n[delta] SUMMARY (lower is better)", flush=True)
    ref = 1.0
    if ref not in runs:
        print("[delta] no delta=1.0 arm; nothing to compare against", flush=True); return 0
    base = runs[ref]
    rows = {}
    for dl in sorted(runs):
        seeds = sorted(set(runs[dl]) & set(base))
        v = np.array([runs[dl][s]["valid"] for s in seeds])
        print(f"    delta {dl:<4g} n={len(v)} mean {v.mean():.4f} sd {v.std(ddof=1) if len(v)>1 else 0:.4f} "
              f"best-epochs {[runs[dl][s]['best_epoch'] for s in seeds]}")
        rows[dl] = (seeds, v)

    print("\n[delta] paired against delta=1.0 (positive = better than incumbent)", flush=True)
    for dl in sorted(runs):
        if dl == ref:
            continue
        seeds, v = rows[dl]
        b = np.array([base[s]["valid"] for s in seeds])
        diff = b - v
        if len(diff) < 2:
            print(f"    delta {dl:<4g}: {diff.mean():+.4f} (1 seed -- not a result)"); continue
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        t = diff.mean() / se if se > 0 else 0.0
        wins = int((diff > 0).sum())
        print(f"    delta {dl:<4g}: {diff.mean():+.4f} +/- {se:.4f} (n={len(diff)}, "
              f"wins {wins}/{len(diff)}, t={t:+.2f}) -> "
              f"{'REAL' if abs(t) > 2.5 and wins in (0, len(diff)) else 'inside noise'}")
    (out / "summary.json").write_text(json.dumps(
        {str(k): {"seeds": list(map(int, s)), "valid": v.tolist()} for k, (s, v) in rows.items()},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
