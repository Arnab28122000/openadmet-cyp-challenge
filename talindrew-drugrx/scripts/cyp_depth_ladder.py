"""Does depth keep paying, or does it stop? A matched 3/6/9/12-layer ladder.

At a long budget (3000 epochs, patience 500) six layers beat three on both seeds
by ~0.016 with a within-arm spread of 0.0007, where H12 had reported depth as a
null at a fixed epoch budget. If that holds, the obvious next question is whether
depth is a knob that keeps turning or one that saturates immediately past three.

**Everything but `n_layers` is held fixed**, including the member count, which is
reduced to 8 so that the deepest arm fits a 23.7 GB L4 without the OOM backoff
changing the batch size in one arm only -- the defect that invalidated an earlier
comparison here. 8 members is fewer than the 16 used in the long runs, so the
*absolute* scores are not comparable with those; only the within-ladder
differences are, which is all this asks.

Deliberately not inferring a trend from the best point alone: the band mean past
epoch 50 is reported too, because a minimum over ~160 noisy evaluations is partly
an extreme-value artifact, and the winner's-curse test showed the real optimum
here is early (epoch 15-25) with everything after it genuine overfitting.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/content")
from drugrx.cyp.metric import METRIC_ID  # noqa: E402
from drugrx.cyp.publish import push_result  # noqa: E402

LAYERS = [3, 6, 9, 12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-dir", default="/content/out/cyp_ladder")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for L in (args.layers or LAYERS):
        for s in range(args.seeds):
            tag = f"L{L}"
            if (out / f"{tag}_s{s}.json").exists():
                # Existence is not enough: a JSON written under a different metric
                # must be re-run, not reused. Pooling across a metric change is how
                # five old-metric depth runs were averaged with three official ones.
                try:
                    prev = json.loads((out / f"{tag}_s{s}.json").read_text()).get("metric_id")
                except Exception:
                    prev = None
                if prev == METRIC_ID:
                    print(f"[ladder] {tag} s{s}: cached", flush=True); continue
                print(f"[ladder] {tag} s{s}: stale metric ({prev}) -- re-running",
                      flush=True)
                (out / f"{tag}_s{s}.json").unlink()
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(s),
                 "--d-model", str(args.d_model), "--n-layers", str(L),
                 "--members", str(args.members), "--dropout", "0.15",
                 "--epochs", str(args.epochs), "--patience", str(args.patience),
                 "--batch-size", str(args.batch_size),
                 "--out-dir", args.out_dir],
                capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if ln.startswith("[net]"):
                    print(ln, flush=True)
            if r.returncode != 0:
                print(f"[ladder] {tag} s{s} FAILED rc={r.returncode}", flush=True)
                print((r.stdout or "")[-800:], (r.stderr or "")[-800:], flush=True)
            push_result(out / f"{tag}_s{s}.json", "cyp_ladder")

    import numpy as np

    runs = {}
    for f in sorted(out.glob("*.json")):
        d = json.loads(f.read_text())
        if "curve" not in d:
            continue
        cur = sorted((int(k), float(v)) for k, v in dict(d["curve"]).items())
        ep = np.array([a for a, _ in cur]); sc = np.array([b for _, b in cur])
        late = sc[ep >= 50]
        runs.setdefault(d["n_layers"], []).append(
            {"seed": d["seed"], "best": d["valid"], "best_epoch": d["best_epoch"],
             "band_mean": float(late.mean()) if len(late) else float("nan"),
             "bs": d.get("batch_size_final")})

    print("\n[ladder] SUMMARY", flush=True)
    bss = {r["bs"] for v in runs.values() for r in v}
    if len(bss) > 1:
        print(f"[ladder] REFUSING to compare: batch sizes diverged {bss}", flush=True)
        return 1
    for L in sorted(runs):
        b = np.array([r["best"] for r in runs[L]])
        m = np.array([r["band_mean"] for r in runs[L]])
        print(f"    {L:2d} layers n={len(b)}  best {b.mean():.4f} (sd {b.std(ddof=1) if len(b)>1 else 0:.4f})"
              f"  band mean {m.mean():.4f}  best epochs {sorted(r['best_epoch'] for r in runs[L])}")

    if 3 in runs:
        print("\n[ladder] paired vs 3 layers (positive = deeper better)", flush=True)
        base = {r["seed"]: r for r in runs[3]}
        for L in sorted(runs):
            if L == 3:
                continue
            cur = {r["seed"]: r for r in runs[L]}
            sh = sorted(set(base) & set(cur))
            for key in ("best", "band_mean"):
                d = np.array([base[s][key] - cur[s][key] for s in sh])
                if len(d) < 2:
                    print(f"    {L:2d}L {key:10s} {d.mean():+.4f} (n={len(d)} -- not a result)")
                    continue
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = d.mean() / se if se > 0 else float("inf")
                w = int((d > 0).sum())
                tag = ("BETTER" if abs(t) > 2.5 and w == len(d) else
                       "WORSE (consistent)" if abs(t) > 2.5 and w == 0 else "inside noise")
                print(f"    {L:2d}L {key:10s} {d.mean():+.4f} +/- {se:.4f} "
                      f"(n={len(d)}, wins {w}/{len(d)}, t={t:+.2f}) -> {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
