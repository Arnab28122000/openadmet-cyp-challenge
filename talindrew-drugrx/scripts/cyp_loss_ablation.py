"""Ablate the programme's founding assumption: must the loss be interval-aware?

`drugrx/cyp/metric.py` has said since day one that "the loss should be
interval-aware -- zero inside, hinge outside. That is what `interval_loss` does,
and it is the only part of our stack that needed a new objective rather than a new
feature." That decision was never ablated, and it has an unmeasured cost: the
hinge is **exactly zero inside the credible interval**, so every compound already
inside contributes no gradient at all. With median interval widths of 0.27-0.53
and most predictions landing inside, the model stops learning from the majority of
its data -- which is consistent with our validation R-squared of 0.318 against the
leaderboard leaders' 0.635 on data that is 100% in-domain.

Two arms, matched in everything else, both scored with the **official** metric and
both given the per-endpoint additive offset afterwards (the offset is the only
decode step that has ever survived a paired test, +0.0211 +/- 0.0015):

* `interval` -- the incumbent hinge loss.
* `point` -- Huber on the point estimate; every labelled compound informative.

If `point` wins, the founding decision cost us accuracy for three weeks and the
right architecture is "fit the conditional mean, then decode for the metric",
which is also what the decision-theory analysis concluded independently.

R-squared and MAE are reported next to ST-RAE, but **ST-RAE decides** -- the
leaderboard ranks on it alone, and this programme has already established that
shrinkage helps ST-RAE while hurting R-squared, so an arm could legitimately win
the ranked metric while losing the diagnostics.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content")
from drugrx.cyp.metric import METRIC_ID  # noqa: E402
from drugrx.cyp.publish import push_result  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-dir", default="/content/out/cyp_loss")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for loss in ("interval", "point"):
        for s in range(args.seeds):
            tag = f"loss-{loss}"
            if (out / f"{tag}_s{s}.json").exists():
                # Existence is not enough: a JSON written under a different metric
                # must be re-run, not reused. Pooling across a metric change is how
                # five old-metric depth runs were averaged with three official ones.
                try:
                    prev = json.loads((out / f"{tag}_s{s}.json").read_text()).get("metric_id")
                except Exception:
                    prev = None
                if prev == METRIC_ID:
                    print(f"[loss] {tag} s{s}: cached", flush=True); continue
                print(f"[loss] {tag} s{s}: stale metric ({prev}) -- re-running",
                      flush=True)
                (out / f"{tag}_s{s}.json").unlink()
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(s), "--loss", loss,
                 "--d-model", str(args.d_model), "--n-layers", str(args.n_layers),
                 "--members", str(args.members), "--dropout", "0.15",
                 "--epochs", str(args.epochs), "--patience", str(args.patience),
                 "--batch-size", str(args.batch_size), "--out-dir", args.out_dir],
                capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if ln.startswith("[net]"):
                    print(ln, flush=True)
            if r.returncode != 0:
                print(f"[loss] {tag} s{s} FAILED rc={r.returncode}", flush=True)
                print((r.stdout or "")[-900:], (r.stderr or "")[-900:], flush=True)
            # Push the moment it lands: four Pro+ runtimes died at once on
            # 2026-10-05 and this driver had no upload, so the whole experiment
            # was lost while `cyp_long.py` lost only its run in progress.
            push_result(out / f"{tag}_s{s}.json", "cyp_loss")

    runs = {}
    bss = set()
    for f in sorted(out.glob("*.json")):
        d = json.loads(f.read_text())
        if "loss" not in d:
            continue
        runs.setdefault(d["loss"], {})[int(d["seed"])] = d
        bss.add(d.get("batch_size_final"))
    if len(bss) > 1:
        print(f"\n[loss] REFUSING to compare: batch sizes diverged {bss} -- "
              f"OOM backoff broke the matching", flush=True)
        return 1
    if len(runs) < 2:
        print(f"\n[loss] need both arms; have {sorted(runs)}", flush=True)
        return 1

    print("\n[loss] OFFICIAL validation MA-ST-RAE (1.0 == predicting the mean)", flush=True)
    for k in sorted(runs):
        v = np.array([runs[k][s]["valid"] for s in sorted(runs[k])])
        eps = [runs[k][s]["best_epoch"] for s in sorted(runs[k])]
        print(f"    {k:9s} n={len(v)} mean {v.mean():.4f} "
              f"sd {v.std(ddof=1) if len(v) > 1 else 0:.4f} best epochs {sorted(eps)}")

    seeds = sorted(set(runs["interval"]) & set(runs["point"]))
    d = np.array([runs["interval"][s]["valid"] - runs["point"][s]["valid"] for s in seeds])
    if len(d) < 2:
        print(f"\n[loss] only {len(d)} shared seed(s) -- not a result", flush=True)
        return 0
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("inf")
    w = int((d > 0).sum())
    tag = ("point IS BETTER" if abs(t) > 2.5 and w == len(d) else
           "point is WORSE (consistent)" if abs(t) > 2.5 and w == 0 else "inside noise")
    print(f"\n[loss] point vs interval: {d.mean():+.4f} +/- {se:.4f} "
          f"(n={len(d)}, wins {w}/{len(d)}, t={t:+.2f}) -> {tag}", flush=True)
    if w == len(d) and abs(t) > 2.5:
        print("[loss] the founding assumption that the loss must be interval-aware "
              "is refuted on this data.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
