r"""Does the ChEMBL tail acquisition actually help? Four arms, and a tail-bin read.

The tail analysis established what new data must do: the most-active quintile
carries **45.8% of the ST-RAE numerator** on 20% of the molecules, 94.8% of it
outside its interval, and our bias there is **-0.818** (we under-predict actives).
ChEMBL was acquired against a pre-registered tail criterion and supplies **4,902
compounds at pIC50 >= 5.5** against the 960 the challenge training set holds.

Four arms, matched in everything else:

* `none`    -- no auxiliary heads at all. The control, which did not exist before
              today: every previous comparison was between two auxiliary
              configurations, so H7a's question could not be re-asked.
* `veith`   -- the incumbent. Binary qHTS labels; cannot inform the tail.
* `chembl`  -- continuous pIC50 only.
* `both`    -- 8 heads.

**The macro score is not the only thing read.** A source acquired to fix the tail
should be judged on the tail, so each arm's saved validation predictions are binned
by true label and the most-active bin is reported separately. An arm that improves
the macro by moving the inactive bulk has not done what this acquisition was for,
and an arm that improves the tail while the macro stays flat is still the right
direction given where the numerator mass lives.

Paired across seeds, official metric.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content")
from drugrx.cyp.metric import METRIC_ID  # noqa: E402
from drugrx.cyp.publish import push_result  # noqa: E402

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = {"none": ["none"], "veith": ["veith_"], "chembl": ["chembl_"],
        "both": ["veith_", "chembl_"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--external", default="/content/out/cyp/external_merged.parquet")
    ap.add_argument("--out-dir", default="/content/out/cyp_aux")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for arm, prefixes in ARMS.items():
        if args.only and arm not in args.only:
            continue
        for s in range(args.seeds):
            tag = f"aux-{arm}"
            if (out / f"{tag}_s{s}.json").exists():
                # Existence is not enough: a JSON written under a different metric
                # must be re-run, not reused. Pooling across a metric change is how
                # five old-metric depth runs were averaged with three official ones.
                try:
                    prev = json.loads((out / f"{tag}_s{s}.json").read_text()).get("metric_id")
                except Exception:
                    prev = None
                if prev == METRIC_ID:
                    print(f"[aux] {tag} s{s}: cached", flush=True); continue
                print(f"[aux] {tag} s{s}: stale metric ({prev}) -- re-running",
                      flush=True)
                (out / f"{tag}_s{s}.json").unlink()
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(s),
                 "--aux-prefix", *prefixes,
                 "--external", args.external,
                 "--d-model", "1024", "--n-layers", "3", "--members", str(args.members),
                 "--dropout", "0.15", "--epochs", str(args.epochs),
                 "--patience", str(args.patience), "--batch-size", "4096",
                 "--out-dir", args.out_dir],
                capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if ln.startswith("[net]") and ("auxiliary heads" in ln or tag in ln):
                    print(ln, flush=True)
            if r.returncode != 0:
                print(f"[aux] {tag} s{s} FAILED rc={r.returncode}", flush=True)
                print((r.stdout or "")[-1000:], (r.stderr or "")[-1000:], flush=True)
            # Push the moment it lands: four Pro+ runtimes died at once on
            # 2026-10-05 and this driver had no upload, so the whole experiment
            # was lost while `cyp_long.py` lost only its run in progress.
            push_result(out / f"{tag}_s{s}.json", "cyp_aux")

    # ── collect, check matching, report macro then the tail bin ──────
    from drugrx.cyp.data import PRIMARY, load
    from drugrx.cyp.metric import soft_threshold_error
    from drugrx.data.splits import scaffold_split

    runs, bss = {}, set()
    for f in sorted(out.glob("*.json")):
        d = json.loads(f.read_text())
        if "aux_prefix" not in d:
            continue
        arm = d["tag"].replace("aux-", "")
        runs.setdefault(arm, {})[int(d["seed"])] = d
        bss.add(d.get("batch_size_final"))
    if len(bss) > 1:
        print(f"\n[aux] REFUSING to compare: batch sizes diverged {bss}", flush=True)
        return 1

    print("\n[aux] OFFICIAL macro ST-RAE (1.0 == predicting the mean)", flush=True)
    for arm in ARMS:
        if arm not in runs:
            continue
        v = np.array([runs[arm][s]["valid"] for s in sorted(runs[arm])])
        print(f"    {arm:7s} n={len(v)} mean {v.mean():.4f} "
              f"sd {v.std(ddof=1) if len(v) > 1 else 0:.4f}  "
              f"heads {runs[arm][sorted(runs[arm])[0]]['n_aux_heads']}  "
              f"best epochs {sorted(runs[arm][s]['best_epoch'] for s in sorted(runs[arm]))}")

    ref = "veith" if "veith" in runs else "none"
    print(f"\n[aux] paired vs {ref} (positive = better)", flush=True)
    for arm in ARMS:
        if arm == ref or arm not in runs:
            continue
        sh = sorted(set(runs[arm]) & set(runs[ref]))
        if len(sh) < 2:
            print(f"    {arm:7s} only {len(sh)} shared seed(s) -- not a result"); continue
        d = np.array([runs[ref][s]["valid"] - runs[arm][s]["valid"] for s in sh])
        se = d.std(ddof=1) / np.sqrt(len(d)); t = d.mean() / se if se > 0 else float("inf")
        w = int((d > 0).sum())
        tag = ("BETTER" if abs(t) > 2.5 and w == len(d) else
               "WORSE (consistent)" if abs(t) > 2.5 and w == 0 else "inside noise")
        print(f"    {arm:7s} {d.mean():+.4f} +/- {se:.4f} (n={len(d)}, wins {w}/{len(d)}, "
              f"t={t:+.2f}) -> {tag}")

    # ── the tail bin, which is what the acquisition was for ─────────
    data = load(workers=12)
    frame = data.frame
    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=0)
    va = np.concatenate([va, rest]) if len(rest) else va

    print(f"\n[aux] most-active bin (top {100//args.bins}% by true label): mean hinge "
          f"error per molecule, and bias", flush=True)
    for arm in ARMS:
        if arm not in runs:
            continue
        hinges, biases = [], []
        for s in sorted(runs[arm]):
            npz = out / f"aux-{arm}_s{s}.npz"
            if not npz.exists():
                continue
            z = np.load(npz, allow_pickle=True)
            pv, idx = z["val"], z["va_index"]
            for i, c in enumerate(PRIMARY):
                y = frame[c].to_numpy(float)[idx]
                k = np.isfinite(y)
                if k.sum() < 20:
                    continue
                yy, p = y[k], pv[k, i]
                l, h = lo[idx][k, i], hi[idx][k, i]
                thr = np.quantile(yy, 1.0 - 1.0 / args.bins)
                m = yy >= thr
                if m.sum() < 5:
                    continue
                hinges.append(float(soft_threshold_error(p[m], l[m], h[m]).mean()))
                biases.append(float(np.mean(p[m] - yy[m])))
        if hinges:
            print(f"    {arm:7s} tail mean hinge {np.mean(hinges):.4f}  "
                  f"tail bias {np.mean(biases):+.4f}  (n={len(hinges)} arm-endpoint-seeds)")
    print("\n[aux] a source acquired for the tail must move the tail; a macro gain that "
          "comes from the inactive bulk is not what this was for.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
