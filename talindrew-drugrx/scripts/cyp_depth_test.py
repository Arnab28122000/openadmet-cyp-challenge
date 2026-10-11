"""Was H12's depth null a budget confound? A paired depth test at a long budget.

H12 reported depth as a null (p=0.723) and that conclusion has shaped this
programme: it is one of the six nulls behind "bound by data and representation,
not model class". The long runs cast doubt on it.

| config | seed 0 | seed 1 | within-arm spread |
|---|---|---|---|
| long-base (1024x3) | 0.4441 | 0.4448 | 0.0007 |
| long-deep (1024x6) | 0.4278 | 0.4281 | 0.0007 |

Two seeds each, non-overlapping, a gap of ~0.016 against a within-arm spread of
0.0007. That is not a result at n=2, but it has a mechanism behind it: the
shallow model peaks at epoch 20-25 and the deep one at 45-230, so **a comparison
run to a fixed epoch budget stops the deep arm before it has arrived.** H12's own
measurement recorded exactly that shift (+116.2 +/- 20.2 epochs, p=0.0014) and
reported it as a curiosity beside the score null -- so the evidence for the
confound was already in hand and was read the other way.

This script reads the paired runs and tests it properly. It does **not** retrain:
`cyp_long.py` writes one JSON per (config, seed) and the comparison is a paired
test over seeds, which is the right unit -- members are vmapped inside a run, so
seeds are the independent replicates.

Reported three ways, because "best epoch" is itself suspect here (the validation
band oscillates with sd ~0.01 for hundreds of epochs, so a minimum over ~160
evaluations is partly an extreme-value artifact):

* **best** -- minimum over the curve, i.e. what every arm in this programme has
  quoted. Comparable to prior results, and optimistically biased in both arms.
* **band mean** -- mean of the curve past `--band-start`. Immune to the
  extreme-value problem and the honest estimate of where the arm actually sits.
* **band min** -- for completeness.

If depth wins on the band mean as well as on the best point, the effect is not an
artifact of selection noise and H12's depth conclusion needs revising in the log.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def load_runs(d: Path):
    runs = {}
    for f in sorted(d.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if "tag" not in r or "curve" not in r:
            continue
        runs.setdefault(r["tag"], {})[int(r["seed"])] = r
    return runs


def stats(r, band_start):
    cur = sorted((int(k), float(v)) for k, v in dict(r["curve"]).items())
    ep = np.array([a for a, _ in cur]); sc = np.array([b for _, b in cur])
    late = sc[ep >= band_start]
    return {"best": float(r["valid"]), "best_epoch": int(r["best_epoch"]),
            "band_mean": float(late.mean()) if len(late) else float("nan"),
            "band_min": float(late.min()) if len(late) else float("nan"),
            "band_sd": float(late.std()) if len(late) else float("nan"),
            "n_band": int(len(late)), "epochs_run": int(r.get("epochs_run", -1))}


def paired(a, b, label):
    """a, b are dicts seed -> value. Positive difference means `a` is better
    (lower score), since these are errors."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        print(f"    {label:12s} only {len(seeds)} shared seed(s) -- not a result")
        return
    d = np.array([b[s] - a[s] for s in seeds])
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("inf")
    wins = int((d > 0).sum())
    verdict = "REAL" if abs(t) > 2.5 and wins in (0, len(d)) else "inside noise"
    print(f"    {label:12s} {d.mean():+.4f} +/- {se:.4f} (n={len(d)}, "
          f"wins {wins}/{len(d)}, t={t:+.2f}) -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/content/out/cyp_long")
    ap.add_argument("--shallow", default="long-base")
    ap.add_argument("--deep", default="long-deep")
    ap.add_argument("--band-start", type=int, default=50)
    args = ap.parse_args()

    runs = load_runs(Path(args.dir))
    if args.shallow not in runs or args.deep not in runs:
        print(f"[depth] need both {args.shallow} and {args.deep}; have {sorted(runs)}")
        return 1

    ids = {r.get("metric_id", "MISSING") for v in runs.values()
           for r in (v.values() if isinstance(v, dict) else v)}
    if len(ids) > 1 or "MISSING" in ids:
        print(f"\n[depth] REFUSING to compare: runs scored under different metrics "
              f"{sorted(ids)}. Delete the stale ones and re-run; averaging across a "
              f"metric change is how five old-metric depth runs were pooled with three "
              f"official ones.", flush=True)
        return 1
    table = {}
    for tag in (args.shallow, args.deep):
        table[tag] = {s: stats(r, args.band_start) for s, r in runs[tag].items()}
        print(f"\n[depth] {tag}")
        for s in sorted(table[tag]):
            v = table[tag][s]
            print(f"    seed {s}: best {v['best']:.4f}@{v['best_epoch']:<4d} "
                  f"band mean {v['band_mean']:.4f} sd {v['band_sd']:.4f} "
                  f"(n={v['n_band']}, ran {v['epochs_run']})")

    print(f"\n[depth] paired, deep vs shallow (positive = deep better)")
    for key in ("best", "band_mean", "band_min"):
        paired({s: v[key] for s, v in table[args.deep].items()},
               {s: v[key] for s, v in table[args.shallow].items()}, key)

    print(f"\n[depth] best-epoch shift (the budget-confound mechanism)")
    for tag in (args.shallow, args.deep):
        e = [v["best_epoch"] for v in table[tag].values()]
        print(f"    {tag:12s} best epochs {sorted(e)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
