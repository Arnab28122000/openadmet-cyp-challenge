"""Is the in-domain threshold informative, or does it flag almost everything?

**A flag that fires on every row carries no information**, and that failure is
invisible from the inside: each verdict is individually defensible, the column is
full of warnings, and nobody can tell the warnings apart. This is the same trap as
the Avalon row (T6) one level up — there the *threshold* was uncalibrated, here the
threshold is calibrated and the question is whether the *operating point* is useful.

The measurement is simple and decisive: run each endpoint's own **test fold**
through its own domain reference and count how many come back in domain. The test
fold is the population we have actually measured performance on, so it is the one
population where "in domain" has a known meaning.

Three readings, fixed before running so the number cannot be rationalised after:

* **A test-fold in-domain rate near 100%** means the bar is too low to separate
  anything — every molecule passes and novel chemistry would too.
* **A rate near 0%** means the bar is too high — the model's own benchmark is
  out of its stated domain, which makes the flag unreadable rather than strict.
* **Somewhere in between**, and the rate itself is the thing to publish: it says
  what fraction of molecules *like the ones we measured on* we are prepared to
  vouch for.

    python scripts/domain_calibration.py --bundle /content/router

Writes `domain_calibration.json`, sweeping several percentiles so the operating
point is chosen from evidence rather than from the convention 95 happens to be.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm  # noqa: F401,E402  (import order: see drugrx/serve.py)

from drugrx.data.tdc import ADMET_TASKS  # noqa: E402
from drugrx.domain import DomainReference, fingerprints, max_tanimoto  # noqa: E402

PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/content/router")
    ap.add_argument("--out", default="/content/out/domain_calibration.json")
    args = ap.parse_args()

    from drugrx.data.tdc import load_task

    bundle = Path(args.bundle)
    meta = json.loads((bundle / "gbm" / "meta.json").read_text())["endpoints"]

    rows = []
    for task in ADMET_TASKS:
        reference_file = bundle / "domain" / f"{task.key}.npz"
        if not reference_file.exists() or task.key not in meta:
            continue
        with np.load(reference_file) as handle:
            reference = DomainReference.from_npz(task.key, handle["packed"], handle["meta"])

        frame, splits = load_task(task)
        smiles = frame["smiles"].tolist()
        test = [smiles[i] for i in splits["test"]]
        packed, _ = fingerprints(test)
        sims = max_tanimoto(packed, reference.packed)

        # What bar would admit what fraction of the molecules we actually measured
        # on. This is the operating-point question stated directly, and it is the
        # number to choose a percentile from — rather than from the fact that 95
        # is a convention.
        sweep = {
            f"bar_admitting_{p:g}pct_of_test": float(np.percentile(sims, 100.0 - p))
            for p in PERCENTILES
        }

        rows.append({
            "task": task.key,
            "n_test": int(len(sims)),
            "threshold": reference.threshold,
            "percentile": reference.percentile,
            "background_median": reference.background_median,
            "background_mean": reference.background_mean,
            "n_reference": reference.n_reference,
            "test_nn_median": float(np.median(sims)),
            "test_nn_mean": float(sims.mean()),
            "in_domain_rate": float((sims >= reference.threshold).mean()),
            "above_0_3": float((sims >= 0.3).mean()),
            **sweep,
        })
        print(f"{task.key:16s} n={len(sims):5d}  thr={reference.threshold:.3f}  "
              f"test median={np.median(sims):.3f}  in-domain={100 * (sims >= reference.threshold).mean():5.1f}%  "
              f"(fixed 0.3: {100 * (sims >= 0.3).mean():5.1f}%)", flush=True)

    rates = [r["in_domain_rate"] for r in rows]
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoints": len(rows),
        "median_in_domain_rate": float(np.median(rates)) if rates else None,
        "min_in_domain_rate": float(min(rates)) if rates else None,
        "max_in_domain_rate": float(max(rates)) if rates else None,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nmedian in-domain rate on the models' own test folds: "
          f"{100 * payload['median_in_domain_rate']:.1f}% "
          f"(range {100 * payload['min_in_domain_rate']:.1f}–"
          f"{100 * payload['max_in_domain_rate']:.1f}%)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
