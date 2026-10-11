"""Re-derive TDC's split leakage with a background-calibrated threshold.

The figure on the model card — "47% of caco2 test molecules have a training
neighbour within Tanimoto 0.4" — is our own measurement, but it uses a **fixed
0.4 cut**, which is precisely the uncalibrated threshold that produced the Avalon
99.8% artefact. That row was retired by comparing against a random-background
null; this figure never received the same treatment, so by this project's own
current standard it is unverified.

The calibrated question is not "how many test molecules sit within 0.4 of a
training molecule" — that depends on the fingerprint's density. It is **"are test
molecules closer to the training set than to ordinary chemistry?"** If the shift
over background is ~0, the raw fraction is baseline density and says nothing
about the split.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/content")
from drugrx.data.splits import nearest_neighbour_similarity, proximity_verdict
from drugrx.data.standardise import standardise_many


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/out/leakage_calibrated.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=700)
    args = ap.parse_args()

    frame = pd.read_parquet(args.data)
    std = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in std])
    good = frame[frame["key"] != ""]
    rng = np.random.default_rng(0)

    rows = []
    for col in sorted(c for c in good.columns if c.endswith("__fold")):
        endpoint = col[: -len("__fold")]
        test = good.loc[good[col] == "test", "smiles"].tolist()
        train = good.loc[good[col] == "train", "smiles"].tolist()
        if len(test) < 20 or len(train) < 50:
            continue

        # Background: molecules from other endpoints entirely, never in this
        # endpoint's train or test. "Ordinary chemistry" for this comparison.
        other = good.loc[(good[col] == "") , "smiles"].tolist()
        if len(other) < 200:
            other = good["smiles"].tolist()
        bg = [other[i] for i in rng.choice(len(other), min(len(train), len(other)), replace=False)]

        t = test if len(test) <= args.sample else [test[i] for i in rng.choice(len(test), args.sample, replace=False)]
        to_train = nearest_neighbour_similarity(t, train)
        to_bg = nearest_neighbour_similarity(t, bg)
        shift = float(to_train.mean() - to_bg.mean())
        rows.append({
            "endpoint": endpoint, "test": len(test), "train": len(train),
            "raw_frac_above_0.4": round(float((to_train > 0.4).mean()), 4),
            "mean_to_train": round(float(to_train.mean()), 4),
            "mean_to_background": round(float(to_bg.mean()), 4),
            "shift": round(shift, 4),
            "verdict": proximity_verdict(shift),
        })
        print(f"{endpoint:16s} raw>0.4 {rows[-1]['raw_frac_above_0.4']:6.1%} | "
              f"train {rows[-1]['mean_to_train']:.3f} bg {rows[-1]['mean_to_background']:.3f} "
              f"shift {shift:+.3f} → {rows[-1]['verdict']}", flush=True)

    real = [r for r in rows if r["verdict"] == "real proximity"]
    print(f"\nendpoints where test IS closer to train than to background: {len(real)}/{len(rows)}")
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
