"""Positive control: spike a pull with test molecules and confirm the audit goes hot.

The null-vs-null fixture certifies that the comparison path does not invent an
effect. It cannot certify that the path would *find* one. **A detector validated
only against the null cannot distinguish "no effect" from "blind"**, and
everything downstream of here is a question whose headline could be a null — so
the complementary fixture has to exist before those nulls are believed.

The construction: take a clean source (ExpansionRx, measured at **zero** identity
overlap with every test fold), spike it with a known fraction of actual test
molecules carrying their true labels, and confirm the leakage audit reports them.
The spike levels bracket what we would want to catch — 1% is below the level that
would distort a benchmark, 10% is comfortably above it, and the audit should be
cold at 0% and hot by 5%.

This is deliberately a test of the **audit**, not of the model. A spiked training
set would of course score better; the question here is only whether the detector
says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")

from drugrx.data.splits import nearest_neighbour_similarity  # noqa: E402
from drugrx.data.standardise import audit_overlap, standardise_many  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/out/positive_control.json")
    ap.add_argument("--endpoint", default="cl_micro")
    ap.add_argument("--spikes", type=float, nargs="*", default=[0.0, 0.01, 0.05, 0.10, 0.25])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from datasets import load_dataset

    frame = pd.read_parquet(args.data)
    held = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in held])
    good = frame[frame["key"] != ""]

    col = f"{args.endpoint}__fold"
    test = good[good[col] == "test"]
    test_keys = set(test["key"])
    test_smiles = test["smiles"].tolist()
    print(f"[pos] {args.endpoint}: {len(test_keys)} test molecules", flush=True)

    parts = []
    for repo in ("openadmet/openadmet-expansionrx-challenge-data",
                 "openadmet/openadmet-expansionrx-challenge-train-data"):
        try:
            ds = load_dataset(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[pos] {repo}: {exc}", flush=True)
            continue
        for _, p in ds.items():
            parts.append(p.to_pandas())
    erx = pd.concat(parts, ignore_index=True)
    std = standardise_many([s for s in erx["SMILES"] if isinstance(s, str)], workers=args.workers)
    clean_keys = [s.inchikey for s in std if s.ok]
    clean_smiles = [s.smiles for s in std if s.ok]
    print(f"[pos] clean source: {len(clean_keys)} molecules", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for spike in args.spikes:
        n_spike = int(round(spike * len(test_keys)))
        picked = list(rng.choice(len(test_smiles), min(n_spike, len(test_smiles)), replace=False))
        spiked_keys = set(clean_keys) | {test["key"].iloc[i] for i in picked}
        spiked_smiles = clean_smiles + [test_smiles[i] for i in picked]

        result = audit_overlap(spiked_keys, test_keys, f"spike={spike:.0%}")
        sample = spiked_smiles
        if len(sample) > 900:
            sample = [sample[i] for i in rng.choice(len(sample), 900, replace=False)]
        nn = nearest_neighbour_similarity(sample, test_smiles)

        row = {
            "spike_fraction": spike,
            "molecules_spiked": len(picked),
            "detected_overlap": result["n_overlap"],
            "frac_test_flagged": result["frac_test_leaked"],
            "nn_frac_above_0.9": float((nn > 0.9).mean()) if len(nn) else 0.0,
            # The audit must flag exactly what was planted — no more, no fewer.
            "exact": result["n_overlap"] == len(picked),
        }
        rows.append(row)
        print(f"  spike {spike:5.0%}: planted {len(picked):4d}, detected "
              f"{result['n_overlap']:4d} ({result['frac_test_leaked']:.1%} of fold) "
              f"{'OK' if row['exact'] else 'MISMATCH'}", flush=True)

    cold = rows[0]["detected_overlap"] == 0
    hot = all(r["detected_overlap"] == r["molecules_spiked"] for r in rows[1:])
    verdict = "PASS" if (cold and hot) else "FAIL"
    print(f"\nPOSITIVE CONTROL: cold at 0% = {cold}, exact at every spike = {hot} → {verdict}")
    print("A detector certified only against the null cannot tell 'no effect' from 'blind'.")

    Path(args.out).write_text(json.dumps(
        {"endpoint": args.endpoint, "rows": rows, "verdict": verdict}, indent=2, default=float))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
