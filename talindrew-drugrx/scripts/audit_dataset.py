"""Audit the merged dataset for duplicate molecules and split leakage.

Run after every merge. Standardisation collapses salts, charge states, tautomers
and alternate SMILES writings onto one InChIKey, which is the only way to find a
molecule that sits in training under one spelling and in a test fold under
another — leakage no split logic can catch, because at the string level the two
rows look like different compounds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/content")
from drugrx.data.standardise import audit_overlap, standardise_many  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/runs/audit.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    frame = pd.read_parquet(args.data)
    print(f"rows: {len(frame)}", flush=True)

    std = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    ok = sum(s.ok for s in std)
    print(f"standardised ok: {ok}/{len(std)}  failures: {len(std) - ok}", flush=True)

    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in std])
    good = frame[frame["key"] != ""]
    dupes = int(good["key"].duplicated().sum())

    sets = {s: set(good.loc[good["split"] == s, "key"]) for s in ("train", "valid", "test")}
    report = {
        "rows": int(len(frame)),
        "standardised_ok": int(ok),
        "failures": int(len(std) - ok),
        "duplicate_rows_after_standardisation": dupes,
        "duplicate_fraction": round(dupes / max(1, len(good)), 5),
        "train_vs_test": audit_overlap(sets["train"], sets["test"], "train-vs-test"),
        "train_vs_valid": audit_overlap(sets["train"], sets["valid"], "train-vs-valid"),
    }
    print(json.dumps(report, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
