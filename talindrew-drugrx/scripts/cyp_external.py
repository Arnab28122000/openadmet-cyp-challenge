"""Add PubChem qHTS CYP inhibition data as auxiliary heads — if it is clean.

CYP2D6 is the weak endpoint: 0.6013 against a 0.6107 floor, with the narrowest
credible intervals (median width 0.27) and only 8.6% of labels below pIC50 4, so
it has the least free score and needs a genuinely better point prediction. The
challenge gives it 1,493 labels.

TDC mirrors Veith's PubChem qHTS panels for **all four** CYPs at ~12–13k molecules
each — roughly 50,000 extra inhibition labels, public domain, on exactly these
enzymes. They are **binary** (inhibitor / non-inhibitor) where the challenge is
pIC50, so they cannot be merged into the target; they go in as auxiliary
classification heads, which is the one use of auxiliary data this challenge's
test format permits.

**The gating question is contamination.** The challenge test set is blinded but
its molecules are real compounds that may well appear in PubChem. Any overlap
between an external panel and the 750 test molecules would train the model on its
own test set, so overlap is measured on standardised InChIKey *before* anything is
used, and colliding molecules are dropped.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/content")
from drugrx.data.standardise import standardise_many
from drugrx.data.tdc import AdmetTask, load_task

B = "hf://datasets/openadmet/cyp-challenge-train-test/"

# The four Veith panels, each with its **pinned mirror row count**.
#
# These are deliberately NOT added to `ADMET_TASKS`. That tuple is the 21-endpoint
# benchmark group: it sets the multi-task model's head count
# (`MultiTaskAdmet(ADMET_TASKS, ...)`), the cached `admet_multitask.parquet`, the
# shipped router's endpoint list and `tests/test_endpoint_names.py`. A 22nd entry
# there would silently change every macro this programme has reported over "21
# endpoints" and invalidate the head layout of every existing checkpoint. CYP1A2
# is also not a TDC ADMET Benchmark Group endpoint, so it does not belong there on
# its own terms.
#
# The counts matter because `load_task`'s drift check was previously handed
# `n_rows=0` from a task fabricated here, so it compared every panel against 0,
# could never fire, and printed "expected 0" -- which reads as noise rather than as
# a disabled guard. The mirrors have moved under this programme before.
# Verified 2026-10-05: all four observed counts equal the values below, and the
# 2C9/2D6/3A4 figures agree exactly with the ones pinned in `ADMET_TASKS`.
PANEL_ROWS = {"CYP1A2": 12579, "CYP2C9": 12092,
              "CYP2D6": 13130, "CYP3A4": 12328}
PANELS = {c: f"TDC_{c.lower()}_veith" for c in PANEL_ROWS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp/external.parquet")
    ap.add_argument("--report", default="/content/out/cyp/external_audit.json")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    test = pd.read_csv(B + "cyp-challenge-TEST-BLINDED.csv")
    ts = standardise_many(test["SMILES"].astype(str).tolist(), workers=args.workers)
    test_keys = {s.inchikey for s in ts if s.ok}
    print(f"[ext] challenge test: {len(test_keys)} standardised keys", flush=True)

    frames, audit = [], {}
    for cyp, repo in PANELS.items():
        # Carry the pinned count so `load_task`'s drift check is live.
        task = AdmetTask(cyp.lower(), repo, "binary", "pr_auc",
                         PANEL_ROWS[cyp], "")
        try:
            f, _ = load_task(task)
        except Exception as exc:  # noqa: BLE001
            print(f"[ext] {cyp}: unavailable ({exc})", flush=True)
            continue
        s = standardise_many(f["smiles"].astype(str).tolist(), workers=args.workers)
        f = f.assign(key=[x.inchikey if x.ok else "" for x in s],
                     std=[x.smiles if x.ok else "" for x in s])
        f = f[f["key"] != ""]
        collisions = set(f["key"]) & test_keys
        before = len(f)
        f = f[~f["key"].isin(test_keys)]
        audit[cyp] = {"rows": before, "unique": int(f["key"].nunique()),
                      "test_collisions_dropped": len(collisions)}
        print(f"[ext] {cyp}: {before} rows, {len(collisions)} collided with the blinded "
              f"test set and were DROPPED, {f['key'].nunique()} kept", flush=True)
        frames.append(f.groupby("key", as_index=False)
                       .agg(**{"smiles": ("std", "first"), f"veith_{cyp}": ("y", "mean")}))

    if not frames:
        print("[ext] nothing usable")
        return 1
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["key", "smiles"], how="outer")
    cols = [c for c in merged.columns if c.startswith("veith_")]
    print(f"[ext] {len(merged)} unique external molecules, "
          f"{int(merged[cols].notna().sum().sum())} labels across {len(cols)} panels", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    Path(args.report).write_text(json.dumps(audit, indent=2))
    print(f"[ext] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
