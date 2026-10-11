"""Assemble every label the CYP challenge provides, as one multi-task table.

The primary file is thin: 4,905 compounds but only **1,285–2,335 labels per
endpoint** and 1.33 labels per compound. The challenge ships three further files
that nothing in the primary task scores, and they are where the extra supervision
is:

| file | compounds | what it adds |
|---|---|---|
| TDI | 6,145 | TDI-condition pIC50 for all four CYPs, plus **1,240 molecules absent from the primary set** |
| Emax | 6,145 | maximal-effect values for all four CYPs |
| single-concentration | 4,376 | 17,504 log2 fold-change readouts, long-format (4 enzymes x 4,376) — **4,375 unique keys**, see below |

The direct-inhibition labels inside the TDI file are the *same* measurements as
the primary file — **checked, not assumed: counts are identical at
1412/1285/1493/2335**, so they add chemistry and auxiliary signal but not primary
labels, and re-adding them would have silently double-weighted those compounds.

Everything is keyed on **standardised InChIKey**, for the reason established
earlier in this programme: a SMILES-level identity check hid 425 train/test
collisions in another dataset, and these files share molecules by construction.

## Why the auxiliary columns are declared rather than matched

They were originally selected with `endswith("_TDI_condition") and "Emax" in c`.
That filter is the shape of defect this programme keeps finding: it **silently
dropped 6 usable heads carrying 9,223 labels** while producing a head count (14)
that matched the docstring, so nothing looked wrong. What it missed:

* `CYP*_EmaxVsPosCtrl_direct_inhibition` — 6,525 labels, the maximal-effect value
  **under the primary assay condition**. Its per-enzyme counts are *exactly* the
  primary pIC50 counts, because it is the same dose-response fit. That makes it
  the most on-target auxiliary signal in the whole release, and it was excluded.
* `CYP1A2_is_TDI` / `CYP2C9_is_TDI` — 2,698 labels, present **only** in the Emax
  file. The TDI file carries just the 2D6 and 3A4 flags.

So the columns are now **named explicitly and checked on load**: a missing one
raises rather than quietly shrinking the head count. 14 heads → **20**,
38,137 labels → **47,360** (pre-merge).

Those two totals read 38,141 and 47,364 until 2026-10-05, and the four-label
difference is worth stating because it is not drift. The added heads are exactly
as described above — `EmaxVsPosCtrl_direct_inhibition` contributes 6,525,
matching the primary count, and the two extra `is_TDI` columns 2,698, so the
documented +9,223 is exact. The discrepancy was entirely in `log2fc`: the
single-concentration file holds **17,504 rows over 4,376 distinct deposited
SMILES**, but two of those SMILES differ only in declared double-bond geometry on
a conjugated cyanine chain where RDKit judges the bond non-stereogenic, so they
standardise to **one** InChIKey. 4,375 unique keys x 4 enzymes = 17,500, and the
old figures had counted the raw rows for this file while counting post-
standardisation keys everywhere else.

The dedup is correct and the standardiser is not lossy about stereochemistry —
checked directly: `C/C=C/C` vs `C/C=C\\C`, `C[C@H](N)C(=O)O` vs
`C[C@@H](N)C(=O)O`, and stereo-vs-unspecified all keep distinct keys.

`is_TDI` is taken from the Emax file for all four enzymes. It is identical in both
files where both carry it (6,145 rows compared, **zero disagreements**), the Emax
file has all four, and taking it from one file avoids a same-name merge collision
that pandas would resolve into `_x`/`_y` suffixes — i.e. silently renamed heads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE = "hf://datasets/openadmet/cyp-challenge-train-test/"
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
PRIMARY = [f"{c}_pIC50_direct_inhibition" for c in CYPS]

# Auxiliary heads, declared per source file. Order fixes the head order.
AUX_TDI = [f"{c}_pIC50_TDI_condition" for c in CYPS]
AUX_EMAX = [f"{c}_EmaxVsPosCtrl_TDI_condition" for c in CYPS]
AUX_EMAX_DIRECT = [f"{c}_EmaxVsPosCtrl_direct_inhibition" for c in CYPS]
AUX_IS_TDI = [f"{c}_is_TDI" for c in CYPS]          # from the Emax file — see above
AUX_LOG2FC = [f"log2fc_{c}" for c in CYPS]
AUXILIARY = [*AUX_TDI, *AUX_EMAX, *AUX_EMAX_DIRECT, *AUX_IS_TDI, *AUX_LOG2FC]


def _require(frame: pd.DataFrame, cols: list[str], source: str) -> None:
    """Fail loudly when a declared column is absent.

    The defect this replaces produced a *plausible* head count from a filter that
    silently matched fewer columns than intended. A rename upstream must break the
    run, not shrink it.
    """
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{source}: {len(missing)} declared column(s) absent: {missing}. "
            "The release layout changed — update AUX_* rather than filtering loosely."
        )


@dataclass
class CypData:
    frame: pd.DataFrame          # smiles + all heads + lo/hi for primary
    primary: list[str]
    auxiliary: list[str]
    test: pd.DataFrame


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE + name)


def load(workers: int = 12) -> CypData:
    from drugrx.data.standardise import standardise_many

    train = _read("cyp-challenge-TRAIN_inhibition.csv")
    tdi = _read("cyp-challenge-TRAIN_TDI.csv")
    emax = _read("cyp-challenge-TRAIN_Emax.csv")
    single = _read("cyp-challenge-single-concentration-TRAIN.csv")
    test = _read("cyp-challenge-TEST-BLINDED.csv")

    # Single-concentration is long-format; pivot to one column per enzyme.
    single_w = (
        single.pivot_table(index="SMILES", columns="enzyme",
                           values="log2fc_estimate", aggfunc="mean")
        .add_prefix("log2fc_")
        .reset_index()
    )

    keep_primary = ["SMILES", *PRIMARY]
    for c in PRIMARY:
        keep_primary += [f"{c}_conf_low", f"{c}_conf_high"]
    _require(train, keep_primary[1:], "TRAIN_inhibition")
    base = train[keep_primary].copy()

    _require(tdi, AUX_TDI, "TRAIN_TDI")
    tdi_small = tdi[["SMILES", *AUX_TDI]].copy()

    # is_TDI comes from the Emax file: it has all four enzymes, the TDI file has
    # only 2D6/3A4, and sourcing one column name from two frames would collide.
    _require(emax, [*AUX_EMAX, *AUX_EMAX_DIRECT, *AUX_IS_TDI], "TRAIN_Emax")
    emax_small = emax[["SMILES", *AUX_EMAX, *AUX_EMAX_DIRECT, *AUX_IS_TDI]].copy()
    for c in AUX_IS_TDI:
        # Stored as object-dtype Python bools; map explicitly so an unexpected
        # value becomes NaN (and is masked out) rather than silently casting.
        emax_small[c] = emax_small[c].map({True: 1.0, False: 0.0})

    merged = (
        base.merge(tdi_small, on="SMILES", how="outer")
        .merge(emax_small, on="SMILES", how="outer")
        .merge(single_w, on="SMILES", how="outer")
    )
    collided = [c for c in merged.columns if c.endswith(("_x", "_y"))]
    if collided:
        raise KeyError(f"merge collision produced suffixed columns: {collided}")

    std = standardise_many(merged["SMILES"].astype(str).tolist(), workers=workers)
    merged["key"] = [s.inchikey if s.ok else "" for s in std]
    merged["smiles"] = [s.smiles if s.ok else "" for s in std]
    merged = merged[merged["key"] != ""]
    # Collapse duplicates across files on standardised identity.
    value_cols = [c for c in merged.columns if c not in ("SMILES", "key", "smiles")]
    merged = merged.groupby("key", as_index=False).agg(
        {**{c: "mean" for c in value_cols}, "smiles": "first"})

    std_t = standardise_many(test["SMILES"].astype(str).tolist(), workers=workers)
    test = test.assign(
        key=[s.inchikey if s.ok else "" for s in std_t],
        smiles=[s.smiles if s.ok else t for s, t in zip(std_t, test["SMILES"])],
    )

    overlap = set(merged["key"]) & set(test["key"])
    if overlap:
        # Not an error: the organisers may legitimately include known compounds.
        # It is reported because a silent overlap would inflate our own validation.
        print(f"[cyp] NOTE: {len(overlap)} test molecules also appear in training files")

    # Declared, not inferred: the head list is a contract the loader checks.
    _require(merged, AUXILIARY, "merged frame")
    auxiliary = list(AUXILIARY)

    print(f"[cyp] {len(merged)} unique training molecules, "
          f"{len(PRIMARY)} primary heads, {len(auxiliary)} auxiliary heads")
    n_primary = 0
    for c in PRIMARY:
        n = int(merged[c].notna().sum())
        n_primary += n
        print(f"    {c}: {n} labels")
    # Per-head auxiliary counts, because a silently empty head is the defect this
    # loader was rewritten to prevent and a total alone would hide one.
    for c in auxiliary:
        n = int(merged[c].notna().sum())
        if n == 0:
            raise ValueError(f"auxiliary head '{c}' has no labels after the merge")
        print(f"    aux {c}: {n} labels")
    print(f"[cyp] primary labels {n_primary}; "
          f"auxiliary labels {int(merged[auxiliary].notna().sum().sum())}")
    print(f"[cyp] test: {len(test)} molecules")

    return CypData(frame=merged, primary=PRIMARY, auxiliary=auxiliary, test=test)
