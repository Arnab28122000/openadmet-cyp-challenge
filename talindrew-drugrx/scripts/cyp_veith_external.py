r"""A drop-in continuous replacement for `external.parquet`. Runs on Colab.

`cyp_net_one.py` -- the incumbent trainer -- already reads `external.parquet`, turns every
`veith_*` column into an auxiliary head, adds those compounds as extra **rows**, and weights them
at **0.3**. That last number is worth noting: it is the same weight the `rows-w` sweep found
optimal independently.

So the cheapest incumbent-level test of the Veith defect needs **no trainer change at all** --
only a parquet with the same schema (`key`, `smiles`, `veith_CYP1A2..CYP3A4`) carrying
**continuous** values where the current file carries a 0/1 inhibitor flag.

## The trap in a naive swap, and the fix

The binary file has **12,045-13,084 labels per isoform** because it scores every screened
compound, inactives included. The continuous rebuild has only **5,462-9,625**, because an
inactive compound has no AC50 to report -- the curve never crossed. A naive swap would therefore
buy potency resolution on the actives and **throw away the inactive majority**, which is most of
the file.

That is avoidable, because "no AC50" is not missing data -- it is a **censored measurement**. The
screen runs 15 points to a top concentration of **57.14 uM**, so an inactive means

    pAC50 < 6 - log10(57.14) = 4.243

and the honest label is the censoring limit, not a blank. Assigning it keeps every row the binary
file had *and* adds potency resolution on the actives, so the result is **strictly more
informative than the flag** rather than a trade.

`PUBCHEM_ACTIVITY_OUTCOME` distinguishes Inactive from Inconclusive; only Inactive is censored at
the limit. Inconclusive stays missing, because it genuinely is.

Three arms become available from this one file, and the trainer needs editing for none of them:

* `--external external.parquet` -- the incumbent, binary.
* `--external external_continuous.parquet` -- continuous + censored inactives.
* `--aux-prefix none` -- no auxiliary heads, the control the trainer already supports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOP_CONC_UM = 57.14                       # highest point of the 15-point series
CENSOR_PAC50 = 6.0 - np.log10(TOP_CONC_UM)
PANELS = {"p450-cyp1a2": "CYP1A2", "p450-cyp2c9": "CYP2C9",
          "p450-cyp2d6": "CYP2D6", "p450-cyp3a4": "CYP3A4"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default="/content/out/veith/aid1851.parquet")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/cyp/external_continuous.parquet")
    args = ap.parse_args()

    from drugrx.cyp.data import load
    from drugrx.data.standardise import standardise_many

    print(f"[ext] censoring limit: pAC50 < {CENSOR_PAC50:.3f} "
          f"(top concentration {TOP_CONC_UM} uM)")
    df = pd.read_parquet(args.aid)
    df = df[pd.to_numeric(df["PUBCHEM_SID"], errors="coerce").notna()].copy()
    df["iso"] = df["Panel Name"].map(PANELS)
    df = df[df["iso"].notna() & df["PUBCHEM_EXT_DATASOURCE_SMILES"].notna()].copy()
    df["potency_uM"] = pd.to_numeric(df["Potency"], errors="coerce")
    outcome = df["PUBCHEM_ACTIVITY_OUTCOME"].astype(str).str.lower()

    print(f"[ext] outcomes: "
          f"{df['PUBCHEM_ACTIVITY_OUTCOME'].astype(str).value_counts().to_dict()}")

    active = df["potency_uM"].notna() & (df["potency_uM"] > 0)
    inactive = (~active) & outcome.str.startswith("inactive")
    df["pac50"] = np.nan
    df.loc[active, "pac50"] = 6.0 - np.log10(df.loc[active, "potency_uM"])
    df.loc[inactive, "pac50"] = CENSOR_PAC50          # censored, not missing
    df["censored"] = inactive
    df = df[df["pac50"].notna()]
    print(f"[ext] {len(df)} measurements: {int(active.sum())} with an AC50, "
          f"{int(inactive.sum())} censored inactive, "
          f"{int((~active & ~inactive).sum())} inconclusive and dropped")

    std = standardise_many(df["PUBCHEM_EXT_DATASOURCE_SMILES"].astype(str).tolist(),
                           workers=args.workers)
    df["key"] = [s.inchikey if s.ok else None for s in std]
    df["smiles"] = [s.smiles if s.ok else r
                    for s, r in zip(std, df["PUBCHEM_EXT_DATASOURCE_SMILES"])]
    df = df[df["key"].notna()]

    wide = (df.pivot_table(index="key", columns="iso", values="pac50", aggfunc="median")
              .rename(columns={v: f"veith_{v}" for v in PANELS.values()}))
    wide = wide.join(df.groupby("key")["smiles"].first()).reset_index()

    data = load(workers=args.workers)
    te = set(data.test["key"])
    before = len(wide)
    wide = wide[~wide["key"].isin(te)]
    print(f"[ext] dropped {before - len(wide)} compounds colliding with the 750 blinded test")

    cols = ["key", "smiles"] + [f"veith_{v}" for v in PANELS.values()
                                if f"veith_{v}" in wide.columns]
    wide = wide[cols]
    print(f"\n[ext] {len(wide)} compounds. Per isoform, against the binary file it replaces:")
    binary_n = {"CYP1A2": 12546, "CYP2C9": 12045, "CYP2D6": 13084, "CYP3A4": 12290}
    for v in PANELS.values():
        c = f"veith_{v}"
        if c not in wide.columns:
            continue
        s = wide[c].dropna()
        at_limit = int((s <= CENSOR_PAC50 + 1e-9).sum())
        print(f"   {c}: n={len(s):6d} (binary had {binary_n[v]:6d})  "
              f"{at_limit:6d} at the censoring limit, {len(s)-at_limit:6d} with a real AC50  "
              f"mean {s.mean():.3f} sd {s.std():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(args.out)
    (Path(args.out).parent / "external_continuous_report.json").write_text(json.dumps(
        dict(censor_pac50=float(CENSOR_PAC50), n=int(len(wide)),
             per_isoform={v: int(wide[f"veith_{v}"].notna().sum())
                          for v in PANELS.values() if f"veith_{v}" in wide.columns}),
        indent=1))
    print(f"\n[ext] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
