r"""Merge the external sources into one auxiliary table, keyed on standardised InChIKey.

Two continuous sources now exist and both are auxiliary heads:

* **Veith** (PubChem qHTS) -- `veith_*`, **binary 0/1**. Retained because it was
  measured as a small positive (+0.0029 +/- 0.0032, 5/5 seeds, under the old
  metric and due a re-measurement), but it cannot inform the active tail: a
  classification label cannot distinguish pIC50 6 from 9.
* **ChEMBL** -- `chembl_*`, continuous pIC50, 9,045 compounds, 4,902 of them in the
  tail. Acquired against a pre-registered tail criterion.

The merge is on **standardised InChIKey**, not SMILES. A SMILES join once matched 21
of 420 molecules in this programme, and salt forms once hid 425 train/test
collisions; the two sources were pulled independently and will not agree on
canonical form.

Columns are kept side by side rather than combined. The two sources measure
different things on different scales -- one binary, one a log-molar potency carrying
an unmeasured cross-assay spread -- so merging them into one column would assert a
relationship instead of letting the model learn one.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veith", default="/content/out/cyp/external.parquet")
    ap.add_argument("--chembl", default="/content/out/cyp/chembl_aux.parquet")
    ap.add_argument("--out", default="/content/out/cyp/external_merged.parquet")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from drugrx.data.standardise import standardise_many

    frames = []
    v = pd.read_parquet(args.veith)
    vc = [c for c in v.columns if c.startswith("veith_")]
    res = standardise_many(v["smiles"].tolist(), workers=args.workers)
    v = v.assign(key=[r.inchikey if r.ok else None for r in res],
                 cano=[r.smiles if r.ok else None for r in res]).dropna(subset=["key"])
    v = v.groupby("key").agg({**{c: "median" for c in vc}, "cano": "first"})
    print(f"[merge] Veith: {len(v)} unique keys, {len(vc)} columns (binary)", flush=True)

    c = pd.read_parquet(args.chembl)
    cc = [x for x in c.columns if x.startswith("chembl_")]
    if "inchikey" in c.columns:
        c = c.set_index("inchikey")
        c = c.groupby(level=0).agg({**{x: "median" for x in cc}, "smiles": "first"})
        c = c.rename(columns={"smiles": "cano"})
    print(f"[merge] ChEMBL: {len(c)} unique keys, {len(cc)} columns (continuous pIC50)",
          flush=True)

    out = v.join(c, how="outer", lsuffix="_v", rsuffix="_c")
    out["smiles"] = out["cano_v"].fillna(out["cano_c"]) if "cano_v" in out else out["cano"]
    out = out.drop(columns=[x for x in ("cano_v", "cano_c", "cano") if x in out])
    out = out.reset_index(names="inchikey").dropna(subset=["smiles"])

    keep = [x for x in out.columns if x.startswith(("veith_", "chembl_"))]
    print(f"\n[merge] {len(out)} compounds; label counts per head:", flush=True)
    for x in keep:
        n = int(out[x].notna().sum())
        tail = int((out[x] >= 5.5).sum()) if x.startswith("chembl_") else None
        print(f"    {x:22s} {n:6d}" + (f"   tail(>=5.5) {tail}" if tail is not None else ""))
    both = int((out[[x for x in keep if x.startswith('veith_')]].notna().any(axis=1) &
                out[[x for x in keep if x.startswith('chembl_')]].notna().any(axis=1)).sum())
    print(f"[merge] compounds carrying BOTH sources: {both}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[merge] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
