r"""Rebuild the Veith external set with its CONTINUOUS potency, not TDC's binary flag. Runs on Colab.

This programme has used Veith since its first external arm, through the HF mirrors
`scikit-fingerprints/TDC_cyp{1a2,2c9,2d6,3a4}_veith`, which are **TDC's binarised** form: a 0/1
inhibitor call. The source, **PubChem AID 1851** (Veith 2009, doi:10.1038/nbt.1581), reports
**AC50 in µM** from a 15-point dose-response over 0.7 nM-57 µM on recombinant BD Gentest CYPs
with P450-Glo luciferin substrates in 1536-well plates.

Compounds carrying an AC50, against what the challenge gives us:

| endpoint | challenge | Veith AC50 | ratio |
|---|---|---|---|
| CYP1A2 | 1,412 | 9,655 | 6.8x |
| CYP2C9 | 1,285 | 7,262 | 5.7x |
| **CYP2D6** | **1,493** | **5,487** | **3.7x** |
| CYP3A4 | 2,335 | 8,654 | 3.7x |

**The recorded Veith result (+0.0029 +/- 0.0032, "small and never harmful") was measured on the
binary label.** A binary inhibitor flag on a compound whose AC50 is known discards the entire
potency axis, which is the axis we are scored on, so that measurement carries no information about
what the continuous label is worth. This is not a new acquisition; it is a defect in how an
existing source was read.

## What this script produces, and the one number that decides whether it is usable

A parquet of standardised-InChIKey -> per-isoform **pAC50** (`pAC50 = 6 - log10(AC50_uM)`), plus:

1. **Per-isoform counts**, and coverage of the challenge training compounds.
2. **Test contamination**, by standardised InChIKey against the 750 blinded compounds. The audit
   of the binarised copy found a single collision (`OCNT-0496981`); it must be dropped, and the
   check re-run here rather than trusted.
3. **The assay offset, measured directly.** Roughly 111 compounds carry *both* a Veith label and a
   challenge direct-inhibition pIC50. On those, `challenge - veith` per isoform is an estimate of
   the offset between a P450-Glo luciferin readout on recombinant enzyme and the challenge's
   fluorogenic Supersome assay. **A cross-lab offset is what killed the ExpansionRx `cl_micro`
   merge**, so it is estimated and reported before any training run, not absorbed into a free
   parameter and hoped about. If the offset's *spread* across those compounds is large relative to
   the label range, pooling rows will not work no matter what the mean offset is -- and that is
   the number to look at, not the mean.

Everything runs on the VM: PubChem bulk is fetched there, not locally.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AID = 1851
BULK = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/0001001_0002000.zip"

# AID 1851 stacks FIVE isoform panels in one table -- 85,715 data rows over 16,560 CIDs, five
# rows each -- discriminated by `Panel Name`. Discovered from the file, not assumed: the first
# version of this script looked for per-isoform AC50 *columns* and found none, because there is
# one `Potency` column and a panel label beside it.
PANEL_COL = "Panel Name"
PANELS = {"p450-cyp1a2": "CYP1A2", "p450-cyp2c9": "CYP2C9",
          "p450-cyp2d6": "CYP2D6", "p450-cyp3a4": "CYP3A4"}
# CYP2C19 (`p450-cyp2c19`) is in the file and is deliberately not mapped: the challenge does not
# score it, so carrying it would only add an unscored head.
SMILES_COL = "PUBCHEM_EXT_DATASOURCE_SMILES"     # SMILES ship with the assay; no PUG-REST needed


def fetch_aid_csv(cache: Path) -> pd.DataFrame:
    """AID 1851's data table, from the PubChem bulk archive (1.3 GB zip, 21 s on the VM)."""
    import requests
    if cache.exists():
        print(f"[veith] cache hit {cache}", flush=True)
        return pd.read_parquet(cache)
    t0 = time.time()
    print(f"[veith] downloading {BULK}", flush=True)
    r = requests.get(BULK, timeout=1800)
    r.raise_for_status()
    print(f"[veith] {len(r.content)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if n.endswith(f"{AID}.csv.gz")
                 or n.endswith(f"{AID}.csv")]
        if not names:
            raise SystemExit(f"AID {AID} not in {BULK}")
        raw = z.read(names[0])
        print(f"[veith] member {names[0]}", flush=True)
    if names[0].endswith(".gz"):
        raw = gzip.decompress(raw)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.to_parquet(cache)
    print(f"[veith] AID {AID}: {df.shape}", flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/veith")
    args = ap.parse_args()

    from drugrx.cyp.data import CYPS, PRIMARY, load
    from drugrx.data.standardise import standardise_many

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = fetch_aid_csv(out / "aid1851.parquet")

    # PubChem assay CSVs carry four metadata rows (RESULT_TYPE / RESULT_DESCR / RESULT_UNIT /
    # RESULT_ATTR_CONC_MICROMOL) before the data. They parse as ordinary rows and would otherwise
    # become compounds.
    df = df[pd.to_numeric(df["PUBCHEM_SID"], errors="coerce").notna()].copy()
    print(f"[veith] {len(df)} data rows, {df['PUBCHEM_CID'].nunique()} CIDs, "
          f"panels {sorted(df[PANEL_COL].dropna().unique())}")

    df["potency_uM"] = pd.to_numeric(df["Potency"], errors="coerce")
    df["iso"] = df[PANEL_COL].map(PANELS)
    df = df[df["iso"].notna() & df["potency_uM"].notna() & (df["potency_uM"] > 0)
            & df[SMILES_COL].notna()].copy()
    print(f"[veith] rows with a mapped panel AND an AC50: {len(df)}")
    print(f"[veith] curve classes (a quality handle the binarised mirror discards):")
    for k, v in df["Curve_Description"].astype(str).value_counts().head(8).items():
        print(f"     {v:6d}  {k}")

    std = standardise_many(df[SMILES_COL].astype(str).tolist(), workers=args.workers)
    df["key"] = [x.inchikey if x.ok else None for x in std]   # full InChIKey; the
    # skeleton is an audit key and must never be a merge key (standardise.py says so)
    df["smiles"] = [x.smiles if x.ok else r for x, r in zip(std, df[SMILES_COL])]
    df = df[df["key"].notna()]
    df["pAC50"] = 6.0 - np.log10(df["potency_uM"])

    agg = (df.pivot_table(index="key", columns="iso", values="pAC50", aggfunc="median")
             .rename(columns={k: f"veithc_{k}" for k in PANELS.values()}))
    agg = agg.join(df.groupby("key")["smiles"].first()).reset_index()
    print(f"\n[veith] {len(agg)} unique standardised compounds")
    for iso in PANELS.values():
        c = f"veithc_{iso}"
        if c in agg:
            s_ = agg[c].dropna()
            print(f"   {c}: n={len(s_)}  pAC50 mean {s_.mean():.3f} sd {s_.std():.3f}  "
                  f"range {s_.min():.2f}-{s_.max():.2f}")

    data = load(workers=args.workers)
    fr, test = data.frame, data.test
    tr_keys = set(fr["key"]); te_keys = set(test["key"])
    agg["in_train"] = agg["key"].isin(tr_keys)
    agg["in_test"] = agg["key"].isin(te_keys)
    print(f"\n[veith] overlap: {int(agg['in_train'].sum())} in challenge TRAIN, "
          f"**{int(agg['in_test'].sum())} in the 750 blinded TEST**")
    if int(agg["in_test"].sum()):
        print(f"   dropping test collisions: "
              f"{agg.loc[agg['in_test'], 'key'].tolist()[:5]}")
    agg = agg[~agg["in_test"]]

    # THE number that decides usability: the offset, and more importantly its spread.
    print(f"\n[veith] assay offset against the challenge, on compounds carrying BOTH labels")
    print(f"{'endpoint':<10s} {'n':>5s} {'mean(chal-veith)':>17s} {'sd':>7s} "
          f"{'median':>8s} {'Pearson r':>10s}")
    offsets = {}
    idx = {k: i for i, k in enumerate(fr["key"])}
    for iso in PANELS.values():
        col = f"{iso}_pIC50_direct_inhibition"
        if col not in fr.columns:
            continue
        if f"veithc_{iso}" not in agg:
            continue
        rows = [(idx[k], v) for k, v in zip(agg["key"], agg[f"veithc_{iso}"])
                if k in idx and np.isfinite(v)]
        if not rows:
            continue
        ii = np.array([r[0] for r in rows]); vv = np.array([r[1] for r in rows])
        ch = fr[col].to_numpy(float)[ii]
        m = np.isfinite(ch)
        if m.sum() < 8:
            print(f"{iso:<10s} {int(m.sum()):5d}   too few paired compounds")
            continue
        d = ch[m] - vv[m]
        r = float(np.corrcoef(ch[m], vv[m])[0, 1]) if m.sum() > 2 else float("nan")
        offsets[iso] = dict(n=int(m.sum()), mean=float(d.mean()), sd=float(d.std(ddof=1)),
                            median=float(np.median(d)), r=r)
        print(f"{iso:<10s} {int(m.sum()):5d} {d.mean():17.3f} {d.std(ddof=1):7.3f} "
              f"{np.median(d):8.3f} {r:10.3f}")
    print("  **Read the sd, not the mean.** A constant offset is absorbable by one fitted term;")
    print("  a large spread means the two assays disagree per-compound and pooling rows cannot")
    print("  work whatever the mean is. The `cl_micro` merge failed on exactly this.")

    path = out / "veith_continuous.parquet"
    agg.to_parquet(path)
    (out / "report.json").write_text(json.dumps(
        dict(aid=AID, panels=PANELS, n_unique=int(len(agg)),
             per_isoform={i: int(agg[f'veithc_{i}'].notna().sum())
                          for i in PANELS.values() if f'veithc_{i}' in agg},
             in_train=int(agg["in_train"].sum()), offsets=offsets), indent=1, default=float))
    print(f"\n[veith] wrote {path} and {out/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
