r"""Measure cross-assay heterogeneity properly, replacing a statistic that measured nothing.

The tail screen reported "median within-compound sd of pIC50 = 0.000" for CYP1A2
and CYP2D6 and offered it as evidence that ChEMBL's assays are interchangeable with
the challenge's. That was wrong: a median of *exactly* zero over 78 and 247
compounds means the repeated rows are **duplicate deposit records**, not independent
measurements, so the statistic measured record duplication rather than assay noise.

The honest estimator groups by compound and then **requires the measurements to come
from distinct `assay_chembl_id`s** before computing a spread. The build step found
61 / 154 / 184 / 703 compounds qualifying, so this is answerable for CYP3A4 and
indicative for the rest.

Why the number matters: it is the noise floor on any ChEMBL-derived label. If the
same compound varies by ~0.5 pIC50 between assays, a ChEMBL head cannot be expected
to transfer at finer resolution than that, and the per-endpoint gains we would be
chasing (~0.02 ST-RAE) need to be read against it.
"""
from __future__ import annotations

import argparse, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CYP_TARGETS = {"CYP1A2": "CHEMBL3356", "CYP2C9": "CHEMBL3397",
               "CYP2D6": "CHEMBL289", "CYP3A4": "CHEMBL340"}
BASE = "https://www.ebi.ac.uk/chembl/api/data"


def get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "drugrx-het/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(2 * (a + 1))


def fetch(cid, limit=30000):
    rows, off = [], 0
    while off < limit:
        q = urllib.parse.urlencode({"target_chembl_id": cid, "standard_type": "IC50",
                                    "limit": 1000, "offset": off})
        d = get(f"{BASE}/activity.json?{q}")
        acts = d.get("activities", [])
        if not acts:
            break
        for a in acts:
            rows.append({"mol": a.get("molecule_chembl_id"),
                         "p": a.get("pchembl_value"),
                         "rel": a.get("standard_relation"),
                         "assay": a.get("assay_chembl_id")})
        off += 1000
        if not d.get("page_meta", {}).get("next"):
            break
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp/chembl_heterogeneity.json")
    args = ap.parse_args()

    report = {}
    print("\n[het] spread of the SAME compound across DISTINCT assays", flush=True)
    print(f"    {'isoform':8s} {'compounds':>10s} {'median sd':>10s} {'mean sd':>9s} "
          f"{'p90 sd':>8s} {'median range':>13s} {'>0.5':>7s} {'>1.0':>7s}", flush=True)
    for name, cid in CYP_TARGETS.items():
        df = fetch(cid)
        df["pv"] = pd.to_numeric(df["p"], errors="coerce")
        df = df[df["rel"].isin(["=", None])].dropna(subset=["pv", "mol", "assay"])
        # One value per (compound, assay): collapse duplicate records FIRST, so
        # record duplication cannot masquerade as agreement.
        per = df.groupby(["mol", "assay"])["pv"].median().reset_index()
        g = per.groupby("mol")["pv"]
        cnt = g.size()
        multi = cnt[cnt > 1].index
        if len(multi) == 0:
            print(f"    {name:8s} {0:10d} {'n/a':>10s}", flush=True)
            report[name] = {"compounds": 0}
            continue
        sub = per[per["mol"].isin(multi)]
        sd = sub.groupby("mol")["pv"].std(ddof=1).dropna()
        rng = sub.groupby("mol")["pv"].agg(lambda v: v.max() - v.min())
        report[name] = {
            "compounds_multi_assay": int(len(multi)),
            "median_sd": float(sd.median()), "mean_sd": float(sd.mean()),
            "p90_sd": float(np.percentile(sd, 90)),
            "median_range": float(rng.median()),
            "frac_range_gt_0.5": float((rng > 0.5).mean()),
            "frac_range_gt_1.0": float((rng > 1.0).mean()),
        }
        r = report[name]
        print(f"    {name:8s} {len(multi):10d} {r['median_sd']:10.3f} "
              f"{r['mean_sd']:9.3f} {r['p90_sd']:8.3f} {r['median_range']:13.3f} "
              f"{r['frac_range_gt_0.5']:7.1%} {r['frac_range_gt_1.0']:7.1%}", flush=True)

    vals = [v["median_sd"] for v in report.values() if "median_sd" in v]
    if vals:
        print(f"\n[het] noise floor on a ChEMBL-derived label: median cross-assay sd "
              f"~{np.mean(vals):.3f} pIC50", flush=True)
        print(f"[het] read any ChEMBL-driven gain against this: the effects we chase are "
              f"~0.02 ST-RAE, and a label carrying {np.mean(vals):.2f} log units of "
              f"assay noise supports a head, not a pooled primary label.", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
