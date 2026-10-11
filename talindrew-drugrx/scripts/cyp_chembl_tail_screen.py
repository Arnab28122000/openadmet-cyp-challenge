r"""Screen ChEMBL for the CYP challenge on the axis that actually matters: the TAIL.

The tail analysis settled what new data has to supply. The most-active quintile
carries **45.8% of the ST-RAE numerator on 20% of the molecules**, 94.8% of it
outside its interval, with interval width collapsing 1.683 -> 0.217 as activity
rises. So errors on actives are charged nearly in full while errors on inactives
are largely forgiven, and our model's bias is -0.818 exactly where that is most
expensive.

That makes **structural** coverage the wrong screen, which is how the previous two
acquisition decisions were argued:

* Veith was declined at 0.0% incremental structural coverage. The decisive reason is
  worse than that: its labels are **binary 0/1**, so it cannot distinguish pIC50 6
  from pIC50 9 and cannot inform a regression tail at all.
* ChEMBL was never screened for CYP. (The 11.2%/6.4% decline on record was
  `chembl_screen.py` against TDC `half_life`; `chembl_audit.py` covered TDC ADME.
  Different datasets, earlier phase.)

ChEMBL does carry continuous CYP IC50 values, so it is the first candidate that can
even be judged on this axis.

**Pre-registered criterion, fixed before any number is seen.** Acquire only if the
source adds at least **750 compounds** with measured CYP pIC50 >= 5.5 that are
*not* already in the challenge training set by standardised InChIKey, for at least
two of the four isoforms. 750 is one test-set's worth and roughly a 60% increase on
the ~1,250 tail compounds the challenge training set holds; anything smaller is a
sub-resolution effect by the same arithmetic that made ExpansionRx's 18% one.

Also reported, because an acquisition can fail for either reason:
* **tail count and tail fraction** per isoform -- can the source speak about actives?
* **novel tail count** -- after removing compounds we already hold.
* **assay heterogeneity** -- ChEMBL pools many assays and a pIC50 from a different
  substrate, microsome lot or readout is not interchangeable with the challenge's.
  The spread of values *within* a repeated compound is the honest estimate of that
  noise floor, and it is reported rather than assumed small.
* **test-set identity overlap** -- must be reported, not because external data is
  disallowed (the rules permit it explicitly) but because a shared compound carries
  no transferable label across assays and silently treating it as one would repeat
  the auxiliary-column leak.
"""
from __future__ import annotations

import argparse, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load          # noqa: E402

# Verified against the ChEMBL target API at run time rather than trusted from memory.
CYP_TARGETS = {
    "CYP1A2": "CHEMBL3356",
    "CYP2C9": "CHEMBL3397",
    "CYP2D6": "CHEMBL289",
    "CYP3A4": "CHEMBL340",
}
BASE = "https://www.ebi.ac.uk/chembl/api/data"
TAIL_PIC50 = 5.5
PREREG_MIN_NOVEL_TAIL = 750
PREREG_MIN_ISOFORMS = 2


def get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "drugrx-screen/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if a == tries - 1:
                raise
            time.sleep(2 * (a + 1))
    return None


def verify_target(chembl_id):
    d = get(f"{BASE}/target/{chembl_id}.json")
    return d.get("pref_name"), d.get("organism"), d.get("target_type")


def fetch_activities(chembl_id, limit=20000):
    """IC50 activities against one target, in pChEMBL form where available."""
    rows, offset = [], 0
    while offset < limit:
        q = urllib.parse.urlencode({
            "target_chembl_id": chembl_id, "standard_type": "IC50",
            "limit": 1000, "offset": offset,
        })
        d = get(f"{BASE}/activity.json?{q}")
        acts = d.get("activities", [])
        if not acts:
            break
        for a in acts:
            rows.append({
                "molecule_chembl_id": a.get("molecule_chembl_id"),
                "canonical_smiles": a.get("canonical_smiles"),
                "pchembl_value": a.get("pchembl_value"),
                "assay_chembl_id": a.get("assay_chembl_id"),
                "standard_relation": a.get("standard_relation"),
            })
        offset += 1000
        if not d.get("page_meta", {}).get("next"):
            break
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--out", default="/content/out/cyp_chembl/screen.json")
    args = ap.parse_args()

    print("[chembl] verifying target ids live rather than trusting them", flush=True)
    for name, cid in CYP_TARGETS.items():
        try:
            pref, org, tt = verify_target(cid)
            print(f"    {name:7s} {cid:12s} {pref} | {org} | {tt}", flush=True)
        except Exception as e:
            print(f"    {name:7s} {cid:12s} VERIFY FAILED: {e}", flush=True)

    data = load(workers=args.workers)
    frame, test = data.frame, data.test
    from drugrx.data.standardise import standardise_many

    def keys(sm):
        return {r.inchikey for r in standardise_many(list(sm), workers=args.workers)
                if r.ok and r.inchikey}

    train_keys = keys(frame["smiles"].tolist())
    test_keys = keys(test["SMILES"].tolist())
    print(f"\n[chembl] challenge train keys {len(train_keys)}, test keys {len(test_keys)}",
          flush=True)

    # How much tail does the challenge training set already hold? The denominator
    # the pre-registered threshold is relative to.
    held_tail = {}
    for c in PRIMARY:
        y = frame[c].to_numpy(float)
        held_tail[c] = int(np.sum(np.isfinite(y) & (y >= TAIL_PIC50)))
    print(f"[chembl] challenge training compounds at pIC50 >= {TAIL_PIC50}: "
          f"{held_tail} (total {sum(held_tail.values())})", flush=True)

    report = {"targets": CYP_TARGETS, "tail_threshold": TAIL_PIC50,
              "prereg_min_novel_tail": PREREG_MIN_NOVEL_TAIL,
              "prereg_min_isoforms": PREREG_MIN_ISOFORMS,
              "challenge_held_tail": held_tail, "per_isoform": {}}

    isoforms_passing = 0
    for name, cid in CYP_TARGETS.items():
        try:
            df = fetch_activities(cid, args.limit)
        except Exception as e:
            print(f"\n[chembl] {name}: fetch failed: {e}", flush=True)
            report["per_isoform"][name] = {"error": str(e)}
            continue
        if df.empty:
            print(f"\n[chembl] {name}: no activities returned", flush=True)
            report["per_isoform"][name] = {"n": 0}
            continue
        df["p"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        # Censored measurements ('>' / '<') cannot be read as point values.
        exact = df[df["standard_relation"].isin(["=", None])]
        d = exact.dropna(subset=["p", "canonical_smiles"])
        tail = d[d["p"] >= TAIL_PIC50]

        tk = keys(tail["canonical_smiles"].unique())
        novel = tk - train_keys
        overlap_test = len(tk & test_keys)

        # Assay heterogeneity: spread within repeated compounds.
        rep = d.groupby("molecule_chembl_id")["p"].agg(["count", "std"])
        rep = rep[rep["count"] > 1]
        het = float(rep["std"].median()) if len(rep) else float("nan")

        ok = len(novel) >= PREREG_MIN_NOVEL_TAIL
        isoforms_passing += int(ok)
        report["per_isoform"][name] = {
            "activities": int(len(df)), "exact_with_pchembl": int(len(d)),
            "n_assays": int(d["assay_chembl_id"].nunique()),
            "tail_rows": int(len(tail)), "tail_unique_keys": len(tk),
            "novel_tail_keys": len(novel), "overlap_with_test": overlap_test,
            "median_within_compound_sd": het, "passes": bool(ok),
        }
        print(f"\n[chembl] {name}: {len(df)} IC50 rows, {len(d)} exact with pChEMBL, "
              f"{d['assay_chembl_id'].nunique()} distinct assays", flush=True)
        print(f"    tail (pIC50 >= {TAIL_PIC50}): {len(tail)} rows, {len(tk)} unique "
              f"compounds, {len(novel)} NOT already held  -> "
              f"{'passes' if ok else 'below bar'}", flush=True)
        print(f"    identity overlap with the 750 test compounds: {overlap_test}", flush=True)
        print(f"    assay heterogeneity: median within-compound sd of pIC50 = {het:.3f} "
              f"over {len(rep)} repeated compounds", flush=True)

    verdict = "ACQUIRE" if isoforms_passing >= PREREG_MIN_ISOFORMS else "DECLINE"
    report["isoforms_passing"] = isoforms_passing
    report["verdict"] = verdict
    print(f"\n[chembl] {isoforms_passing} of 4 isoforms clear {PREREG_MIN_NOVEL_TAIL} novel "
          f"tail compounds; rule needs {PREREG_MIN_ISOFORMS} -> **{verdict}**", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
