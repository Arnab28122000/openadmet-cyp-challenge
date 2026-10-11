r"""Build the approved ChEMBL acquisition as AUXILIARY heads, keyed on InChIKey.

The screen approved this on a pre-registered tail criterion: the challenge training
set holds only **960** (compound, isoform) measurements at pIC50 >= 5.5 across all
four isoforms, and ChEMBL adds **4,975** novel ones -- a 5.2x increase in exactly
the region carrying 45.8% of the ST-RAE numerator.

Both of those are counts of **(compound, isoform) pairs, not of compounds**, and
until 2026-10-05 both were described as compounds. Verified on the rebuild: the
challenge's 960 pairs are 853 distinct compounds, and ChEMBL's 4,902 pairs
(742 + 913 + 1,139 + 2,108) are 4,329. **The criterion is unaffected** -- the two
sides were counted the same way, so the ratio is 5.1x by pairs and 5.1x by
compounds. Recorded because the natural reaction to "4,902 is really 4,329" is to
assume the acquisition was approved on an inflated figure, and it was not.

**Auxiliary heads, not pooled primary labels.** ChEMBL spreads these measurements
over 478-1,545 distinct assays per isoform, with different substrates, microsome
lots and readouts, and the screen's heterogeneity statistic turned out not to
measure that: a median within-compound sd of exactly 0.000 over hundreds of
compounds means the repeats are duplicate deposit records rather than independent
measurements. Cross-assay noise is therefore **unmeasured**, so pooling ChEMBL
pIC50 with the challenge's as if interchangeable would be an unquantified label
merge. As a separate head per isoform the model can learn the relationship instead
of having one asserted, which is the same arrangement that already handles the
TDI-condition and Emax columns.

Three safeguards, each the residue of a defect this programme has already paid for:

* **Identity is standardised InChIKey, stereo preserved.** A SMILES join once
  matched 21 of 420 molecules, and salt forms once hid 425 train/test collisions.
* **Censored values are dropped.** A `>` or `<` relation is not a point estimate;
  keeping them would import a bound as a measurement.
* **Test-set collisions are removed even though the screen found none**, because
  the check must live in the pipeline rather than in a one-off audit, and because
  a ChEMBL pIC50 for a test compound carries no transferable label across assays
  anyway -- treating one as a label would repeat the auxiliary-column leak that
  produced a spurious 7x gain.

Duplicates within ChEMBL are resolved by **median** per (compound, isoform): the
mean is pulled by the censored-looking extremes that survive filtering, and this
programme has already learned that averaging the wrong statistic over duplicates
can produce a value no measurement ever took.
"""
from __future__ import annotations

import argparse, json, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load          # noqa: E402

CYP_TARGETS = {"CYP1A2": "CHEMBL3356", "CYP2C9": "CHEMBL3397",
               "CYP2D6": "CHEMBL289", "CYP3A4": "CHEMBL340"}
BASE = "https://www.ebi.ac.uk/chembl/api/data"


def get(url, tries=7):
    """GET with exponential backoff.

    The first run of this build died on `HTTP 500` from ChEMBL's activity
    endpoint after CYP1A2 had already been fetched. The old schedule was four
    tries at 2/4/6 s -- twelve seconds in total, which is not a retry policy for
    a public API that 500s in bursts. Exponential to ~2 min instead, and 5xx and
    timeouts are retried while a 404 is not, because retrying a genuine
    "no such resource" only wastes the budget.
    """
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "drugrx-build/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or a == tries - 1:
                raise
            print(f"[build] HTTP {exc.code}; retry {a + 1}/{tries - 1}", flush=True)
            time.sleep(min(2 ** (a + 1), 120))
        except Exception as exc:  # noqa: BLE001 — timeouts and resets are transient
            if a == tries - 1:
                raise
            print(f"[build] {type(exc).__name__}; retry {a + 1}/{tries - 1}", flush=True)
            time.sleep(min(2 ** (a + 1), 120))
    return None


def fetch(chembl_id, limit, page=200, floor=25):
    """Page through the activity endpoint, halving the page on failure.

    The page size was a hard-coded 1000 and that is past what this endpoint will
    serve reliably. Probed 2026-10-05 against CHEMBL3397 (7,520 activities):
    `limit=1` and `limit=20` return promptly, `limit=1000` raises TimeoutError,
    and a server-side timeout here also surfaces as HTTP 500 — which is how this
    build died twice with nothing but a 500 to go on. The endpoint 500s
    intermittently on *small* pages too, so both mechanisms are real and the
    retry in `get` stays.

    Halve-and-retry rather than a fixed smaller page, for the same reason the
    trainer sizes its batch that way: the limit is a property of the server's
    current load, not something to predict. Shrinking is sticky — once a smaller
    page works we keep it, because the next page is no more likely to be served.
    """
    rows, offset = [], 0
    while offset < limit:
        q = urllib.parse.urlencode({"target_chembl_id": chembl_id,
                                    "standard_type": "IC50",
                                    "limit": page, "offset": offset})
        try:
            d = get(f"{BASE}/activity.json?{q}")
        except Exception:
            if page <= floor:
                raise
            page = max(floor, page // 2)
            print(f"[build] page too large; retrying at limit={page}", flush=True)
            continue
        acts = d.get("activities", [])
        if not acts:
            break
        for a in acts:
            rows.append({"smiles": a.get("canonical_smiles"),
                         "p": a.get("pchembl_value"),
                         "rel": a.get("standard_relation"),
                         "assay": a.get("assay_chembl_id")})
        offset += page
        if not d.get("page_meta", {}).get("next"):
            break
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--page-size", type=int, default=200,
                    help="activity-endpoint page size; halves on failure")
    ap.add_argument("--out", default="/content/out/cyp/chembl_aux.parquet")
    ap.add_argument("--report", default="/content/out/cyp/chembl_build.json")
    args = ap.parse_args()

    data = load(workers=args.workers)
    from drugrx.data.standardise import standardise_many

    def std(sm):
        return standardise_many(list(sm), workers=args.workers)

    test_keys = {r.inchikey for r in std(data.test["SMILES"].tolist())
                 if r.ok and r.inchikey}
    print(f"[build] {len(test_keys)} test keys held out of the acquisition", flush=True)

    # Cache each isoform's raw fetch. The first run lost a completed CYP1A2
    # because the next isoform 500'd, which is the same defect as an ablation
    # driver that only writes its results at the end: finished work must be
    # durable before the next thing is attempted.
    raw_dir = Path(args.out).parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_iso, report = {}, {"targets": CYP_TARGETS, "per_isoform": {}}
    for name, cid in CYP_TARGETS.items():
        cache = raw_dir / f"chembl_raw_{name}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            print(f"[build] {name}: {len(df)} raw rows from cache", flush=True)
        else:
            df = fetch(cid, args.limit, page=args.page_size)
            df.to_parquet(cache, index=False)
            print(f"[build] {name}: {len(df)} raw rows cached", flush=True)
        n0 = len(df)
        df["pv"] = pd.to_numeric(df["p"], errors="coerce")
        # Censored relations are bounds, not measurements.
        df = df[df["rel"].isin(["=", None])].dropna(subset=["pv", "smiles"])
        n1 = len(df)
        res = std(df["smiles"].tolist())
        df = df.assign(key=[r.inchikey if r.ok else None for r in res],
                       cano=[r.smiles if r.ok else None for r in res])
        df = df.dropna(subset=["key"])
        n2 = len(df)
        hit = df["key"].isin(test_keys)
        df = df[~hit]
        # Median per compound: the mean is pulled by surviving extremes.
        g = df.groupby("key").agg(smiles=("cano", "first"),
                                  val=("pv", "median"),
                                  n=("pv", "size"),
                                  assays=("assay", "nunique"))
        per_iso[name] = g
        report["per_isoform"][name] = {
            "rows_fetched": int(n0), "after_censor_filter": int(n1),
            "after_standardise": int(n2), "test_collisions_removed": int(hit.sum()),
            "unique_compounds": int(len(g)),
            "tail_ge_5_5": int((g["val"] >= 5.5).sum()),
            "multi_assay_compounds": int((g["assays"] > 1).sum()),
        }
        print(f"[build] {name}: {n0} rows -> {n1} exact -> {n2} standardised -> "
              f"{len(g)} unique compounds ({int((g['val']>=5.5).sum())} in the tail); "
              f"{int(hit.sum())} test collisions removed", flush=True)

    keys = sorted(set().union(*[set(g.index) for g in per_iso.values()]))
    out = pd.DataFrame(index=keys)
    for name, g in per_iso.items():
        out[f"chembl_{name}"] = g["val"].reindex(keys)
    out["smiles"] = pd.concat([g["smiles"] for g in per_iso.values()]).groupby(level=0).first().reindex(keys)
    out = out.reset_index(names="inchikey").dropna(subset=["smiles"])

    # Cross-assay heterogeneity, measured properly this time: the same compound
    # across DISTINCT assays. The screen's within-compound sd was ~0 because the
    # repeats were duplicate records.
    het = {}
    for name, g in per_iso.items():
        multi = g[g["assays"] > 1]
        het[name] = {"compounds_in_2plus_assays": int(len(multi))}
    report["cross_assay"] = het
    report["final_rows"] = int(len(out))
    report["final_tail_rows"] = {c: int((out[c] >= 5.5).sum())
                                 for c in out.columns if c.startswith("chembl_")}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n[build] wrote {args.out}: {len(out)} compounds, "
          f"tail counts {report['final_tail_rows']}", flush=True)
    print(f"[build] compounds appearing in 2+ distinct assays (the only honest basis "
          f"for a cross-assay noise estimate): "
          f"{ {k: v['compounds_in_2plus_assays'] for k, v in het.items()} }", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
