"""Pull ChEMBL ADME data and measure, per endpoint, how much of it is our test set.

**This audit gates the acquisition, and the licence does not.** Four of the TDC
ADMET endpoints we benchmark *are themselves AstraZeneca deposits into ChEMBL* —
`ppbr_az`, `lipophilicity_astrazeneca`, `clearance_hepatocyte_az`,
`clearance_microsome_az`. A naive ChEMBL pull therefore re-imports test
molecules for exactly the endpoints we have just reported results on, including
`ppbr`, which is one of the two endpoints that *lost* under H5 and whose number
would become uninterpretable.

So: overlap is measured **per endpoint against that endpoint's own test fold**,
on standardised InChIKey — never on SMILES (a string join matched 21 of
lipophilicity's 420 molecules once already), and never only against the union of
all test molecules, which would hide which endpoint is contaminated.

What the audit reports, and why each number matters:

* **gross** — rows pulled. The headline figure, and the least useful one.
* **test collisions, per endpoint** — molecules that would poison a specific
  benchmark. These are dropped, not flagged.
* **already held** — present in the current training table; no new information,
  only a duplicate to resolve.
* **residual** — ChEMBL-exclusive molecules that are genuinely new and safe.
  **This is the number the keep/drop decision turns on.** If the residual is a
  fraction of the gross, dropping ChEMBL becomes cheap and no governance effort
  needs to be spent on it.

Provenance is recorded **per record**, not as a flag on a chunk: a partition flag
gives a retrain path, while a `source_license` per measurement lets us say which
obligations attach to a given trained artefact — which is the question actually
asked. Where a measurement is reachable from a more permissive source
(BindingDB CC BY 3.0, PubChem), that provenance is preferred and the ChEMBL copy
is dropped from the obligation set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

#: The AstraZeneca ChEMBL deposits that *are* four of our TDC endpoints, plus the
#: other ADME pulls worth auditing. Assay ids come from research/04; they are the
#: reason this audit exists.
PULLS: dict[str, dict] = {
    "lipophilicity": {"assay_chembl_id": "CHEMBL3301363", "endpoint": "lipophilicity"},
    "ppbr":          {"assay_chembl_id": "CHEMBL3301365", "endpoint": "ppbr"},
    "cl_micro":      {"assay_chembl_id": "CHEMBL3301370", "endpoint": "cl_micro"},
    "herg":          {"target_chembl_id": "CHEMBL240", "standard_type": "IC50",
                      "endpoint": "herg"},
    "half_life":     {"standard_type": "T1/2", "endpoint": "half_life"},
    "caco2":         {"standard_type": "Papp", "endpoint": "caco2"},
    "solubility":    {"standard_type": "Solubility", "endpoint": "solubility"},
    "vdss":          {"standard_type": "Vdss", "endpoint": "vdss"},
}

#: **Label heterogeneity is a bigger risk than leakage in the mechanism-B pulls,
#: and it is the one that would be misread as a result.** TDC's hERG is binary
#: blocker/non-blocker from one curated source at one threshold; 14,406 ChEMBL
#: hERG records are patch-clamp IC50, binding displacement, several cell lines
#: and protocols. Obach's half-life set is ~670 *human intravenous terminal*
#: half-lives, carefully curated; ChEMBL `T1/2` spans species, routes and
#: oral/IV without discrimination.
#:
#: Merge rat oral half-life into a human-IV benchmark and the result is "more
#: data did not help" — which we would then read as confirming the data-bound
#: conclusion, when it actually measured our filtering. So the residual that
#: goes in the table is the **post-filter** residual.
ASSAY_FILTERS: dict[str, dict] = {
    "half_life": {"assay_organism": "Homo sapiens"},
    "herg": {"assay_organism": "Homo sapiens"},
    "caco2": {},          # Caco-2 is a human cell line by definition
    "solubility": {},
    "vdss": {"assay_organism": "Homo sapiens"},
}

#: Rows ChEMBL itself flags as suspect, and exact duplicates.
def passes_quality(row: dict) -> bool:
    if row.get("data_validity_comment"):
        return False
    if str(row.get("potential_duplicate", 0)) in {"1", "True"}:
        return False
    return True


def fetch(params: dict, limit: int = 20000, page_size: int = 1000) -> list[dict]:
    """Page through the ChEMBL activity endpoint.

    Note what is **not** filtered here: `pchembl_value`. Requiring it forces
    `standard_relation = '='` and silently deletes every censored record — all
    the ">100 uM" rows, i.e. the inactives. On hERG that alone costs 5,326
    molecules, eight times TDC's entire hERG set, and they are precisely the
    non-blockers a triage model most needs to recognise.
    """
    out: list[dict] = []
    offset = 0
    while len(out) < limit:
        query = {**params, "limit": page_size, "offset": offset,
                 "only": "molecule_chembl_id,canonical_smiles,standard_type,"
                         "standard_value,standard_units,standard_relation,assay_chembl_id,"
                         "assay_organism,data_validity_comment,potential_duplicate,"
                         "assay_description,bao_format,confidence_score"}
        try:
            r = requests.get(CHEMBL, params=query, timeout=90,
                             headers={"User-Agent": "talindrew-drugrx-audit/1.0"})
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"    request failed at offset {offset}: {exc}", flush=True)
            break
        rows = r.json().get("activities", [])
        if not rows:
            break
        out.extend(rows)
        offset += page_size
        if len(rows) < page_size:
            break
        time.sleep(0.2)
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/out/chembl_audit.json")
    ap.add_argument("--raw", default="/content/out/chembl_raw.json")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import pandas as pd

    from drugrx.data.standardise import standardise_many

    frame = pd.read_parquet(args.data)
    print(f"[audit] existing table: {len(frame)} molecules", flush=True)

    # Standardised identity for everything we already hold, split by fold.
    held = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in held])
    good = frame[frame["key"] != ""]

    all_keys = set(good["key"])
    test_keys_per_endpoint: dict[str, set[str]] = {}
    for col in [c for c in good.columns if c.endswith("__fold")]:
        endpoint = col[: -len("__fold")]
        test_keys_per_endpoint[endpoint] = set(good.loc[good[col] == "test", "key"])

    report: list[dict] = []
    raw_store: dict[str, list[dict]] = {}

    for name, spec in PULLS.items():
        endpoint = spec.pop("endpoint")
        print(f"\n[audit] {name} → endpoint {endpoint}: {spec}", flush=True)
        extra = ASSAY_FILTERS.get(endpoint, {})
        rows = fetch({**spec, **extra}, limit=args.limit)
        gross_rows = len(rows)
        rows = [r for r in rows if passes_quality(r)]
        print(f"    pulled {gross_rows} rows, {len(rows)} after quality filter "
              f"(organism/validity/duplicate: {extra or 'none'})", flush=True)
        smiles = [r.get("canonical_smiles") for r in rows]
        smiles = [s for s in smiles if s]
        print(f"    {len(smiles)} with SMILES", flush=True)
        if not smiles:
            report.append({"pull": name, "endpoint": endpoint, "gross_rows": len(rows),
                           "error": "no SMILES returned"})
            continue

        std = standardise_many(smiles, workers=args.workers)
        keys = {s.inchikey for s in std if s.ok}

        own_test = test_keys_per_endpoint.get(endpoint, set())
        collisions = keys & own_test
        # Contamination of ANY endpoint's test fold, not just this one's: the
        # encoder is shared, so a molecule that is test anywhere must not train.
        any_test = set().union(*test_keys_per_endpoint.values()) if test_keys_per_endpoint else set()
        any_collisions = keys & any_test
        already = keys & all_keys
        residual = keys - all_keys - any_test

        row = {
            "pull": name,
            "endpoint": endpoint,
            "gross_rows": gross_rows,
            "rows_after_quality_filter": len(rows),
            "assay_filter": extra,
            "unique_molecules": len(keys),
            "collides_own_test": len(collisions),
            "own_test_size": len(own_test),
            "frac_own_test_hit": round(len(collisions) / max(1, len(own_test)), 4),
            "collides_any_test": len(any_collisions),
            "already_held": len(already),
            "residual_new_and_safe": len(residual),
        }
        report.append(row)
        raw_store[name] = [{"smiles": s.smiles, "key": s.inchikey} for s in std if s.ok]
        print(f"    unique {len(keys)} | own-test hits {len(collisions)}/{len(own_test)} "
              f"({row['frac_own_test_hit']:.1%}) | any-test {len(any_collisions)} "
              f"| already held {len(already)} | RESIDUAL {len(residual)}", flush=True)

    gross = sum(r.get("unique_molecules", 0) for r in report)
    residual = sum(r.get("residual_new_and_safe", 0) for r in report)
    print(f"\n[audit] gross unique {gross} → residual new-and-safe {residual} "
          f"({residual / max(1, gross):.0%} of gross)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"pulls": report, "gross_unique": gross, "residual": residual}, indent=2))
    Path(args.raw).write_text(json.dumps(raw_store))
    print(f"[audit] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
