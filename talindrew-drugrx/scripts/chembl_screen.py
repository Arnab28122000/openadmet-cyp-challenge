"""Screen the ChEMBL mechanism-B residual: would it actually unblock anything?

**Run the screen, not the acquisition.** "The only remaining source that could
move half_life" is a conditional, and it can be tested directly with machinery
already built rather than acquired first and discovered afterwards.

The test: take the mechanism-B pulls, exclude identity collisions, **then exclude
scaffold collisions too**, and measure what is left — specifically its Morgan-0.3
coverage of each TDC test fold, exactly as measured for ExpansionRx.

The expectation, recorded as H10 before running: scaffold-filtering removes the
chemistry nearest the test folds **by construction**, so the survivor should look
like ExpansionRx — plenty of labels, little coverage, a sub-resolution effect. If
coverage lands in the teens, ChEMBL does not make H7b decidable and the
acquisition is not worth its ShareAlike exposure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chembl_audit import ASSAY_FILTERS, PULLS, fetch, passes_quality  # noqa: E402
from drugrx.data.splits import murcko_scaffold, nearest_neighbour_similarity  # noqa: E402
from drugrx.data.standardise import standardise_many  # noqa: E402

#: Only the mechanism-B pulls. The mechanism-A documents ARE the benchmarks and
#: were dropped outright; re-screening them would be meaningless.
MECHANISM_B = ("herg", "half_life", "caco2", "solubility", "vdss")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/out/chembl_screen.json")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=1200)
    args = ap.parse_args()

    frame = pd.read_parquet(args.data)
    held = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in held])
    good = frame[frame["key"] != ""]
    held_keys = set(good["key"])
    any_test = set()
    for c in [c for c in good.columns if c.endswith("__fold")]:
        any_test |= set(good.loc[good[c] == "test", "key"])

    rng = np.random.default_rng(0)
    rows = []
    for name in MECHANISM_B:
        spec = dict(PULLS[name])
        endpoint = spec.pop("endpoint")
        pulled = fetch({**spec, **ASSAY_FILTERS.get(endpoint, {})}, limit=args.limit)
        pulled = [r for r in pulled if passes_quality(r)]
        smiles = [r.get("canonical_smiles") for r in pulled]
        smiles = [s for s in smiles if s]
        if not smiles:
            print(f"[screen] {name}: nothing pulled", flush=True)
            continue

        std = standardise_many(smiles, workers=args.workers)
        ok = [s for s in std if s.ok]

        test = good[good[f"{endpoint}__fold"] == "test"]
        test_smiles = test["smiles"].tolist()
        test_scaffolds = {s for s in (murcko_scaffold(x) for x in test_smiles) if s}

        # Identity exclusion, then scaffold exclusion.
        after_identity = [s for s in ok if s.inchikey not in held_keys and s.inchikey not in any_test]
        survivors = [s for s in after_identity
                     if (murcko_scaffold(s.smiles) or "") not in test_scaffolds]

        sample = [s.smiles for s in survivors]
        if len(sample) > args.sample:
            sample = [sample[i] for i in rng.choice(len(sample), args.sample, replace=False)]
        cov = nearest_neighbour_similarity(test_smiles, sample) if sample else np.zeros(0)

        row = {
            "pull": name, "endpoint": endpoint,
            "pulled": len(pulled), "unique": len(ok),
            "after_identity_exclusion": len(after_identity),
            "after_scaffold_exclusion": len(survivors),
            "test_size": len(test_smiles),
            "coverage_0.3": round(float((cov > 0.3).mean()), 4) if len(cov) else 0.0,
            "coverage_0.4": round(float((cov > 0.4).mean()), 4) if len(cov) else 0.0,
            "median_sim": round(float(np.median(cov)), 4) if len(cov) else 0.0,
        }
        rows.append(row)
        print(f"{name:11s} unique {row['unique']:6d} → identity {row['after_identity_exclusion']:6d} "
              f"→ scaffold {row['after_scaffold_exclusion']:6d} | covers "
              f"{row['coverage_0.3']:6.1%} of the {row['test_size']}-molecule fold at 0.3", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2, default=float))
    hl = next((r for r in rows if r["endpoint"] == "half_life"), None)
    if hl:
        print(f"\nhalf_life coverage after both exclusions: {hl['coverage_0.3']:.1%}")
        print("H10 predicted the teens or below; if so H7b stays undecidable.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
