"""Scaffold-level contamination of a candidate pull, after identity exclusion.

Removing exact InChIKey matches is necessary and **not sufficient**. TDC's ADMET
Group splits by Bemis-Murcko scaffold, so a pull that is heavy in a scaffold
series the test fold also sits in inflates the benchmark while every identity
join reports clean. It is the same contamination wearing a different mask.

Three numbers per endpoint, and they say different things:

* **scaffold overlap** — fraction of the surviving residual whose Murcko
  scaffold also appears in that endpoint's test fold. High means the pull is
  concentrated in series the benchmark is measuring.
* **test-fold coverage** — fraction of the *test* scaffolds the residual
  touches. This is the one that matters for inflation: covering most test
  scaffolds means the model sees the series it will be scored on, even with no
  molecule in common.
* **nearest-neighbour similarity** — the continuous version, because two
  molecules can share no Murcko scaffold and still be near-identical (76% of
  ESOL's test set has a training neighbour above Tanimoto 0.4).

A pull that passes identity exclusion but covers most test scaffolds should be
scaffold-filtered too, or the endpoint it feeds has to be reported as
non-comparable to the leaderboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drugrx.data.splits import murcko_scaffold, nearest_neighbour_similarity
from drugrx.data.standardise import standardise_many


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--raw", default="/content/out/chembl_raw.json")
    ap.add_argument("--audit", default="/content/out/chembl_audit.json")
    ap.add_argument("--out", default="/content/out/scaffold_audit.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nn-sample", type=int, default=1500,
                    help="molecules sampled for the similarity estimate")
    args = ap.parse_args()

    frame = pd.read_parquet(args.data)
    held = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in held])
    good = frame[frame["key"] != ""]
    held_keys = set(good["key"])

    raw = json.loads(Path(args.raw).read_text())
    audit = {r["pull"]: r for r in json.loads(Path(args.audit).read_text())["pulls"]}

    rng = np.random.default_rng(0)
    out = []

    for pull, rows in raw.items():
        endpoint = audit.get(pull, {}).get("endpoint", pull)
        fold_col = f"{endpoint}__fold"
        if fold_col not in good.columns:
            continue

        test = good[good[fold_col] == "test"]
        test_smiles = test["smiles"].tolist()
        test_keys = set(test["key"])
        if not test_smiles:
            continue

        # The residual: new molecules that survive identity exclusion against
        # everything we hold and against every test fold.
        any_test = set()
        for col in [c for c in good.columns if c.endswith("__fold")]:
            any_test |= set(good.loc[good[col] == "test", "key"])
        residual = [r for r in rows if r["key"] not in held_keys and r["key"] not in any_test]
        if not residual:
            out.append({"pull": pull, "endpoint": endpoint, "residual": 0})
            continue

        res_smiles = [r["smiles"] for r in residual]
        # The definition must match the one the split boundary was drawn with:
        # RDKit MurckoScaffoldSmiles, non-generic, chirality off (the DeepChem
        # lineage TDC inherited). A *generic framework* — every atom to carbon,
        # every bond to single — yields a far higher collision rate that
        # corresponds to nothing the split actually did.
        raw_test = [murcko_scaffold(x) for x in test_smiles]
        raw_res = [murcko_scaffold(x) for x in res_smiles]

        # Acyclic molecules return an EMPTY scaffold and would all collapse into
        # a single bucket, manufacturing a collision spike — a half-life pull
        # contains a real number of small aliphatic drugs. Segregate and report
        # them rather than letting them land in the count either way.
        test_acyclic = sum(1 for s in raw_test if s == "")
        res_acyclic = sum(1 for s in raw_res if s == "")
        test_scaffolds = {s for s in raw_test if s}
        res_scaffolds = [s for s in raw_res if s]

        shared = [s for s in res_scaffolds if s in test_scaffolds]
        covered = test_scaffolds & set(res_scaffolds)

        sample = res_smiles
        if len(sample) > args.nn_sample:
            sample = [sample[i] for i in rng.choice(len(sample), args.nn_sample, replace=False)]
        nn = nearest_neighbour_similarity(sample, test_smiles)

        row = {
            "pull": pull,
            "endpoint": endpoint,
            "residual": len(residual),
            "test_acyclic_excluded": test_acyclic,
            "residual_acyclic_excluded": res_acyclic,
            "test_scaffolds": len(test_scaffolds),
            "residual_in_test_scaffold": len(shared),
            "frac_residual_in_test_scaffold": round(len(shared) / max(1, len(res_scaffolds)), 4),
            "test_scaffolds_covered": len(covered),
            "frac_test_scaffolds_covered": round(len(covered) / max(1, len(test_scaffolds)), 4),
            "nn_sim_mean": round(float(nn.mean()), 4) if len(nn) else None,
            "nn_sim_frac_above_0.4": round(float((nn > 0.4).mean()), 4) if len(nn) else None,
        }
        out.append(row)
        # **Test-scaffold recovery is the headline**, not residual sharing: the
        # question is what fraction of test scaffolds — absent from train by
        # construction — the residual puts back.
        print(f"{pull:14s} residual {row['residual']:6d} | "
              f"RECOVERS {row['frac_test_scaffolds_covered']:6.1%} of test scaffolds "
              f"({row['test_scaffolds_covered']}/{row['test_scaffolds']}) | "
              f"{row['frac_residual_in_test_scaffold']:5.1%} of residual in a test scaffold | "
              f"NN>0.4 {row['nn_sim_frac_above_0.4']:.1%} | acyclic excluded "
              f"{res_acyclic}/{test_acyclic}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
