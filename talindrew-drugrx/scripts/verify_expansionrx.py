"""Re-verify ExpansionRx's reported zero overlap on the fixed InChIKey join.

The figure — 30,369 labels with "measured zero overlap" — was computed before
the standardisation fix, and **a broken SMILES join returns exactly zero**. Zero
is then reported as good news and never questioned; the same artefact matched 21
of lipophilicity's 420 molecules and was only caught because the number was
absurdly low rather than absurdly good. A clean zero across 30k molecules is
entirely possible for unpublished proprietary chemistry, but it has to be
established on the join that produced 418/420, not inherited.

Checks three things, because a zero can be true, artefactual, or both:

* **overlap on standardised InChIKey** — the real identity test;
* **how many of the pull's molecules standardise at all** — if standardisation
  fails en masse the keys are empty strings and everything "misses" by default,
  which is the artefact wearing the shape of the result;
* **nearest-neighbour similarity to each test fold** — because identity zero
  with Tanimoto 0.9 neighbours is not novel chemistry, it is the same chemistry
  spelled differently, and under a scaffold split that still inflates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drugrx.data.splits import nearest_neighbour_similarity
from drugrx.data.standardise import standardise_many

REPOS = [
    "openadmet/openadmet-expansionrx-challenge-data",
    "openadmet/openadmet-expansionrx-challenge-train-data",
]


def load_rows(repo: str) -> pd.DataFrame:
    from datasets import load_dataset

    frames = []
    try:
        ds = load_dataset(repo)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{repo}] could not load: {exc}", flush=True)
        return pd.DataFrame()
    for split, part in ds.items():
        df = part.to_pandas()
        df["__split"] = split
        frames.append(df)
        print(f"  [{repo}] {split}: {len(df)} rows, cols {list(df.columns)[:8]}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def find_smiles_column(df: pd.DataFrame) -> str | None:
    for candidate in df.columns:
        low = candidate.lower()
        if "smiles" in low or low in {"structure", "canonical_smiles"}:
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/out/expansionrx_verify.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nn-sample", type=int, default=1200)
    args = ap.parse_args()

    frame = pd.read_parquet(args.data)
    held = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in held])
    good = frame[frame["key"] != ""]
    held_keys = set(good["key"])

    folds = {}
    for col in [c for c in good.columns if c.endswith("__fold")]:
        endpoint = col[: -len("__fold")]
        sub = good[good[col] == "test"]
        folds[endpoint] = (set(sub["key"]), sub["smiles"].tolist())
    any_test = set().union(*(k for k, _ in folds.values())) if folds else set()
    print(f"[verify] held {len(held_keys)} molecules, {len(any_test)} in some test fold", flush=True)

    report = {"repos": [], "endpoints": []}
    all_keys: set[str] = set()
    all_smiles: list[str] = []

    for repo in REPOS:
        df = load_rows(repo)
        if df.empty:
            report["repos"].append({"repo": repo, "rows": 0, "error": "empty"})
            continue
        col = find_smiles_column(df)
        if col is None:
            report["repos"].append({"repo": repo, "rows": len(df),
                                    "error": f"no SMILES column in {list(df.columns)}"})
            continue
        smiles = [s for s in df[col].tolist() if isinstance(s, str) and s]
        std = standardise_many(smiles, workers=args.workers)
        ok = [s for s in std if s.ok]
        keys = {s.inchikey for s in ok}
        all_keys |= keys
        all_smiles.extend(s.smiles for s in ok)

        entry = {
            "repo": repo,
            "rows": int(len(df)),
            "smiles_column": col,
            "with_smiles": len(smiles),
            # If this is far below `with_smiles`, a reported zero is an artefact
            # of failed standardisation rather than novel chemistry.
            "standardised_ok": len(ok),
            "standardise_fail_rate": round(1 - len(ok) / max(1, len(smiles)), 4),
            "unique_molecules": len(keys),
            "overlap_any_test": len(keys & any_test),
            "overlap_held": len(keys & held_keys),
        }
        report["repos"].append(entry)
        print(f"  → unique {len(keys)}, standardise failures "
              f"{entry['standardise_fail_rate']:.1%}, overlap-any-test "
              f"{entry['overlap_any_test']}, overlap-held {entry['overlap_held']}", flush=True)

    rng = np.random.default_rng(0)
    sample = all_smiles
    if len(sample) > args.nn_sample:
        sample = [sample[i] for i in rng.choice(len(sample), args.nn_sample, replace=False)]

    for endpoint, (keys, smiles) in sorted(folds.items()):
        if not smiles:
            continue
        overlap = len(all_keys & keys)
        nn = nearest_neighbour_similarity(sample, smiles) if sample else np.zeros(0)
        row = {
            "endpoint": endpoint,
            "test_size": len(keys),
            "identity_overlap": overlap,
            "frac_test_hit": round(overlap / max(1, len(keys)), 4),
            "nn_sim_mean": round(float(nn.mean()), 4) if len(nn) else None,
            "nn_frac_above_0.4": round(float((nn > 0.4).mean()), 4) if len(nn) else None,
            "nn_frac_above_0.7": round(float((nn > 0.7).mean()), 4) if len(nn) else None,
        }
        report["endpoints"].append(row)
        print(f"{endpoint:16s} test {row['test_size']:5d} | identity overlap {overlap:4d} "
              f"({row['frac_test_hit']:.1%}) | NN>0.4 {row['nn_frac_above_0.4']:.1%} "
              f"| NN>0.7 {row['nn_frac_above_0.7']:.1%}", flush=True)

    report["total_unique"] = len(all_keys)
    report["total_overlap_any_test"] = len(all_keys & any_test)
    print(f"\n[verify] {len(all_keys)} unique molecules, "
          f"{len(all_keys & any_test)} collide with some test fold", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[verify] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
