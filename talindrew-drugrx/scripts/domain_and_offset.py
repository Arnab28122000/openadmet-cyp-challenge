"""Three checks that have to happen before H7 runs, not after.

**1. Retire or confirm the Avalon row.** A percentile threshold against a random
drug-like background does not settle it on its own: ExpansionRx is a focused
commercial panel, so its *internal* similarity is elevated too. The comparison
that actually decides it is **ExpansionRx-vs-random-background against
ExpansionRx-vs-test**. If the test-fold similarity distribution is
indistinguishable from the random background in Avalon space, then 99.8% is
baseline density and the row is dead. If test is shifted above background, it is
real proximity and the row stands.

**2. The applicability domain, stated as a number before the run.** ExpansionRx
sits largely outside the test distribution — which is what makes it safe and
also what predicts weak transfer. So a null is ambiguous unless the readout is
**stratified by domain membership** in advance. Here we compute what fraction of
each test fold falls inside ExpansionRx's domain, so the prediction can be
"movement concentrated in that fraction" rather than a directional hope. Then a
null decomposes: if the in-domain subset moves and the out-of-domain subset does
not, transfer worked exactly as expected and the aggregate null is a coverage
artefact, not evidence about data-boundedness.

**3. The cross-lab offset on cl_micro.** ExpansionRx's 18,514 microsomal
clearance values are *their* protocol; TDC's `clearance_microsome_az` is the
AstraZeneca assay — which, per the mechanism-A finding, is the ChEMBL document
essentially whole. Two labs' microsomal clearance correlate but are not the same
scale. Without this check, a null cannot be told apart from "a cross-lab offset
ate the signal". If there is a systematic offset it should be *fitted*, not left
sitting inside the null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdFingerprintGenerator

from drugrx.data.splits import proximity_verdict
from drugrx.data.standardise import standardise_many

REPOS = [
    "openadmet/openadmet-expansionrx-challenge-data",
    "openadmet/openadmet-expansionrx-challenge-train-data",
]


def fps(smiles: list[str], kind: str):
    out = []
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            out.append(pyAvalonTools.GetAvalonFP(m, nBits=1024) if kind == "avalon"
                       else morgan.GetFingerprint(m))
        except Exception:  # noqa: BLE001
            pass
    return out


def max_sim(query_fps, ref_fps) -> np.ndarray:
    if not query_fps or not ref_fps:
        return np.zeros(0)
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref_fps)) for f in query_fps])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/out/domain_offset.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=600)
    args = ap.parse_args()

    from datasets import load_dataset

    frame = pd.read_parquet(args.data)
    held = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in held])
    good = frame[frame["key"] != ""].copy()

    parts = []
    for repo in REPOS:
        try:
            ds = load_dataset(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[dom] {repo}: {exc}", flush=True)
            continue
        for _, p in ds.items():
            parts.append(p.to_pandas())
    erx = pd.concat(parts, ignore_index=True)
    erx_std = standardise_many([s for s in erx["SMILES"] if isinstance(s, str)],
                               workers=args.workers)
    erx["key"] = [s.inchikey if s.ok else "" for s in erx_std]
    erx["std_smiles"] = [s.smiles if s.ok else "" for s in erx_std]
    erx = erx[erx["key"] != ""]
    print(f"[dom] ExpansionRx {len(erx)} standardised rows", flush=True)

    rng = np.random.default_rng(0)
    erx_smiles = erx["std_smiles"].drop_duplicates().tolist()
    sample = (erx_smiles if len(erx_smiles) <= args.sample
              else [erx_smiles[i] for i in rng.choice(len(erx_smiles), args.sample, replace=False)])

    # A random drug-like background drawn from our own table, excluding anything
    # in a test fold, so "background" means ordinary chemistry rather than the
    # thing being tested against.
    any_test = set()
    for col in [c for c in good.columns if c.endswith("__fold")]:
        any_test |= set(good.loc[good[col] == "test", "key"])
    bg_pool = good.loc[~good["key"].isin(any_test), "smiles"].tolist()
    background = [bg_pool[i] for i in rng.choice(len(bg_pool), min(args.sample, len(bg_pool)),
                                                 replace=False)]

    report: dict = {"avalon_null": [], "domain": [], "cl_micro_offset": {}}

    for kind in ("avalon", "morgan"):
        q = fps(sample, kind)
        bg = fps(background, kind)
        sim_bg = max_sim(q, bg)
        for endpoint in sorted({c[: -len("__fold")] for c in good.columns if c.endswith("__fold")}):
            test_smiles = good.loc[good[f"{endpoint}__fold"] == "test", "smiles"].tolist()
            if len(test_smiles) < 20:
                continue
            sim_test = max_sim(q, fps(test_smiles, kind))
            if not len(sim_test) or not len(sim_bg):
                continue
            # The decisive comparison: is proximity to the test fold any higher
            # than proximity to ordinary chemistry?
            shift = float(sim_test.mean() - sim_bg.mean())
            report["avalon_null"].append({
                "representation": kind, "endpoint": endpoint,
                "mean_vs_test": round(float(sim_test.mean()), 4),
                "mean_vs_background": round(float(sim_bg.mean()), 4),
                "shift": round(shift, 4),
                "verdict": proximity_verdict(shift),
            })
        print(f"[dom] {kind}: background mean {sim_bg.mean():.3f}", flush=True)

    for row in report["avalon_null"]:
        print(f"  {row['representation']:7s} {row['endpoint']:16s} test {row['mean_vs_test']:.3f} "
              f"bg {row['mean_vs_background']:.3f} shift {row['shift']:+.3f} → {row['verdict']}",
              flush=True)

    # ── applicability domain: what fraction of each test fold is reachable ──
    erx_morgan = fps(sample, "morgan")
    for endpoint in sorted({c[: -len("__fold")] for c in good.columns if c.endswith("__fold")}):
        test_smiles = good.loc[good[f"{endpoint}__fold"] == "test", "smiles"].tolist()
        if len(test_smiles) < 20:
            continue
        sims = max_sim(fps(test_smiles, "morgan"), erx_morgan)
        if not len(sims):
            continue
        report["domain"].append({
            "endpoint": endpoint,
            "test_size": len(test_smiles),
            "frac_in_domain_0.3": round(float((sims > 0.3).mean()), 4),
            "frac_in_domain_0.4": round(float((sims > 0.4).mean()), 4),
            "median_sim": round(float(np.median(sims)), 4),
        })

    print("\n[dom] applicability domain (test molecules within reach of ExpansionRx):", flush=True)
    for r in report["domain"]:
        print(f"  {r['endpoint']:16s} n={r['test_size']:5d} in-domain@0.3 "
              f"{r['frac_in_domain_0.3']:6.1%} @0.4 {r['frac_in_domain_0.4']:6.1%} "
              f"median {r['median_sim']:.3f}", flush=True)

    # ── cross-lab offset on cl_micro ────────────────────────────────
    col = next((c for c in ("HLM CLint", "MLM CLint") if c in erx.columns), None)
    if col and "cl_micro" in good.columns:
        ours = good[["key", "cl_micro"]].dropna()
        theirs = erx[["key", col]].dropna().groupby("key")[col].mean()
        merged = ours.set_index("key").join(theirs, how="inner")
        report["cl_micro_offset"] = {
            "shared_molecules": int(len(merged)),
            "column": col,
        }
        if len(merged) >= 10:
            from scipy.stats import pearsonr, spearmanr

            a = merged["cl_micro"].to_numpy(dtype=float)
            b = np.log10(np.clip(merged[col].to_numpy(dtype=float), 1e-6, None))
            report["cl_micro_offset"].update({
                "pearson": round(float(pearsonr(a, b).statistic), 4),
                "spearman": round(float(spearmanr(a, b).statistic), 4),
                "mean_offset": round(float(np.mean(b - a)), 4),
            })
        print(f"\n[dom] cl_micro cross-lab: {report['cl_micro_offset']}", flush=True)
    else:
        print("\n[dom] cl_micro cross-lab: no shared column/endpoint to compare", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[dom] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
