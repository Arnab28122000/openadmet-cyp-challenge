"""Screen Veith for domain coverage of the blind test set BEFORE promoting it.

This programme has earned an acquisition criterion and has twice been right to
apply it. ExpansionRx came in at 18% coverage and produced a sub-resolution
effect, which was predicted in advance. ChEMBL mechanism-B was declined at
11.2%/6.4% against a pre-registered >=40% rule. Promoting Veith from auxiliary
heads to primary training signal is a week of work, so it gets screened first.

Using the 750 test SMILES for this costs nothing against the blind-set rule: the
structures are published, only the labels are withheld, and nothing here reads a
label.

**The measure that matters is incremental coverage, not coverage.** If the
challenge's own training set already covers a test compound, a second source
covering it again adds little. So this reports:

* `cover_train` -- fraction of the 750 within Tanimoto >= t of any challenge
  training compound;
* `cover_veith` -- the same for Veith;
* `incremental` -- covered by Veith and **not** by the challenge training set.
  This is the number the decision hangs on.

Reported at t = 0.3 (the precedent threshold), plus 0.4 and 0.5 so the verdict is
not an artifact of one cut.

**Pre-registered rule, written before the number is known:** promote Veith to
primary only if incremental coverage at Tanimoto 0.3 is **>= 25%**. The threshold
is set above ExpansionRx's 18% precisely because 18% produced an effect below this
task's resolution; a bar at or below it would license repeating that.

Exact-identity overlap is also reported on standardised InChIKey. External data is
explicitly permitted by the challenge rules, so overlap is an opportunity rather
than a violation -- but it must be *known*, because Veith is a qHTS AC50 panel and
this challenge is pIC50 from dose-response, so a shared compound does not carry a
transferable label and silently treating it as one would be the same class of
error as the auxiliary-column leak.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import load                      # noqa: E402

PREREGISTERED_MIN_INCREMENTAL = 0.25


def fps(smiles, workers=12):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        out.append(gen.GetFingerprint(m)); keep.append(i)
    return out, keep


def max_sim_matrix(query_fps, ref_fps, block=256):
    """Max Tanimoto from each query to any reference. BulkTanimoto in blocks."""
    from rdkit import DataStructs
    best = np.zeros(len(query_fps))
    for i, q in enumerate(query_fps):
        m = 0.0
        for s in range(0, len(ref_fps), block):
            sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps[s:s + block])
            if sims:
                m = max(m, max(sims))
        best[i] = m
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default="/content/out/cyp/external.parquet")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/cyp_veith/screen.json")
    args = ap.parse_args()

    data = load(workers=args.workers)
    test = data.test
    train_smiles = data.frame["smiles"].tolist()
    ext = pd.read_parquet(args.external)
    veith_smiles = ext["smiles"].tolist()
    print(f"[screen] test {len(test)} | challenge train {len(train_smiles)} | "
          f"Veith {len(veith_smiles)}", flush=True)

    test_smiles = test["SMILES"].tolist()
    tf, tkeep = fps(test_smiles, args.workers)
    trf, _ = fps(train_smiles, args.workers)
    vf, _ = fps(veith_smiles, args.workers)
    print(f"[screen] fingerprints: test {len(tf)}, train {len(trf)}, veith {len(vf)}",
          flush=True)

    s_train = max_sim_matrix(tf, trf)
    s_veith = max_sim_matrix(tf, vf)

    report = {"n_test_fp": len(tf), "n_train": len(trf), "n_veith": len(vf),
              "prereg_min_incremental_at_0.3": PREREGISTERED_MIN_INCREMENTAL,
              "thresholds": {}}
    print("\n[screen] coverage of the 750 blind test compounds", flush=True)
    print(f"    {'t':>4s} {'cover_train':>12s} {'cover_veith':>12s} {'incremental':>12s}",
          flush=True)
    for t in (0.3, 0.4, 0.5):
        ct = float((s_train >= t).mean())
        cv = float((s_veith >= t).mean())
        inc = float(((s_veith >= t) & (s_train < t)).mean())
        report["thresholds"][str(t)] = {"cover_train": ct, "cover_veith": cv,
                                        "incremental": inc}
        print(f"    {t:4.1f} {ct:12.1%} {cv:12.1%} {inc:12.1%}", flush=True)

    print(f"\n[screen] similarity distribution of test -> Veith: "
          f"median {np.median(s_veith):.3f} p90 {np.percentile(s_veith,90):.3f} "
          f"max {s_veith.max():.3f}", flush=True)
    print(f"[screen] similarity distribution of test -> challenge train: "
          f"median {np.median(s_train):.3f} p90 {np.percentile(s_train,90):.3f} "
          f"max {s_train.max():.3f}", flush=True)

    # Exact identity, on standardised InChIKey -- the identity contract this
    # programme settled after 425 collisions hid behind salt forms.
    try:
        from drugrx.data.standardise import standardise_many

        def keys(sm):
            # Full InChIKey, stereo preserved -- the merge key this programme
            # settled on after 425 train/test collisions hid behind salt forms.
            return {r.inchikey for r in standardise_many(sm, workers=args.workers)
                    if r.ok and r.inchikey}

        tk, vk, trk = keys(test_smiles), keys(veith_smiles), keys(train_smiles)
        report["exact_overlap_test_veith"] = len(tk & vk)
        report["exact_overlap_test_train"] = len(tk & trk)
        print(f"\n[screen] exact InChIKey overlap: test&Veith {len(tk & vk)} / {len(tk)}"
              f"   test&challenge-train {len(tk & trk)} / {len(tk)}", flush=True)
    except Exception as e:
        report["exact_overlap_error"] = str(e)
        print(f"[screen] exact overlap unavailable: {e}", flush=True)

    inc03 = report["thresholds"]["0.3"]["incremental"]
    verdict = "PROMOTE" if inc03 >= PREREGISTERED_MIN_INCREMENTAL else "DECLINE"
    report["verdict"] = verdict
    print(f"\n[screen] incremental coverage at 0.3 = {inc03:.1%} vs pre-registered "
          f"{PREREGISTERED_MIN_INCREMENTAL:.0%} -> **{verdict}**", flush=True)
    if verdict == "DECLINE":
        print("[screen] ExpansionRx at 18% produced a sub-resolution effect; a source "
              "below the bar buys another one.", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
