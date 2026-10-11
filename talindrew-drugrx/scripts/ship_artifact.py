"""Train the shippable artifact, write its model card, and push it to the Hub.

**Why now, before any ChEMBL merge.** This artifact is provably free of
ShareAlike obligation: its only non-TDC source is ExpansionRx, which is CC BY 4.0,
and ChEMBL contributes nothing to it. The moment ChEMBL mechanism-B data merges,
every downstream release inherits the trigger conditions recorded in
`docs/drugrx-training/LICENSING.md`, and a clean baseline stops existing
retroactively — there is no way to re-derive one afterwards without re-running
everything. Tagging it now turns ShareAlike into a decision about a *future*
release rather than a constraint on the only release there is.

The card records provenance per source rather than as a blanket statement, so a
reader can tell exactly which obligations attach — attribution to TDC's upstream
curators and to ExpansionRx/OpenADMET, and no share-alike from anywhere.

It also records what the model is *not*: the measured aggregate effect of the
ExpansionRx heads is sub-resolution per endpoint, so this is not a per-endpoint
improvement over the gradient-boosting baseline and must not be presented as one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.data.tdc import ADMET_TASKS  # noqa: E402
from drugrx.fast_train import GPUData, fit_ensemble, predict_ensemble  # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402
from h7a_three_arm import ERX_COLUMNS, build_matrix, load_expansionrx  # noqa: E402
from train_admet import load_data  # noqa: E402
from train_fast import PlainMLP, score_fold  # noqa: E402

CARD = """---
license: cc-by-4.0
tags: [chemistry, admet, drug-discovery, molecular-property-prediction, multi-task]
library_name: pytorch
---

# DrugRx ADMET multi-task ensemble (clean-provenance baseline)

A {members}-member deep ensemble predicting **{n_tasks} ADMET endpoints** from
count-ECFP4 + RDKit2D descriptors, trained with six auxiliary ExpansionRx assay
heads. Produced by `talindrew-drugrx`.

## What this is for

**In-silico triage: ranked hypotheses for wet-lab testing.** It is not a
therapeutic claim, and no output of it is evidence that a compound works.

The yardstick is blind prospective screening, and the honest form of it is not a
single number. On CACHE #1 (LRRK2 WDR) 23 teams produced 1,955 compounds and
**73 bound at K_D < 150 µM — a 3.7% hit rate — but only 7 of the 23 teams found
anything at all**. On CACHE #2 (Nsp13 helicase) the same scale of effort gave
**0.7%, with 5 of 23 teams finding anything**. So the modal outcome of a blind
screen is **zero hits**; 150 µM is a starting point rather than a candidate (the
best confirmed CACHE #1 work was 8 binders at 3–41 µM from 84 compounds over two
rounds, SPR plus 19F-NMR); and difficulty belongs to the **target**, not the
method — the same teams moved 3.7% to 0.7% by changing target.

One finding bears directly on this model: **37 of CACHE #1's 73 hits had poor
solubility despite computational prediction.** Aqueous solubility is one of the
21 endpoints here, and on it we cannot distinguish this model from a tuned
fingerprint baseline.

## What it is not

The ExpansionRx auxiliary heads produce a **consistent but sub-resolution**
improvement: 1 of 21 endpoints clears a permutation null's 95th percentile, while
the aggregate is clearly real (mean percentile 0.698 against a null-vs-null
reference of 0.504 ± 0.034, calibrated p < 0.0001). **No per-endpoint effect is
large enough to change a model choice**, and for most endpoints a tuned
count-ECFP4 + RDKit2D gradient-boosting model is at least as good and cheaper to
serve. Treat per-endpoint differences against that baseline as unresolved.

## Provenance and obligations

| source | licence | obligation |
|---|---|---|
| TDC ADMET Benchmark Group | per upstream dataset | attribution to upstream curators |
| ExpansionRx via OpenADMET | **CC BY 4.0** | attribution |
| **ChEMBL** | CC BY-SA 3.0 | **not used — contributes nothing to these weights** |

**No ShareAlike obligation attaches to this artifact.** That is a deliberate,
dated property of this release and not a general property of the project: a later
release that merges ChEMBL-derived data would inherit CC BY-SA conditions.

## Evaluation caveats, which travel with the numbers

- **Identity in this dataset is standardised InChIKey, and that is not cosmetic.** A SMILES-level
  check reported these splits as clean while **425 molecules — 3.2% of the test set — were present in
  training** as salt forms, alternate tautomers or re-spelled SMILES. Any benchmark compared against
  this one should state how it decided two molecules were the same.
- **Test molecules sit measurably closer to their training set than to ordinary
  chemistry on 20 of 21 endpoints** — mean nearest-neighbour similarity shift of
  **+0.019 to +0.180** over a random-background reference (`runs/leakage_calibrated.json`).
  This is a *background-calibrated* statement: a raw "fraction above Tanimoto 0.4"
  depends on fingerprint density and is not interpretable on its own. The practical
  consequence stands — scores on these splits overstate performance on genuinely
  new chemistry — but the effect is modest, not dramatic, and `ld50` shows none.
- Eight endpoints have test folds of 95–175 molecules; differences around 0.01 are
  inside split noise there, measured by permutation.
- **Externally sourced and not corroborated here:** published audits report that a
  minority of top-ranked TDC ADMET architectures reproduce, and that tuning against
  the public test set inflates apparent rank. We have not verified those claims or
  their reproduction criteria, and they are repeated only as a reason to compare
  against the baseline in this repo rather than against a leaderboard position.

## Training data identity

Molecules are deduplicated on **standardised InChIKey** — salts, charge states and
tautomers collapsed, **stereochemistry preserved**. A SMILES-level identity check
previously hid 425 train/test collisions (3.2% of the test set) that are invisible
at the string level.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--out-dir", default="/content/artifact")
    ap.add_argument("--repo", default="arnab2812/drugrx-admet-ensemble")
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    data, x = load_data(Path(args.data))
    frame = data.frame
    train_rows = (frame["split"] == "train").to_numpy()
    _, stats = standardise_descriptors(x[train_rows])
    x, _ = standardise_descriptors(x, stats=stats)

    erx = load_expansionrx(args.workers)
    erx_cols = [c for c in ERX_COLUMNS if c in erx.columns]
    erx_norm = {}
    for c in erx_cols:
        col = erx[c]
        mu, sd = float(col.mean()), float(col.std()) or 1.0
        erx[c] = (col - mu) / sd
        erx_norm[c] = (mu, sd)

    # **The ExpansionRx molecules must be featurised exactly as the TDC cache was,
    # and that is inferred from the cache's width rather than guessed.** It used to
    # be a bare `== 3091` comparison, which is the same inference but states no
    # expectation: if a cache of some third width appeared, Avalon and ErG were
    # silently dropped from the auxiliary rows only, giving the two halves of one
    # training matrix different features. Worse, it made the *artifact's* feature
    # set a property of whichever cache happened to be on a disposable VM — which
    # is how the shipped 24-member ensemble came to be a 2264-feature model while
    # the changelog recorded the featuriser as 3091.
    widths = {2264: False, 3091: True}
    if x.shape[1] not in widths:
        raise SystemExit(
            f"feature cache is {x.shape[1]} wide; expected 2264 (count+rdkit2d) or "
            "3091 (+Avalon+ErG). Rebuild it with prepare_data.py."
        )
    want_extra = widths[x.shape[1]]
    print(f"[ship] features {x.shape[1]} → extra_fps={want_extra}", flush=True)
    erx_x = featurise(erx["smiles"].tolist(), workers=args.workers,
                      counts=True, desc_mode="rdkit2d", extra_fps=want_extra)
    erx_x, _ = standardise_descriptors(erx_x, stats=stats)

    tr = np.flatnonzero(train_rows)
    va = np.flatnonzero((frame["split"] == "valid").to_numpy())
    keys = [t.key for t in ADMET_TASKS]
    y_full = frame[keys].to_numpy(dtype=np.float32)
    m_full = ~np.isnan(y_full)
    y_full = np.nan_to_num(y_full, nan=0.0)
    pad = np.zeros((len(va), len(erx_cols)), np.float32)
    gpu_valid = GPUData.from_numpy(
        x[va], np.concatenate([y_full[va], pad], 1),
        np.concatenate([m_full[va], pad.astype(bool)], 1), args.device)

    xt, yt, mt = build_matrix(frame.iloc[tr], x[tr], erx, erx_x, True, erx_cols)
    gpu_train = GPUData.from_numpy(xt, yt, mt, args.device)
    n_tasks = yt.shape[1]

    def make_model(_: int) -> nn.Module:
        return PlainMLP(xt.shape[1], n_tasks, 512, 3, 0.1)

    base_meta = make_model(0).to("meta")

    def score_fn(params, buffers) -> float:
        preds = predict_ensemble(base_meta, params, buffers, gpu_valid.x)
        diff = (preds.mean(0) - gpu_valid.y) ** 2
        return float((diff * gpu_valid.mask).sum() / gpu_valid.mask.sum().clamp(min=1.0))

    t0 = time.time()
    params, buffers, _ = fit_ensemble(
        make_model, gpu_train, gpu_valid, n_members=args.members,
        epochs=args.epochs, device=args.device, patience=args.patience,
        score_fn=score_fn,
    )
    minutes = (time.time() - t0) / 60

    x_all = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(args.device)
    preds = predict_ensemble(base_meta, params, buffers, x_all)
    mean = preds.mean(0).float().cpu().numpy()[:, : len(ADMET_TASKS)]
    test_scores = score_fold(data, frame, mean, "test")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": {k: v.cpu() for k, v in params.items()},
            "buffers": {k: v.cpu() for k, v in buffers.items()},
            "members": args.members,
            "in_dim": int(xt.shape[1]),
            # Stated, not inferable from the width alone by whatever loads this.
            # A serve path has to reproduce the featuriser exactly or every
            # prediction is wrong in a way that still looks like a number.
            "extra_fps": bool(want_extra),
            "featuriser": {"n_bits": 2048, "radius": 2, "counts": True,
                           "desc_mode": "rdkit2d", "extra_fps": bool(want_extra)},
            "n_tasks": int(n_tasks),
            "tdc_tasks": keys,
            "erx_heads": erx_cols,
            "feature_stats": {"mean": stats[0].tolist(), "std": stats[1].tolist()},
            "label_norm": data.norm,
            "erx_norm": erx_norm,
            "architecture": {"d_model": 512, "n_layers": 3, "dropout": 0.1},
        },
        out / "ensemble.pt",
    )
    (out / "README.md").write_text(CARD.format(members=args.members, n_tasks=len(keys)))
    (out / "test_scores.json").write_text(json.dumps(test_scores, indent=2, default=float))
    print(f"[ship] wrote {out}, {minutes:.1f} min", flush=True)
    print(json.dumps(test_scores, indent=2, default=float), flush=True)

    if args.push:
        import os

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("[ship] HF_TOKEN unset; not pushing")
            return 1
        api = HfApi(token=token)
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model")
        print(f"[ship] pushed to {args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
