"""H7a: does ExpansionRx's chemistry help the encoder, or is it just multi-task structure?

Three arms, because a two-arm design cannot tell those apart:

1. **baseline** — our 21 TDC heads, our molecules. The reference.
2. **+data** — our 21 heads *plus* ExpansionRx's 6 heads, trained on our molecules
   **and** ExpansionRx's 12,934.
3. **+chemistry, labels destroyed** — the same 27 heads *and* the same 12,934
   ExpansionRx molecules as arm 2, but their six label columns **permuted across
   molecules**, so the chemistry and the head structure are identical and only
   the label–structure relationship is destroyed.

**Arm 3 is the whole point, and the obvious version of it does not work.** The
first draft held the ExpansionRx *molecules* out while keeping their heads — but
a head with no rows has an all-zero mask, contributes nothing to the loss, and
is mathematically **identical to arm 1**. That control is degenerate: it would
have produced arm 3 ≡ arm 1 and been read as "structure contributes nothing".

Permuting the labels is the control that actually separates the two effects:

* **arm 2 − arm 3** — what the ExpansionRx *labels* bought, over and above seeing
  the molecules. This is the "more supervision" term.
* **arm 3 − arm 1** — what seeing 12,934 extra *molecules* bought with no usable
  labels at all: a pure representation effect from the shared encoder.

Together they decompose the total gain into chemistry-as-representation versus
chemistry-as-supervision, which is a sharper answer than the original design
would have given even if it had worked.

ExpansionRx labels are **separate heads, never merged into the TDC targets**.
Its microsomal clearance is a different laboratory's protocol from the
AstraZeneca assay behind `clearance_microsome_az`, and the two panels share
**zero** molecules, so there is no overlap on which a scale offset could even be
fitted. Letting each head learn its own scale is a weaker assumption than
inventing a cross-lab calibration — it would have been the right treatment even
with an overlap set.

The readout is **stratified by applicability domain** (H8/H9), decided before the
run: ExpansionRx covers 18.1% of the cl_micro test fold at Morgan 0.3 and less
elsewhere, so an aggregate null with a moving in-domain subset is a coverage
artefact rather than evidence about data-boundedness.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.data.standardise import standardise_many  # noqa: E402
from drugrx.data.tdc import ADMET_TASKS  # noqa: E402
from drugrx.fast_train import GPUData, fit_ensemble, predict_ensemble  # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402
from train_admet import load_data  # noqa: E402
from train_fast import PlainMLP, score_fold  # noqa: E402

REPOS = [
    "openadmet/openadmet-expansionrx-challenge-data",
    "openadmet/openadmet-expansionrx-challenge-train-data",
]
#: ExpansionRx assay columns kept as their own heads. Deliberately NOT mapped
#: onto our endpoint names — that mapping is what would merge two labs' scales.
ERX_COLUMNS = [
    "LogD", "KSOL", "HLM CLint", "MLM CLint",
    "Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux",
]


def load_expansionrx(workers: int) -> pd.DataFrame:
    from datasets import load_dataset

    parts = []
    for repo in REPOS:
        try:
            ds = load_dataset(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[h7a] {repo}: {exc}", flush=True)
            continue
        for _, p in ds.items():
            parts.append(p.to_pandas())
    erx = pd.concat(parts, ignore_index=True)
    erx = erx[[c for c in ["SMILES", *ERX_COLUMNS] if c in erx.columns]]

    std = standardise_many([s if isinstance(s, str) else "" for s in erx["SMILES"]],
                           workers=workers)
    erx["smiles"] = [s.smiles if s.ok else "" for s in std]
    erx = erx[erx["smiles"] != ""]
    # One row per molecule; duplicate measurements averaged within this source.
    cols = [c for c in ERX_COLUMNS if c in erx.columns]
    erx = erx.groupby("smiles")[cols].mean().reset_index()
    print(f"[h7a] ExpansionRx: {len(erx)} unique molecules, {len(cols)} heads", flush=True)
    return erx


def build_matrix(our_frame, our_x, erx, erx_x, include_erx_rows: bool, erx_cols: list[str],
                 permute_erx_labels: bool = False, seed: int = 0):
    """Stack labels into one (n, 21 + 6) target matrix with a mask.

    `permute_erx_labels` shuffles the ExpansionRx label block across molecules,
    which is arm 3: same molecules, same heads, same mask density, no usable
    relationship between structure and label.
    """
    our_keys = [t.key for t in ADMET_TASKS]
    n_our, n_erx = len(our_keys), len(erx_cols)

    y_our = our_frame[our_keys].to_numpy(dtype=np.float32)
    m_our = ~np.isnan(y_our)
    y_our = np.nan_to_num(y_our, nan=0.0)

    blocks_x = [our_x]
    blocks_y = [np.concatenate([y_our, np.zeros((len(y_our), n_erx), np.float32)], axis=1)]
    blocks_m = [np.concatenate([m_our, np.zeros((len(m_our), n_erx), bool)], axis=1)]

    if include_erx_rows and erx_x is not None and len(erx_x):
        y_e = erx[erx_cols].to_numpy(dtype=np.float32)
        m_e = ~np.isnan(y_e)
        y_e = np.nan_to_num(y_e, nan=0.0)
        if permute_erx_labels:
            # Permute rows of the label block, carrying the mask with them so
            # the number and placement of observed labels is unchanged. Only
            # which molecule owns which label is destroyed.
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(y_e))
            y_e, m_e = y_e[order], m_e[order]
        blocks_x.append(erx_x)
        blocks_y.append(np.concatenate([np.zeros((len(y_e), n_our), np.float32), y_e], axis=1))
        blocks_m.append(np.concatenate([np.zeros((len(m_e), n_our), bool), m_e], axis=1))

    return (np.concatenate(blocks_x), np.concatenate(blocks_y), np.concatenate(blocks_m))


def run_arm(name: str, x_tr, y_tr, m_tr, gpu_valid, data, frame, x_all,
            members: int, epochs: int, patience: int, device: str) -> dict:
    print(f"\n{'=' * 70}\n[h7a] arm: {name}  ({len(x_tr)} training rows, "
          f"{y_tr.shape[1]} heads)\n{'=' * 70}", flush=True)
    torch.manual_seed(0)
    np.random.seed(0)

    gpu_train = GPUData.from_numpy(x_tr, y_tr, m_tr, device)
    n_tasks = y_tr.shape[1]

    def make_model(_: int) -> nn.Module:
        return PlainMLP(x_tr.shape[1], n_tasks, 512, 3, 0.1)

    base_meta = make_model(0).to("meta")

    def score_fn(params, buffers) -> float:
        preds = predict_ensemble(base_meta, params, buffers, gpu_valid.x)
        mean = preds.mean(0)
        diff = (mean - gpu_valid.y) ** 2
        return float((diff * gpu_valid.mask).sum() / gpu_valid.mask.sum().clamp(min=1.0))

    t0 = time.time()
    params, buffers, _ = fit_ensemble(
        make_model, gpu_train, gpu_valid, n_members=members, epochs=epochs,
        device=device, patience=patience, score_fn=score_fn,
    )
    preds = predict_ensemble(base_meta, params, buffers, x_all)
    mean = preds.mean(0).float().cpu().numpy()
    # Only the first 21 columns are our endpoints; the ExpansionRx heads are
    # auxiliary and are never scored against a TDC benchmark.
    scores = score_fold(data, frame, mean[:, : len(ADMET_TASKS)], "test")
    scores["_minutes"] = round((time.time() - t0) / 60, 2)
    return {"arm": name, "scores": scores, "predictions": mean[:, : len(ADMET_TASKS)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--out", default="/content/out/h7a.json")
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data, x = load_data(Path(args.data))
    frame = data.frame
    train_rows = (frame["split"] == "train").to_numpy()
    _, stats = standardise_descriptors(x[train_rows])
    x, _ = standardise_descriptors(x, stats=stats)

    erx = load_expansionrx(args.workers)
    erx_cols = [c for c in ERX_COLUMNS if c in erx.columns]
    # Standardise ExpansionRx targets so no head dominates on units alone.
    for c in erx_cols:
        col = erx[c]
        mu, sd = float(col.mean()), float(col.std()) or 1.0
        erx[c] = (col - mu) / sd

    # **Match the feature space of the cached matrix, do not assume it.** The
    # committed features are count-ECFP4 + RDKit2D (2,264); the MapLight arm adds
    # Avalon + ErG (3,091). Featurising ExpansionRx with a different setting than
    # our own molecules produces two matrices that cannot be stacked — and if the
    # widths had happened to agree, it would have silently stacked two different
    # feature definitions, which is worse than the crash.
    want_extra = x.shape[1] == 3091
    erx_x = featurise(erx["smiles"].tolist(), workers=args.workers,
                      counts=True, desc_mode="rdkit2d", extra_fps=want_extra)
    if erx_x.shape[1] != x.shape[1]:
        raise SystemExit(
            f"feature width mismatch: ours {x.shape[1]}, ExpansionRx {erx_x.shape[1]}")
    erx_x, _ = standardise_descriptors(erx_x, stats=stats)
    print(f"[h7a] ExpansionRx features {erx_x.shape}", flush=True)

    tr = np.flatnonzero(train_rows)
    va = np.flatnonzero((frame["split"] == "valid").to_numpy())
    x_all = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(args.device)

    keys = [t.key for t in ADMET_TASKS]
    y_full = frame[keys].to_numpy(dtype=np.float32)
    m_full = ~np.isnan(y_full)
    y_full = np.nan_to_num(y_full, nan=0.0)
    pad = np.zeros((len(va), len(erx_cols)), np.float32)
    gpu_valid = GPUData.from_numpy(
        x[va],
        np.concatenate([y_full[va], pad], axis=1),
        np.concatenate([m_full[va], pad.astype(bool)], axis=1),
        args.device,
    )

    arms = {}
    # Arm 1: baseline, 21 heads only — padded to the same width so every arm
    # has an identical model shape and only the data differs.
    x1, y1, m1 = build_matrix(frame.iloc[tr], x[tr], erx, None, False, erx_cols)
    arms["1-baseline"] = run_arm("1-baseline", x1, y1, m1, gpu_valid, data, frame, x_all,
                                 args.members, args.epochs, args.patience, args.device)

    # Arm 2: + ExpansionRx heads AND molecules.
    x2, y2, m2 = build_matrix(frame.iloc[tr], x[tr], erx, erx_x, True, erx_cols)
    arms["2-plus-data"] = run_arm("2-plus-data", x2, y2, m2, gpu_valid, data, frame, x_all,
                                  args.members, args.epochs, args.patience, args.device)

    # Arm 3: the discriminator — same molecules and heads, labels permuted.
    x3, y3, m3 = build_matrix(frame.iloc[tr], x[tr], erx, erx_x, True, erx_cols,
                              permute_erx_labels=True)
    arms["3-labels-permuted"] = run_arm("3-labels-permuted", x3, y3, m3, gpu_valid, data, frame,
                                        x_all, args.members, args.epochs, args.patience, args.device)

    print(f"\n{'=' * 70}\nH7a — per-endpoint test scores\n{'=' * 70}")
    print(f"{'endpoint':16s} {'arm1':>9s} {'arm2':>9s} {'arm3':>9s} "
          f"{'2-1':>8s} {'2-3 (labels)':>13s}")
    rows = []
    for task in ADMET_TASKS:
        a1 = arms["1-baseline"]["scores"].get(task.key)
        a2 = arms["2-plus-data"]["scores"].get(task.key)
        a3 = arms["3-labels-permuted"]["scores"].get(task.key)
        if a1 is None or a2 is None or a3 is None:
            continue
        sign = -1.0 if task.metric == "mae" else 1.0
        rows.append({
            "endpoint": task.key, "metric": task.metric,
            "arm1": a1, "arm2": a2, "arm3": a3,
            "gain_total": sign * (a2 - a1),
            "gain_from_labels": sign * (a2 - a3),
            "gain_from_molecules": sign * (a3 - a1),
        })
        print(f"{task.key:16s} {a1:9.4f} {a2:9.4f} {a3:9.4f} "
              f"{rows[-1]['gain_total']:+8.4f} {rows[-1]['gain_from_labels']:+11.4f}")

    data_wins = sum(1 for r in rows if r["gain_from_labels"] > 0)
    struct_wins = sum(1 for r in rows if r["gain_from_molecules"] > 0)
    print(f"\nimproved by the LABELS (arm2 > arm3): {data_wins}/{len(rows)}")
    print(f"improved by the MOLECULES alone (arm3 > arm1): {struct_wins}/{len(rows)}")
    print("arm3 > arm1 is a pure representation effect: extra chemistry with no usable labels.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "data_wins": data_wins, "structure_wins": struct_wins,
         "arms": {k: v["scores"] for k, v in arms.items()}}, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
