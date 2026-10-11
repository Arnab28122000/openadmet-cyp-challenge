"""Build the permutation null for H7a, and place the real effect inside it.

Arm 3 was one permuted draw — a *sample* from the null, not the null. At effects
of 0.01–0.03 the permutation arm's own sampling variance is the same size as the
thing being measured, so arm 3 landing at 0.1966 says almost nothing: it could as
easily have come back at 0.1940. Comparing arm 2 against a single realisation is
how a meaningless 15/21 tally gets produced, which is the H6 failure wearing a
new costume.

So: **k permuted label matrices, k independent draws, one null distribution**,
and arm 2 reported as a **percentile within it** rather than as a win count.

The k draws run **concurrently on one device**, each as a member of the vmapped
ensemble carrying its own permuted labels (`per_member_labels`). That is the
legitimate claim on the spare VRAM — larger batches would buy gradient-noise
reduction this dataset has no use for, while independent draws are exactly the
binding resource for resolving a small effect.
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

from drugrx.data.tdc import ADMET_TASKS  # noqa: E402
from drugrx.fast_train import GPUData, fit_ensemble, predict_ensemble  # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402
from h7a_three_arm import ERX_COLUMNS, build_matrix, load_expansionrx  # noqa: E402
from train_admet import load_data  # noqa: E402
from train_fast import PlainMLP, score_fold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--out", default="/content/out/h7a_null.json")
    ap.add_argument("--draws", type=int, default=20, help="permutation draws (concurrent)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="FIXED for both arms; auto-sizing would give the real and "
                         "permuted arms different batches and make them incomparable")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--null-vs-null", action="store_true",
                    help="certification run: compare two permuted arms against each other. "
                         "Percentiles MUST centre on 0.5; anything else means the two arms "
                         "differ in something that is not the variable under test.")
    args = ap.parse_args()

    data, x = load_data(Path(args.data))
    frame = data.frame
    train_rows = (frame["split"] == "train").to_numpy()
    _, stats = standardise_descriptors(x[train_rows])
    x, _ = standardise_descriptors(x, stats=stats)

    erx = load_expansionrx(args.workers)
    erx_cols = [c for c in ERX_COLUMNS if c in erx.columns]
    for c in erx_cols:
        col = erx[c]
        mu, sd = float(col.mean()), float(col.std()) or 1.0
        erx[c] = (col - mu) / sd

    want_extra = x.shape[1] == 3091
    erx_x = featurise(erx["smiles"].tolist(), workers=args.workers,
                      counts=True, desc_mode="rdkit2d", extra_fps=want_extra)
    erx_x, _ = standardise_descriptors(erx_x, stats=stats)

    tr = np.flatnonzero(train_rows)
    va = np.flatnonzero((frame["split"] == "valid").to_numpy())
    x_all = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(args.device)

    keys = [t.key for t in ADMET_TASKS]
    y_full = frame[keys].to_numpy(dtype=np.float32)
    m_full = ~np.isnan(y_full)
    y_full = np.nan_to_num(y_full, nan=0.0)
    pad = np.zeros((len(va), len(erx_cols)), np.float32)
    gpu_valid = GPUData.from_numpy(
        x[va], np.concatenate([y_full[va], pad], 1),
        np.concatenate([m_full[va], pad.astype(bool)], 1), args.device)

    # Real arm (arm 2) and k permuted arms share one X; only labels differ.
    x2, y2, m2 = build_matrix(frame.iloc[tr], x[tr], erx, erx_x, True, erx_cols)
    perms = []
    for seed in range(args.draws):
        _, yp, mp = build_matrix(frame.iloc[tr], x[tr], erx, erx_x, True, erx_cols,
                                 permute_erx_labels=True, seed=seed)
        perms.append((yp, mp))

    n_tasks = y2.shape[1]

    def make_model(_: int) -> nn.Module:
        return PlainMLP(x2.shape[1], n_tasks, 512, 3, 0.1)

    base_meta = make_model(0).to("meta")

    def run(label_sets: list[tuple[np.ndarray, np.ndarray]], tag: str) -> list[dict]:
        """Train one member per label set, all concurrently.

        **Batch size is fixed, not auto-sized.** The auto-sizer scales the batch
        with the member count, so a 1-member real arm and a 20-member null arm
        received different batches and therefore different numbers of optimiser
        steps per epoch — at a short budget that difference swamped the effect
        being measured, and produced a table where almost every endpoint scored
        *below* its own permutation null. Same batch, same epochs, same
        initialisation seeds: only the labels differ.
        """
        ys = np.stack([a for a, _ in label_sets])
        ms = np.stack([b for _, b in label_sets])
        gd = GPUData(
            x=torch.from_numpy(np.ascontiguousarray(x2, np.float32)).to(args.device),
            y=torch.from_numpy(np.ascontiguousarray(ys, np.float32)).to(args.device),
            mask=torch.from_numpy(np.ascontiguousarray(ms)).to(args.device),
            n=x2.shape[0],
        )
        print(f"[null] {tag}: {len(label_sets)} concurrent draws, "
              f"{gd.bytes()/1e9:.2f} GB resident", flush=True)
        t0 = time.time()
        params, buffers, _ = fit_ensemble(
            make_model, gd, gpu_valid, n_members=len(label_sets),
            per_member_labels=True, epochs=args.epochs, device=args.device,
            batch_size=args.batch_size,
            patience=args.epochs,  # fixed budget: every draw must see the same training
        )
        preds = predict_ensemble(base_meta, params, buffers, x_all)
        out = []
        for i in range(preds.shape[0]):
            arr = preds[i].float().cpu().numpy()[:, : len(ADMET_TASKS)]
            out.append(score_fold(data, frame, arr, "test"))
        print(f"[null] {tag} done in {(time.time()-t0)/60:.2f} min", flush=True)
        return out

    # **The real arm is replicated too.** A single real run against a 20-draw
    # null compares a point to a distribution and attributes all of the real
    # arm's initialisation variance to the effect. Identical labels, different
    # initialisations, same count as the null.
    if args.null_vs_null:
        # **The fixture that certifies the comparison path itself.** Two
        # permutation arms differ in nothing but their shuffle, so their
        # percentiles must centre on 0.5. The batch-size defect — where the real
        # arm and the null arm received different batches because the auto-sizer
        # scales with member count — would have shown up here immediately as a
        # badly off-centre distribution. That was a broken experimental control
        # rather than a bug: the two arms differed in a hyperparameter that was
        # not the variable under test.
        half = args.draws
        extra = []
        for seed in range(1000, 1000 + half):
            _, yp, mp = build_matrix(frame.iloc[tr], x[tr], erx, erx_x, True, erx_cols,
                                     permute_erx_labels=True, seed=seed)
            extra.append((yp, mp))
        real_runs = run(extra, "permuted-A (stands in for real)")
        null = run(perms, "permuted-B")
    else:
        real_runs = run([(y2, m2)] * args.draws, "real")
        null = run(perms, "permuted")

    print(f"\n{'endpoint':16s} {'real':>9s} {'null mean':>10s} {'null sd':>8s} "
          f"{'percentile':>11s} {'verdict':>12s}")
    rows = []
    for task in ADMET_TASKS:
        reals = np.array([d[task.key] for d in real_runs if task.key in d])
        vals = np.array([d[task.key] for d in null if task.key in d])
        if len(vals) < 5 or len(reals) < 5:
            continue
        r = float(reals.mean())
        better_is_lower = task.metric == "mae"
        # Percentile of the real effect within the null, oriented so high = good.
        pct = float((vals > r).mean() if better_is_lower else (vals < r).mean())
        verdict = "above null" if pct >= 0.95 else ("below null" if pct <= 0.05 else "inside null")
        rows.append({"endpoint": task.key, "metric": task.metric, "real": r,
                     "real_sd": float(reals.std()),
                     "null_mean": float(vals.mean()), "null_sd": float(vals.std()),
                     "percentile": pct, "verdict": verdict, "draws": int(len(vals))})
        print(f"{task.key:16s} {r:9.4f} {vals.mean():10.4f} {vals.std():8.4f} "
              f"{pct:11.2f} {verdict:>12s}")

    above = sum(1 for r in rows if r["verdict"] == "above null")
    print(f"\nendpoints clearing the null's 95th percentile: {above}/{len(rows)}")
    print("A 15/21 win count against ONE permuted draw says nothing; this does.")

    mean_pct = float(np.mean([r["percentile"] for r in rows]))
    above_median = sum(1 for r in rows if r["percentile"] > 0.5)
    print(f"{above_median}/{len(rows)} endpoints above the null median; "
          f"mean percentile {mean_pct:.3f} (0.5 = no effect)")
    if args.null_vs_null:
        print("CERTIFICATION: mean percentile must be ~0.5. "
              f"{'PASS' if 0.40 <= mean_pct <= 0.60 else 'FAIL — the arms differ in something untested'}")

    Path(args.out).write_text(json.dumps(
        {"rows": rows, "above_null": above, "draws": args.draws,
         "mean_percentile": mean_pct, "above_median": above_median,
         "null_vs_null": args.null_vs_null,
         # Raw per-replicate scores: the endpoints share an encoder, share
         # molecules across folds and are biologically coupled, so a sign test
         # treating 21 endpoints as 21 independent trials overstates its case.
         # These allow an effective-n estimate and an endpoint-level permutation.
         "raw_real": real_runs, "raw_null": null}, indent=2, default=float))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
