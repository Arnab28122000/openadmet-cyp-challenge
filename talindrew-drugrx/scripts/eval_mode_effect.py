"""How much did predicting with dropout on cost us?

`predict_ensemble` never called `.eval()`, so every neural number this programme
produced ran with dropout active at 0.1 (see `PROGRESS.md`, defect 6). The fix is
one line; this script measures what the line is worth, on the **shipped**
checkpoint, with no retraining — so the comparison is the same weights scored two
ways and nothing else moves.

**The defect was never "dropout is wrong at inference".** It is that nobody chose
it, nobody could reproduce a number produced under it, it applied to the neural arm
and to nothing it was compared against, and the member spread it inflated was being
read as calibrated uncertainty. All four are fixed by making it a flag with a
deterministic default; whether `mc_dropout=True` is *better* is then an ordinary
hyper-parameter question, to be settled on validation like any other.

Two quantities matter, and they are different:

* **The score shift.** Note what this is *not*: inference-time dropout is not pure
  noise. With K members each drawing an independent mask and the predictions then
  averaged, it **is MC-dropout ensembling** — a second source of averaging on top
  of initialisation diversity, and one that can genuinely help. So the expected
  sign is not obvious, and a mixed result is informative rather than suspicious.
  (It came out mixed: better on 18 of 21 here, worse on 3, mean gain +0.0024. The
  concurrent CYP work measured the opposite sign on its own four endpoints at
  dropout 0.15 with 24 members, where the extra averaging has more to work with.)
* **The spread shift.** This is the one that bears on Phase 4. The across-member
  standard deviation is what we were about to serve as calibrated uncertainty, and
  with dropout on part of it is a member disagreeing *with itself*. The honest
  uncertainty is the eval-mode spread; the difference between the two is noise we
  would have been selling as knowledge.

    python scripts/eval_mode_effect.py --artifact /content/artifact/ensemble.pt

Writes `eval_mode.json`. Reports per endpoint, and in aggregate, how many endpoints
move in which direction — a direction count is reported *with* the magnitudes, never
instead of them (T5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.data.tdc import ADMET_TASKS, LOWER_IS_BETTER  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/content/out/eval_mode.json")
    ap.add_argument("--repeats", type=int, default=5,
                    help="dropout-on passes; one pass is a sample, not the behaviour")
    args = ap.parse_args()

    import torch

    from drugrx.fast_train import predict_ensemble
    from drugrx.features import standardise_descriptors
    from train_admet import load_data
    from train_fast import PlainMLP, score_fold

    blob = torch.load(args.artifact, map_location="cpu", weights_only=False)
    data, x = load_data(Path(args.data))
    frame = data.frame
    if x.shape[1] != blob["in_dim"]:
        raise SystemExit(f"features {x.shape[1]} vs checkpoint {blob['in_dim']}")

    stats = (np.array(blob["feature_stats"]["mean"], dtype=np.float32),
             np.array(blob["feature_stats"]["std"], dtype=np.float32))
    x, _ = standardise_descriptors(x, stats=stats)

    arch = blob["architecture"]
    base = PlainMLP(blob["in_dim"], blob["n_tasks"], arch["d_model"],
                    arch["n_layers"], arch["dropout"]).to("meta")
    params = {k: v.to(args.device) for k, v in blob["params"].items()}
    buffers = {k: v.to(args.device) for k, v in blob["buffers"].items()}
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(args.device)

    n_tdc = len(ADMET_TASKS)

    def pass_once(mc: bool) -> tuple[dict, np.ndarray]:
        preds = predict_ensemble(base, params, buffers, xt, mc_dropout=mc)
        mean = preds.mean(0).float().cpu().numpy()[:, :n_tdc]
        spread = preds.std(0).float().cpu().numpy()[:, :n_tdc]
        return score_fold(data, frame, mean, "test"), spread

    torch.manual_seed(0)
    fixed, fixed_spread = pass_once(False)
    again, _ = pass_once(False)
    assert all(abs(fixed[k] - again[k]) < 1e-9 for k in fixed), \
        "eval mode is not deterministic — the fix is not doing what it claims"

    # Dropout on is a *sample*; one pass would tell us about one mask (T5 again).
    noisy_runs, noisy_spreads = [], []
    for seed in range(args.repeats):
        torch.manual_seed(seed)
        scores, spread = pass_once(True)
        noisy_runs.append(scores)
        noisy_spreads.append(spread)

    rows = []
    better_count = 0
    for i, task in enumerate(ADMET_TASKS):
        if task.key not in fixed:
            continue
        noisy = [r[task.key] for r in noisy_runs if task.key in r]
        if not noisy:
            continue
        sign = -1.0 if task.metric in LOWER_IS_BETTER else 1.0
        gain = sign * (fixed[task.key] - float(np.mean(noisy)))
        better_count += int(gain > 0)

        col = f"{task.key}__fold"
        mask = (frame[col] == "test").to_numpy() if col in frame.columns else None
        sp_fixed = float(np.mean(fixed_spread[mask, i])) if mask is not None else float("nan")
        sp_noisy = float(np.mean([np.mean(s[mask, i]) for s in noisy_spreads])) \
            if mask is not None else float("nan")

        rows.append({
            "task": task.key, "metric": task.metric,
            "eval_mode": fixed[task.key],
            "dropout_on_mean": float(np.mean(noisy)),
            "dropout_on_std": float(np.std(noisy)),
            "gain": gain,
            "spread_eval": sp_fixed,
            "spread_dropout_on": sp_noisy,
            "spread_inflation": (sp_noisy / sp_fixed) if sp_fixed else float("nan"),
        })

    infl = [r["spread_inflation"] for r in rows if np.isfinite(r["spread_inflation"])]
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifact": args.artifact,
        "repeats": args.repeats,
        "endpoints": len(rows),
        "endpoints_improved_by_eval_mode": better_count,
        "mean_gain": float(np.mean([r["gain"] for r in rows])),
        "median_spread_inflation": float(np.median(infl)) if infl else None,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))

    print(f"\n{'endpoint':16s} {'metric':9s} {'eval':>9s} {'drop-on':>9s} "
          f"{'gain':>9s} {'spread x':>9s}")
    for r in sorted(rows, key=lambda r: -r["gain"]):
        print(f"{r['task']:16s} {r['metric']:9s} {r['eval_mode']:9.4f} "
              f"{r['dropout_on_mean']:9.4f} {r['gain']:+9.4f} "
              f"{r['spread_inflation']:9.2f}")
    print(f"\neval mode is better on {better_count}/{len(rows)}; "
          f"mean gain {payload['mean_gain']:+.4f}; "
          f"median spread inflation {payload['median_spread_inflation']:.2f}x")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
