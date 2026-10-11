"""Build the multi-task ADMET table and its features, on the laptop.

The L4 runtime has four vCPU, so nothing here runs on the GPU box: this produces
one parquet and one npz, and the VM loads them.

    python scripts/prepare_data.py --out data/
    python scripts/prepare_data.py --out data/ --push <hf-user>/drugrx-admet

With `--push`, the artifacts go to a private Hugging Face dataset repo so a
fresh VM pulls them in seconds. Needs HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from drugrx.data.multitask import build
from drugrx.data.tdc import ADMET_TASKS
from drugrx.features import featurise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--seed", type=int, default=0, help="scaffold split seed")
    ap.add_argument("--push", default="", help="HF dataset repo to upload to")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--extra-fps", action="store_true",
                    help="add Avalon + ErG (MapLight's recipe): 2264 -> 3091 features. "
                         "The shipped ensemble was trained at 3091, so a cache built "
                         "without this will not load its checkpoint.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = build(ADMET_TASKS, seed=args.seed, cache=out / "admet_multitask.parquet")

    smiles = data.frame["smiles"].tolist()
    x = featurise(smiles, cache=out / "features.npz", workers=args.workers,
                  extra_fps=args.extra_fps)
    print(f"[prep] features {x.shape}")

    # Per-split label counts: the number that tells you whether an endpoint is
    # even trainable. An endpoint with 30 test labels cannot support a claim.
    counts = {}
    for task in ADMET_TASKS:
        col = data.frame[task.key]
        fold_col = f"{task.key}__fold"
        if fold_col in data.frame:
            # Scoring folds: each endpoint on exactly TDC's own molecules.
            counts[task.key] = {
                split: int((data.frame[fold_col] == split).sum())
                for split in ("train", "valid", "test")
            }
        else:
            counts[task.key] = {
                split: int(col[data.frame["split"] == split].notna().sum())
                for split in ("train", "valid", "test")
            }

    meta = {
        "n_molecules": int(len(data.frame)),
        "n_features": int(x.shape[1]),
        # Recorded so a checkpoint/cache mismatch is detectable rather than a
        # shape error three scripts downstream.
        "extra_fps": bool(args.extra_fps),
        "split_seed": args.seed,
        "tasks": [t.key for t in ADMET_TASKS],
        "norm": {k: list(v) for k, v in data.norm.items()},
        "label_counts": counts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nlabels per split:")
    for key, c in counts.items():
        flag = "  <-- thin test set" if c["test"] < 50 else ""
        print(f"  {key:18s} train {c['train']:6d}  valid {c['valid']:5d}  test {c['test']:5d}{flag}")

    if args.push:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("[prep] --push given but HF_TOKEN is unset; skipping upload")
            return 1
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.push, repo_type="dataset", private=True, exist_ok=True)
        for name in ("admet_multitask.parquet", "features.npz", "meta.json"):
            api.upload_file(
                path_or_fileobj=str(out / name),
                path_in_repo=name,
                repo_id=args.push,
                repo_type="dataset",
            )
            print(f"[prep] pushed {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
