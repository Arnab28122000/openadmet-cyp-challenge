"""Paired replication of base vs deep, on BOTH statistics that differ.

The depth result (0.4289 vs 0.4377) is one draw each, against a bootstrap sd of
~0.03. This programme has been wrong at exactly that ratio before — H6 killed a
10-vs-11 tally, and a batch-size mismatch produced a table of impossible results.
On current evidence the PubChem gain (+0.0029, 5/5 paired seeds) is better
supported than the depth gain, despite being three times smaller.

**But the score is the weaker of the two claims.** The best epoch moved 20 -> 65
under identical schedule, patience and data order. Two models differing only by
seed land at similar best epochs; two that stop improving 45 epochs apart are
solving a different optimisation problem. A shifted optimum is much harder to
manufacture from seed noise than a 0.009 score gap, so both are recorded here and
they can disagree:

* score replicates, epoch doesn't -> probably noise that happened to look like depth
* epoch replicates, score doesn't -> **depth changed the optimisation problem**
  even though it did not buy accuracy. That is the more interesting outcome and
  the one that would genuinely bear on the capacity question.

Paired: identical seeds, identical split, identical data order for both arms.
Pairing bought ~9x in H7a and costs nothing here.

Width is deliberately NOT compared against base at 32 members. Base already peaks
at 37.1 of 40 GB, so 2048-wide only fits with fewer members — and comparing a
wide/few-member model against a narrow/many-member one confounds capacity with
ensemble size. Members buy decorrelation, which is where this track's largest
measured gain came from (the blend, +0.028); width buys capacity, which has one
unreplicated point. Width is therefore tested at MATCHED member count or not at
all.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ARMS = {
    "rep-base": dict(d_model=1024, n_layers=3, members=24),
    "rep-deep": dict(d_model=1024, n_layers=5, members=24),
    # matched-member width test: same members as the two above, so the only
    # difference against rep-base is width.
    "rep-wide": dict(d_model=2048, n_layers=3, members=24),
}


def _push(path: Path, repo: str) -> None:
    """Upload one result file to the Hub; never fail the run over it."""
    if not repo or not path.exists() or not os.environ.get("HF_TOKEN"):
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for p in (path, path.with_suffix(".npz")):
            if p.exists():
                api.upload_file(path_or_fileobj=str(p), path_in_repo=f"cyp_rep/{p.name}",
                                repo_id=repo, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"[rep] hub push failed ({exc}); result is on the VM only", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--out-dir", default="/content/out/cyp_rep")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--summary", default="/content/out/cyp/replicate.json")
    ap.add_argument("--hub-repo", default="arnab2812/drugrx-cyp-runs")
    args = ap.parse_args()

    for s in range(args.seed_offset, args.seed_offset + args.seeds):
        for tag in args.arms:
            cfg = ARMS[tag]
            done = Path(args.out_dir) / f"{tag}_s{s}.json"
            if done.exists():
                print(f"[rep] {tag} s{s}: cached", flush=True); continue
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(s),
                 "--d-model", str(cfg["d_model"]), "--n-layers", str(cfg["n_layers"]),
                 "--members", str(cfg["members"]), "--dropout", "0.15",
                 "--epochs", str(args.epochs), "--patience", str(args.patience),
                 "--out-dir", args.out_dir],
                cwd="/content", env={**os.environ, "PYTHONPATH": "/content"},
                capture_output=True, text=True)
            tail = [x for x in r.stdout.splitlines() if x.startswith(f"[net] {tag}")]
            print(tail[-1] if tail else
                  f"[rep] {tag} s{s} FAILED rc={r.returncode}: "
                  f"{(r.stderr.strip().splitlines() or ['no stderr'])[-1][:150]}", flush=True)
            # **Persist immediately.** A Colab Pro+ runtime died mid-sweep and took
            # its completed results with it, because they only existed on the VM's
            # disk. Results are small; pushing each one as it lands makes the whole
            # experiment survive a VM that disappears, which this tier's runtimes do.
            _push(done, args.hub_repo)

    runs = {}
    for f in Path(args.out_dir).glob("*.json"):
        d = json.loads(f.read_text())
        runs.setdefault(d["tag"], {})[d["seed"]] = d
    if "rep-base" in runs and "rep-deep" in runs:
        shared = sorted(set(runs["rep-base"]) & set(runs["rep-deep"]))
        if shared:
            bs = np.array([runs["rep-base"][s]["valid"] for s in shared])
            ds = np.array([runs["rep-deep"][s]["valid"] for s in shared])
            be = np.array([runs["rep-base"][s]["best_epoch"] for s in shared])
            de = np.array([runs["rep-deep"][s]["best_epoch"] for s in shared])
            print(f"\npaired over {len(shared)} seeds")
            print(f"  score  base {bs.mean():.4f}+/-{bs.std():.4f}  deep {ds.mean():.4f}+/-{ds.std():.4f}"
                  f"  gain {(bs-ds).mean():+.4f}+/-{(bs-ds).std():.4f}"
                  f"  ({int(((bs-ds)>0).sum())}/{len(shared)} seeds deep better)")
            print(f"  epoch  base {be.mean():.1f}+/-{be.std():.1f}  deep {de.mean():.1f}+/-{de.std():.1f}"
                  f"  shift {(de-be).mean():+.1f}+/-{(de-be).std():.1f}"
                  f"  ({int(((de-be)>0).sum())}/{len(shared)} seeds later)")
            Path(args.summary).write_text(json.dumps(
                {"seeds": shared, "base_score": bs.tolist(), "deep_score": ds.tolist(),
                 "base_epoch": be.tolist(), "deep_epoch": de.tolist(),
                 "score_gain_mean": float((bs-ds).mean()), "score_gain_sd": float((bs-ds).std()),
                 "epoch_shift_mean": float((de-be).mean()), "epoch_shift_sd": float((de-be).std())},
                indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
