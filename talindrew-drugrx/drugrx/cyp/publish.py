"""Push a finished per-seed result to the Hub the moment it lands.

A Colab Pro+ runtime dies mid-experiment — `colabtools` #5950 — and on
2026-10-05 four sessions went at once. `cyp_long.py` survived it because it
uploaded each seed's JSON as soon as the seed finished, so a dead VM cost the
run in progress and nothing else. `cyp_loss_ablation.py` and
`cyp_aux_ablation.py` had the same loop and no upload, so the 10-seed loss
ablation and the whole four-arm auxiliary ablation were lost entirely and had to
start from zero.

The difference was thirteen lines of inline code in one of three drivers. That is
exactly the kind of asymmetry a shared helper exists to remove, so the push lives
here and every driver calls it.

Two deliberate properties:

* **Never fatal.** A failed upload must not kill a run that has already produced
  its result; the local file is still there and the next push retries.
* **Silent without a token.** An agent-driven session may have no `HF_TOKEN`, in
  which case results are pulled with `colab download` instead. The helper says so
  once rather than warning per seed, because a warning printed 40 times is a
  warning nobody reads.
"""
from __future__ import annotations

import os
from pathlib import Path

HUB_REPO_DEFAULT = "arnab2812/drugrx-cyp-runs"

_warned = False


def push_result(json_path: Path | str, subdir: str,
                repo: str = HUB_REPO_DEFAULT) -> bool:
    """Upload `<stem>.json` and its `.npz` siblings under `subdir/`.

    Returns True when something was uploaded. The `.npz` carries the saved
    predictions, which is what every downstream re-analysis (bin reads, offset
    fitting, blending) needs — a JSON of summary scores alone cannot be
    re-measured under a corrected metric, and this programme has had to do that.
    """
    global _warned
    p = Path(json_path)
    if not os.environ.get("HF_TOKEN"):
        if not _warned:
            print("[publish] HF_TOKEN unset; results stay on the VM "
                  "(pull them with `colab download`)", flush=True)
            _warned = True
        return False
    if not p.exists():
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for f in (p, p.with_suffix(".npz")):
            if f.exists():
                api.upload_file(path_or_fileobj=str(f),
                                path_in_repo=f"{subdir}/{f.name}",
                                repo_id=repo, repo_type="dataset")
        return True
    except Exception as exc:  # noqa: BLE001 — a push must never kill a run
        print(f"[publish] push failed for {p.name}: {exc}", flush=True)
        return False
