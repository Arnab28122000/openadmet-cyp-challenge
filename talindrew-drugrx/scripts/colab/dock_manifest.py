"""Build `smiles.json`, the docking run's input manifest -- and the alignment contract.

This file exists because the first docking build did not have it. The manifest was typed into
a heredoc on the VM, so when that VM was reaped the ordering that every downstream artefact is
indexed by existed nowhere. `dock_cyp3a4.jsonl` keys rows by integer `i`, `desc3d.npz` is a
dense matrix in the same order, and `cyp_dock_eval.py` assigns both straight onto
`data.frame` rows. If the order is rebuilt even slightly differently -- a different dedup, a
different sort, a different source -- every docking feature lands on the wrong molecule and
the result is not a failed run but a *plausible* one, which is worse.

So the order is pinned to one source, the same call the evaluator makes:

    drugrx.cyp.data.load().frame["smiles"]

and the manifest carries a checksum of that list. `dock_run.py` and `cyp_dock_eval.py` both
verify it before using anything.

Usage (local, not on the VM):

    python scripts/colab/dock_manifest.py            # writes data/cyp_cache/dock/smiles.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/cyp_cache/dock"


def digest(smiles: list[str]) -> str:
    h = hashlib.sha256()
    for s in smiles:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def main() -> int:
    from drugrx.cyp.data import load

    data = load(workers=8)
    train = data.frame.reset_index(drop=True)["smiles"].tolist()
    test = data.test.reset_index(drop=True)["smiles"].tolist()

    # Train FIRST, then the blinded test set. `cyp_dock_eval.py` takes n = len(frame) and
    # keeps only rows with i < n, so this order is what makes the first block the train
    # block. Reversing it would silently feed test-set docking scores to the evaluator.
    smiles = train + test
    sha = digest(smiles)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smiles.json").write_text(json.dumps(smiles))
    (OUT / "manifest.json").write_text(json.dumps(
        {"n": len(smiles), "n_train": len(train), "n_test": len(test), "sha": sha,
         "order": "data.frame['smiles'] then data.test['smiles']",
         "source": "drugrx.cyp.data.load()"},
        indent=1))

    print(f"[manifest] {len(train)} train + {len(test)} test = {len(smiles)}, sha {sha}")
    print(f"[manifest] wrote {OUT/'smiles.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
