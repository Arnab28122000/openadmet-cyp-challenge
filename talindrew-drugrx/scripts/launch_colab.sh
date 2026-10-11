#!/usr/bin/env bash
# Provision an L4, ship the code and the cached data, and start training.
#
#   scripts/launch_colab.sh <session-name> [extra args for train_admet.py]
#
# Everything here is deliberately idempotent: a session that already exists is
# reused, an upload that is already there is cheap, and the trainer resumes from
# its checkpoint. Re-running this after a dropped session is the normal way to
# continue a 24h run, not an error path.
#
# It never runs `colab auth` or `colab drivemount` — both need a TTY and hang
# when driven by an agent. Checkpoints go to the Hugging Face Hub instead.

set -euo pipefail

SESSION="${1:?usage: launch_colab.sh <session-name> [train args...]}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

COLAB="${COLAB:-colab}"
GPU="${GPU:-L4}"   # L4 only — this programme is L4-only by instruction.

say() { printf '\n\033[1m[launch]\033[0m %s\n' "$*"; }

say "session '$SESSION' on --gpu $GPU"
if $COLAB status -s "$SESSION" >/dev/null 2>&1; then
  say "session already exists; reusing it"
else
  # An unrecognised --gpu value silently falls back to A100, so the smoke test
  # below asserts the device name rather than trusting this call.
  $COLAB new -s "$SESSION" --gpu "$GPU"
fi

say "asserting we actually got an L4"
$COLAB exec -s "$SESSION" -f scripts/smoke_l4.py

say "installing deps on the VM"
$COLAB install -s "$SESSION" lightgbm 'huggingface_hub>=0.25' || true

say "uploading the package and the cached data"
# The VM has four vCPU: it must never featurise. These are the precomputed
# tensors, ~8 MB, so this is seconds rather than minutes.
for f in data/admet_multitask.parquet data/features.npz data/meta.json; do
  [ -f "$f" ] || { echo "missing $f — run scripts/prepare_data.py first" >&2; exit 1; }
  $COLAB upload -s "$SESSION" "$f" "/content/$f"
done

# The package itself, so `import drugrx` works on the VM.
tar czf /tmp/drugrx-src.tgz drugrx
$COLAB upload -s "$SESSION" /tmp/drugrx-src.tgz /content/drugrx-src.tgz
echo 'import subprocess, sys
subprocess.run(["tar","xzf","/content/drugrx-src.tgz","-C","/content"], check=True)
sys.path.insert(0, "/content")
import drugrx; print("drugrx importable on the VM")' | $COLAB exec -s "$SESSION"

say "starting training (resumes automatically if a checkpoint exists)"
$COLAB exec -s "$SESSION" -f scripts/train_admet.py -- \
  --data /content/data --run-dir /content/runs --device cuda "$@"

say "done. Monitor with:"
echo "  python scripts/monitor.py --session $SESSION --run <run-id>"
echo "Stop the VM when finished — an idle VM burns compute units:"
echo "  $COLAB stop -s $SESSION"
