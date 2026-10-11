#!/usr/bin/env bash
# Copy a running docking job's JSONL back to the laptop, repeatedly.
#
# WHY THIS EXISTS. `dock_run.py` appends to a JSONL after every ligand and skips on restart,
# which makes it resumable against a dead *process*. It is not resumable against a dead *VM*,
# because the JSONL lives on the VM's disk. On 2026-10-09 a 4.6 h run was lost exactly that
# way: the session was reaped, and the only copy of 4+ hours of docking went with it.
#
# `colab_keepalive.sh` reduces how often that happens. This reduces what it costs when it does.
# The two are complements, not alternatives -- a keepalive is best-effort, a local copy is not.
#
#   scripts/dock_pull.sh <session> <remote_basename> [interval_seconds] [max_hours]
#
# <remote_basename> is the file inside /content/dock to copy, e.g. dock_cyp3a4_s0of3.jsonl.
# It lands locally under data/cyp_cache/dock/ with the same name, so sharded runs from
# several sessions never overwrite each other and merge by concatenation.
#
# Downloads are idempotent (whole-file overwrite of a growing append-only log), so a pull that
# lands mid-append simply gets a slightly shorter file and the next one catches up. A failed
# pull is logged and retried; unlike the keepalive this never gives up on its own, because the
# session can come back and the next pull should still happen.
set -uo pipefail

SESSION="${1:?usage: dock_pull.sh <session> <remote_basename> [interval] [max_hours]}"
BASENAME="${2:?usage: dock_pull.sh <session> <remote_basename> [interval] [max_hours]}"
INTERVAL="${3:-300}"
MAX_HOURS="${4:-11}"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/data/cyp_cache/dock"
REMOTE="/content/dock/$BASENAME"

mkdir -p "$DEST"
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
pulls=0
echo "[pull] $SESSION:$BASENAME -> $DEST every ${INTERVAL}s for up to ${MAX_HOURS}h"

while [ "$(date +%s)" -lt "$deadline" ]; do
  # Download to a temp name first: a half-transferred file must never replace a good one.
  if "$COLAB" download -s "$SESSION" "$REMOTE" "$DEST/.$BASENAME.part" > /dev/null 2>&1; then
    if [ -s "$DEST/.$BASENAME.part" ]; then
      mv "$DEST/.$BASENAME.part" "$DEST/$BASENAME"
      pulls=$((pulls + 1))
      n=$(wc -l < "$DEST/$BASENAME" | tr -d ' ')
      echo "[pull] $(date -u '+%H:%M:%S') $BASENAME pull $pulls: $n rows"
    fi
  else
    echo "[pull] $(date -u '+%H:%M:%S') download failed (session gone or busy); retrying"
  fi
  sleep "$INTERVAL"
done
echo "[pull] deadline reached after $pulls pulls"
