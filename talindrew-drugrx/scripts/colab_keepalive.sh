#!/usr/bin/env bash
# Keep a Colab session alive while detached work runs on it.
#
# WHY THIS EXISTS. The CLI's own README states the rule: "Sessions stay alive as long
# as the **kernel** is active." Every long run in this programme is launched with
# `nohup ... &` so that `colab exec` can return without killing it — which means the
# kernel goes idle the moment that exec returns, and Colab reaps the session even
# though the GPU is saturated.
#
# Four sessions died on 2026-10-05 and this explains all four. `cyp-r3` survived ~90
# minutes while monitors polled it with `colab exec` every 60-90s; `cyp-h12` died in
# ~15-20 minutes after the monitor was stopped, leaving only `colab download`, which is
# a file transfer and not kernel activity. **The monitors had been acting as accidental
# keepalives, and removing one killed the run.**
#
# So: a deliberate heartbeat. One trivial `colab exec` per interval, which is kernel
# activity by definition.
#
#   scripts/colab_keepalive.sh <session> [interval_seconds] [max_hours]
#
# Runs until the session disappears, the deadline passes, or it is killed. It never
# touches the training process; a failed heartbeat is logged and retried, because one
# dropped request must not end the watch.
set -uo pipefail

SESSION="${1:?usage: colab_keepalive.sh <session> [interval_seconds] [max_hours]}"
INTERVAL="${2:-45}"
MAX_HOURS="${3:-12}"
COLAB="${COLAB:-$HOME/.local/bin/colab}"

BEAT="$(mktemp -t colab_beat_XXXXXX.py)"
# Deliberately trivial, and deliberately not silent: printing the GPU's memory makes
# each heartbeat double as a liveness probe of the thing we actually care about.
cat > "$BEAT" <<'PY'
import subprocess
g = subprocess.run(["bash", "-lc",
                    "nvidia-smi --query-gpu=memory.used,utilization.gpu "
                    "--format=csv,noheader 2>/dev/null | head -1"],
                   capture_output=True, text=True).stdout.strip()
n = subprocess.run(["bash", "-lc", "pgrep -c python 2>/dev/null || echo 0"],
                   capture_output=True, text=True).stdout.strip()
print(f"beat gpu=[{g}] python_procs={n}")
PY
trap 'rm -f "$BEAT"' EXIT

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
beats=0; fails=0
echo "[keepalive] $SESSION every ${INTERVAL}s for up to ${MAX_HOURS}h"
while [ "$(date +%s)" -lt "$deadline" ]; do
  if out=$("$COLAB" exec -s "$SESSION" -f "$BEAT" 2>&1); then
    beats=$((beats + 1)); fails=0
    [ $((beats % 20)) -eq 1 ] && echo "[keepalive] beat $beats: $(echo "$out" | grep -o 'beat .*' | head -1)"
  else
    fails=$((fails + 1))
    echo "[keepalive] heartbeat failed ($fails consecutive)"
    # Three in a row means the session is gone, not that one request dropped.
    if [ "$fails" -ge 3 ]; then
      echo "[keepalive] $SESSION appears gone after $beats beats; stopping"
      exit 1
    fi
  fi
  sleep "$INTERVAL"
done
echo "[keepalive] deadline reached after $beats beats"
