#!/usr/bin/env bash
# Sync training state between a Colab VM and the laptop, so a dead session resumes.
#
#   scripts/colab_sync.sh pull    <session> [out_dir]   # VM -> laptop, repeatedly
#   scripts/colab_sync.sh restore <session> [out_dir]   # laptop -> fresh VM, once
#
# `drugrx/cyp/resume.py` already checkpoints full training state (stacked params and
# buffers, Adam's moments, the scheduler, epoch/best/patience counters, the separate
# best-so-far state, and the RNG) and resumes from it automatically. Its preferred
# durable store is the Hugging Face Hub, which **needs `HF_TOKEN`** and silently no-ops
# without one -- that is how a session death on 2026-10-05 destroyed a run whose push
# had already been wired. `EnsembleCheckpoint.load()` checks the **local directory
# first** and only then the Hub, so pulling the `.pt` files to the laptop and pushing
# them back onto a fresh VM is an exact token-free substitute.
#
# SIZE, BECAUSE IT DETERMINES WHETHER THIS IS WORTH RUNNING. One checkpoint of a
# 16-member d1024/L6 ensemble measures **2.1 GB** (params + Adam's two moments + a full
# duplicate of `best_state`). So:
#
#   * **Short runs (minutes):** do not use this. Pass `--ckpt-every 0` and rely on
#     `colab_keepalive.sh` to stop the death in the first place; a lost 3-minute run
#     costs less than moving 2.1 GB. Pull the 34 KB result files instead.
#   * **Long runs (hours):** this is what it is for. One 2.1 GB transfer protects hours.
#
# Pulls are newest-first and skip a file already present with the same size, so an
# interrupted sync resumes rather than restarting.
set -uo pipefail

MODE="${1:?usage: colab_sync.sh {pull|restore} <session> [out_dir]}"
SESSION="${2:?usage: colab_sync.sh {pull|restore} <session> [out_dir]}"
OUT="${3:-data/cyp_cache/ckpt}"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
REMOTE="/content/ckpt"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"

list_remote() {
  local tmp; tmp="$(mktemp -t colab_ls_XXXXXX.py)"
  cat > "$tmp" <<'PY'
import pathlib
d = pathlib.Path("/content/ckpt")
if d.exists():
    for p in sorted(d.glob("*.pt"), key=lambda q: -q.stat().st_mtime):
        print(f"{p.name} {p.stat().st_size}")
PY
  "$COLAB" exec -s "$SESSION" -f "$tmp" 2>/dev/null | grep -E '\.pt [0-9]+$' || true
  rm -f "$tmp"
}

case "$MODE" in
  pull)
    echo "[sync] pulling $REMOTE from $SESSION into $OUT (newest first)"
    for i in $(seq 1 500); do
      moved=0
      while read -r name size; do
        [ -z "${name:-}" ] && continue
        local_path="$OUT/$name"
        if [ -f "$local_path" ] && [ "$(wc -c < "$local_path" | tr -d ' ')" = "$size" ]; then
          continue                       # already have this exact bytes-count
        fi
        echo "[sync] $name ($(( size / 1048576 )) MB)"
        if "$COLAB" download -s "$SESSION" "$REMOTE/$name" "$local_path" >/dev/null 2>&1; then
          moved=$((moved + 1))
        else
          echo "[sync] FAILED $name (will retry)"
          rm -f "$local_path"            # a partial checkpoint is worse than none
        fi
      done < <(list_remote)
      [ "$moved" -gt 0 ] && echo "[sync] $moved file(s) this pass"
      # Stop when the session is gone; there is nothing left to pull.
      "$COLAB" status -s "$SESSION" >/dev/null 2>&1 || {
        echo "[sync] $SESSION is gone; final state is in $OUT"; exit 0; }
      sleep 120
    done
    ;;
  restore)
    n=0
    shopt -s nullglob
    for f in "$OUT"/*.pt; do
      echo "[sync] restoring $(basename "$f") ($(( $(wc -c < "$f") / 1048576 )) MB)"
      if "$COLAB" upload -s "$SESSION" "$f" "$REMOTE/$(basename "$f")" >/dev/null 2>&1; then
        n=$((n + 1))
      else
        echo "[sync] FAILED to restore $(basename "$f")"
      fi
    done
    echo "[sync] restored $n checkpoint(s) to $SESSION:$REMOTE"
    [ "$n" -eq 0 ] && echo "[sync] nothing in $OUT — training will start fresh"
    # `load()` matches on a config hash, so a checkpoint from a different architecture
    # is ignored rather than loaded as mismatched weights. Restoring extra files is safe.
    echo "[sync] cyp_net_one.py resumes automatically unless --no-resume is passed"
    ;;
  *)
    echo "unknown mode '$MODE' (expected pull or restore)" >&2; exit 2;;
esac
