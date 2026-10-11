"""Watch a training session from the laptop and stop it when it stops earning.

The thin-harness pattern from the harnesses currently topping ARC-AGI-3 and
Terminal-Bench: few tools, durable state in files, a prediction written before
each action, stuck-detection with escalation. This is the stuck-detector.

    python scripts/monitor.py --session drugrx --run <run-id>
    python scripts/monitor.py --session drugrx --run <run-id> --once

It pulls the run's metrics and heartbeat off the VM, prints the curve, and says
whether the run should be killed. It will only actually stop a session with
`--stop-on-verdict`; by default it reports, because a wrong automatic kill of a
20-hour run is expensive and the kill rules are young.

Three states worth distinguishing, and the heartbeat is what separates them:
**improving**, **stalled** (alive, not learning) and **dead** (no heartbeat).
Killing a stalled run saves money; "killing" a dead one just tidies up; waiting
on either because they look alike is how a budget disappears.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

COLAB = "colab"


def sh(args: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except FileNotFoundError:
        return 127, f"{args[0]} not found — is the Colab CLI installed and on PATH?"


def pull(session: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    code, out = sh([COLAB, "download", "-s", session, remote, str(local)])
    if code != 0:
        print(f"  [pull] {remote}: {out.strip().splitlines()[-1] if out.strip() else code}")
    return code == 0


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A half-written final line is normal when the file was copied
            # mid-flush; it is not a reason to lose the rest.
            continue
    return rows


def summarise(rows: list[dict], monitor: str, mode: str, stall_epochs: int) -> dict:
    epochs = [r for r in rows if r.get("kind") == "epoch"]
    if not epochs:
        return {"state": "no epochs yet", "n_epochs": 0}

    values = [(r["epoch"], r.get(monitor)) for r in epochs if r.get(monitor) is not None]
    if not values:
        return {"state": "no monitored metric", "n_epochs": len(epochs)}

    better = min if mode == "min" else max
    best_epoch, best = better(values, key=lambda kv: kv[1] if mode == "min" else -kv[1])
    last_epoch, last = values[-1]
    since_best = last_epoch - best_epoch

    # The log is append-only *across sessions*, so a "stopped" record from an
    # earlier session sits in the middle of the file while the run is live
    # again. Only a stop that comes after the last epoch means this run is over.
    last_epoch_pos = max(i for i, r in enumerate(rows) if r.get("kind") == "epoch")
    stopped = [r for i, r in enumerate(rows) if r.get("kind") == "stopped" and i > last_epoch_pos]

    if stopped:
        state = f"stopped: {stopped[-1].get('reason')}"
    elif since_best >= stall_epochs:
        state = f"stalled ({since_best} epochs since best)"
    else:
        state = "improving"

    return {
        "state": state,
        "n_epochs": len(epochs),
        "last_epoch": last_epoch,
        "last": last,
        "best": best,
        "best_epoch": best_epoch,
        "since_best": since_best,
        "elapsed_hours": epochs[-1].get("elapsed_hours"),
        "train_loss": epochs[-1].get("train_loss"),
    }


def check_heartbeat(path: Path, stale_after: float) -> str:
    if not path.exists():
        return "no heartbeat file"
    try:
        beat = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "heartbeat unreadable"
    age = time.time() - float(beat.get("t", 0))
    phase = beat.get("phase", "?")
    if age > stale_after:
        return f"DEAD? heartbeat is {age / 60:.1f} min old (phase={phase})"
    return f"alive, {age:.0f}s ago (phase={phase}, epoch={beat.get('epoch')})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--run", required=True, help="run id (the checkpoint directory name)")
    ap.add_argument("--remote-dir", default="/content/runs")
    ap.add_argument("--local-dir", default="runs/monitor")
    ap.add_argument("--monitor", default="mean_rank_proxy")
    ap.add_argument("--mode", default="min", choices=["min", "max"])
    ap.add_argument("--stall-epochs", type=int, default=15)
    ap.add_argument("--stale-minutes", type=float, default=20.0)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--stop-on-verdict", action="store_true",
                    help="actually stop the session when the run is stalled or dead")
    args = ap.parse_args()

    local = Path(args.local_dir) / args.run
    remote = f"{args.remote_dir}/{args.run}"

    while True:
        print(f"\n=== {time.strftime('%H:%M:%S')} {args.session}/{args.run} ===")

        code, status = sh([COLAB, "status", "-s", args.session])
        print(f"session: {status.strip().splitlines()[-1] if status.strip() else code}")

        pull(args.session, f"{remote}/metrics.jsonl", local / "metrics.jsonl")
        pull(args.session, f"{remote}/heartbeat.json", local / "heartbeat.json")

        rows = read_jsonl(local / "metrics.jsonl")
        summary = summarise(rows, args.monitor, args.mode, args.stall_epochs)
        heartbeat = check_heartbeat(local / "heartbeat.json", args.stale_minutes * 60)

        print(f"heartbeat: {heartbeat}")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        dead = heartbeat.startswith("DEAD")
        stalled = str(summary.get("state", "")).startswith("stalled")
        finished = str(summary.get("state", "")).startswith("stopped")

        if args.stop_on_verdict and (dead or stalled):
            why = "dead" if dead else "stalled"
            print(f"[monitor] stopping session {args.session}: {why}")
            print(sh([COLAB, "stop", "-s", args.session])[1])
            return 0

        if finished:
            print("[monitor] run finished on its own; leaving the session alone")
            return 0

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
