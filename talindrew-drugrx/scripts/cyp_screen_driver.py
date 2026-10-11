"""Run the screen-label arms of the incumbent net back to back, detached.

Launched once with ``nohup`` on the VM; each arm is its own process (the trainer's
docstring explains why process isolation is the only reliable way to free the card).
Refuses to start if a trainer is already running — a surviving child of a restarted
kernel is what OOM'd two runs on 2026-10-06.
"""
from __future__ import annotations

import subprocess
import sys

ARMS = [
    # tag, extra args
    ("scr-none", ["--screen", "none", "--select-on", "v1"]),
    ("scr-all", ["--screen", "all", "--select-on", "v2"]),
]
COMMON = ["--n-layers", "3", "--members", "16", "--epochs", "600", "--patience", "150",
          "--ckpt-every", "0", "--feat-cache", "/content/featcache",
          "--out-dir", "/content/out/cyp_screen"]


def main() -> int:
    n = subprocess.run(["bash", "-lc", "ps -eo command | grep -c '[c]yp_net_one.py'"],
                       capture_output=True, text=True).stdout.strip()
    if n != "0":
        print(f"[driver] {n} trainer process(es) already running; refusing", flush=True)
        return 1
    seeds = [int(s) for s in (sys.argv[1:] or ["0", "1"])]
    for seed in seeds:
        for tag, extra in ARMS:
            cmd = [sys.executable, "/content/scripts/cyp_net_one.py", "--tag", tag,
                   "--seed", str(seed), *COMMON, *extra]
            print(f"[driver] start {tag} s{seed}", flush=True)
            r = subprocess.run(cmd)
            print(f"[driver] end {tag} s{seed} rc={r.returncode}", flush=True)
    print("[driver] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
