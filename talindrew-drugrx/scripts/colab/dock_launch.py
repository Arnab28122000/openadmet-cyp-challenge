"""Launch a docking job DETACHED on the VM and return immediately.

`colab exec` runs synchronously on the kernel and has a client-side reply timeout. A
foreground job that outlasts it produces `TimeoutError: Timeout waiting for reply`, blocks
every subsequent `exec` behind it, and hides all progress until it ends -- which is how 25
minutes went by with three ligands and no output. Everything long therefore gets launched with
`nohup ... &` writing to its own log, and is monitored by separate short `exec` calls that tail
that log.

Usage from the driver:  colab exec -s cyp-dock -f dock_launch.py
The script reads what to run from /content/dock/CMD, so the same launcher serves every stage.

**CMD must be a bare command, not a `cd X && ...` chain.** The launcher already cd's into D,
and `nohup cd ...` cannot work: `cd` is a shell builtin, so nohup fails to exec it, the `&&`
short-circuits, and the real command never runs. That happened on 2026-10-09 and cost a second
launch -- it is silent, because `$!` is the pid of the backgrounded subshell, which exits
immediately, so the launcher cheerfully reports a pid for a job that never started. The guard
below rejects the chain outright, and the liveness check means a dead launch is never reported
as a live one.
"""
from __future__ import annotations

import os
import subprocess
import time

D = "/content/dock"


def main() -> int:
    cmd = open(f"{D}/CMD").read().strip()
    if cmd.startswith("cd ") or "&&" in cmd.split(">")[0]:
        print(f"REFUSED: CMD must be a bare command, not a shell chain -- got {cmd!r}")
        print(f"         the launcher already cd's into {D}; 'nohup cd' cannot work")
        return 2

    log = f"{D}/job.log"
    # Truncate the log so a poll cannot read a previous run's output as this one's.
    open(log, "w").close()
    full = f"cd {D} && nohup {cmd} >> {log} 2>&1 & echo $!"
    pid = subprocess.run(full, shell=True, capture_output=True, text=True).stdout.strip()
    open(f"{D}/job.pid", "w").write(pid)

    # Never report a launch we have not seen survive. A job that dies on startup leaves a
    # recorded pid and an empty log, which reads exactly like a healthy detached job.
    time.sleep(5)
    alive = subprocess.run(["ps", "-p", pid], capture_output=True).returncode == 0
    if not alive:
        print(f"FAILED: pid {pid} was gone within 5s -- the job did not start")
        print(f"--- {log} ---")
        print(open(log).read()[-2000:] or "(log is empty: the command never ran)")
        return 1

    print(f"launched pid {pid}: {cmd}")
    print(f"log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
