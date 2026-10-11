"""Audit every Colab session: who is working, who is wedged, who is underfilled.

Three failure modes have each cost this programme real time, and a human
eyeballing `colab sessions` catches none of them:

* **A dead session still billing.** `colab sessions` lists a session whose
  training process died hours ago exactly as it lists a working one. Two A100s
  have died mid-experiment (colabtools#5950) and kept their assignment.
* **A session nobody noticed finished.** Results sit on disk unfetched while the
  card idles at 10.60 units/hr.
* **A card half empty.** Peak 18.8 GB on a 40 GB A100 means the run could have
  carried roughly twice the parameters for the same wall-clock and the same
  money. Headroom is waste, so the reviewer reports a concrete scale factor
  rather than a vague "could be bigger".

Verdicts are deliberately conservative about stopping: a session is only STALE
when it has **no** training process *and* has written nothing for
`--stale-minutes`. A session that merely looks quiet, or that this tool could
not probe at all, is never proposed for shutdown -- an unreachable session is
reported UNKNOWN, because "I could not reach it" and "it is dead" are different
claims and only one of them justifies destroying work.
"""
from __future__ import annotations

import argparse, json, re, subprocess, sys
from pathlib import Path

PROBE = Path(__file__).with_name("gpu_probe.py")
ROW = re.compile(r"^\[(?P<name>[^\]]+)\]\s+(?P<id>\S+)\s*\|\s*Hardware:\s*(?P<hw>[^|]+?)\s*\|")


def sh(args, timeout=240):
    """Never raise on timeout. A probe that times out must become an UNKNOWN
    verdict, not an exception that abandons the audit -- the first version died
    on a busy session and so reported nothing about any session after it, which
    is the opposite of what an auditor is for. A busy kernel is a common and
    harmless reason for a slow probe."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"timeout after {timeout}s")


def sessions():
    r = sh(["colab", "sessions"])
    found = []
    for ln in r.stdout.splitlines():
        m = ROW.match(ln.strip())
        if m:
            found.append({"name": m.group("name"), "hardware": m.group("hw").strip()})
    return found


def probe(name, timeout):
    r = sh(["colab", "exec", "-s", name, "-f", str(PROBE)], timeout=timeout)
    for ln in r.stdout.splitlines():
        if ln.startswith("GPUPROBE "):
            return json.loads(ln[len("GPUPROBE "):])
    return None


def classify(info, stale_s, fill_target):
    """Returns (verdict, note). Only STALE is ever acted on."""
    if info is None:
        return "UNKNOWN", "could not probe -- not proposing shutdown"
    procs = info.get("procs") or []
    age = info.get("artifact_age_s")
    total = info.get("smi_total_gb") or info.get("total_gb") or 0
    used = info.get("smi_used_gb") or 0
    if procs:
        # Busy. The only question left is whether the card is being filled.
        hung = all(p["cpu"] < 1.0 and p["etimes"] > 600 for p in procs)
        if hung:
            return "HUNG?", (f"{len(procs)} process(es) at <1% CPU for >10 min "
                             f"-- check before trusting")
        if total and used / total < fill_target:
            scale = fill_target * total / max(used, 0.1)
            return "UNDERFILLED", (f"{used:.1f}/{total:.1f} GB -- room for roughly "
                                   f"{scale:.1f}x the parameters")
        return "BUSY", f"{len(procs)} process(es), {used:.1f}/{total:.1f} GB"
    if age is None:
        return "EMPTY", "no training process and nothing written -- never used"
    if age > stale_s:
        return "STALE", f"no process, nothing written for {age//60} min"
    return "IDLE", f"no process, last wrote {age//60} min ago -- probably just finished"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-minutes", type=int, default=45)
    ap.add_argument("--fill-target", type=float, default=0.55,
                    help="fraction of GPU memory below which a run is called underfilled")
    ap.add_argument("--stop-stale", action="store_true",
                    help="actually stop sessions judged STALE (and EMPTY with --stop-empty)")
    ap.add_argument("--stop-empty", action="store_true")
    ap.add_argument("--keep", nargs="*", default=[],
                    help="session names never to stop, whatever the verdict")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    found = sessions()
    if not found:
        print("[review] no sessions"); return 0

    report, stop = [], []
    for s in found:
        info = probe(s["name"], args.timeout)
        verdict, note = classify(info, args.stale_minutes * 60, args.fill_target)
        s.update({"verdict": verdict, "note": note, "info": info})
        report.append(s)
        print(f"  {s['name']:10s} {s['hardware']:6s} {verdict:12s} {note}")
        for p in (info or {}).get("procs", [])[:3]:
            print(f"      pid {p['pid']:<7d} {p['etimes']//60:4d} min {p['cpu']:5.1f}% {p['cmd']}")
        if verdict == "STALE" or (verdict == "EMPTY" and args.stop_empty):
            if s["name"] in args.keep:
                print(f"      (kept by --keep)")
            else:
                stop.append(s["name"])

    if stop:
        print(f"\n[review] would stop: {', '.join(stop)}")
        if args.stop_stale:
            for n in stop:
                r = sh(["colab", "stop", "-s", n])
                print(f"[review] stopped {n}: {(r.stdout or r.stderr).strip()[:90]}")
        else:
            print("[review] pass --stop-stale to act")
    else:
        print("\n[review] nothing to stop")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
