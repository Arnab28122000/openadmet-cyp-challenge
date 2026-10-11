"""Does more training actually buy score on this problem? Settle it by measuring.

The programme has an answer already, but it is a weak one: every run so far used
`--patience 150`, so every run ended near epoch 170 and **the question "what happens at
epoch 10,000?" was never asked**. The two arms that did train longer (`h12-longbase`,
`h12-longdeep`, best epoch ~210) moved the score by -0.006 and +0.020 -- but they were
still cosine-annealed over their own budget, which confounds "trained longer" with
"decayed more gently", and they stopped three orders of magnitude short of the proposal.

So this sweep removes all three objections at once:

* **No early stopping** (`--patience 0`). A long run is allowed to get worse and then
  recover. That is the only way grokking-shaped behaviour could ever show up, and with
  patience 150 it is unobservable by construction.
* **The schedule is an arm.** `cosine` ties decay to the budget; `sgdr` restarts on a
  lengthening period, so the decay cadence early on differs from the cadence at epoch
  10,000; `plateau` drops the LR only when validation actually stalls. If "decay the LR
  when the error stops falling, on a cadence that itself changes with scale" is what
  makes long training work, `sgdr` and `plateau` are that hypothesis, stated so it can
  fail.
* **Capacity is an arm**, because "more params" is half the proposal.

Everything is scored on **V2** (the validation surface that reproduces the board per
isoform), and the full curve is kept, not just its minimum -- the shape is the finding.

    python scripts/cyp_scale_sweep.py --out data/cyp_cache/scale

Read the result with `--report`, which prints score against budget and the marginal
value of the last 10x of compute.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (tag, epochs, schedule, d_model, n_layers, seeds)
#
# The baseline is the current production config so the comparison is against what we
# actually ship, not against a weakened control. Everything else changes ONE thing from
# it, except `big30k`, which is the "more params AND more epochs" arm stated together
# because that is how the proposal was put.
ARMS = [
    ("base1200",   1_200, "cosine",  1024, 3, (0, 1, 2)),   # what we ship today, 3 seeds
    ("sgdr30k",   30_000, "sgdr",    1024, 3, (0,)),        # restarts on a growing period
    ("plateau30k", 30_000, "plateau", 1024, 3, (0,)),       # decay only when stalled
]

# WHY ONLY THREE ARMS, AND NO SEPARATE 10k/20k BUDGETS. Measured on the A100: 100
# epochs/min, GPU pinned at 100% util and 38.6/40.9 GB, so there is no concurrency
# headroom and a 30,000-epoch run costs 5 h. The original grid was ~35 h.
#
# It is also unnecessary, because `--patience 0` keeps the FULL curve. Under `sgdr` and
# `plateau` the schedule is not tied to the budget, so the point at epoch 1,200 of a
# 30,000-epoch run *is* the 1,200-epoch result, and the curve hands back every
# intermediate budget for free. (That is exactly why cosine cannot serve here: with
# `T_max = epochs` its epoch-1,200 point belongs to a different schedule than a real
# 1,200-epoch cosine, which is the confound that made the earlier `h12-long*` arms
# uninterpretable.) `base1200` stays as the 3-seed control that fixes the noise floor.
#
# Capacity is deliberately absent: `h12-wide` (2048x4) already measured -0.0127 against
# `h12-base` on the same surface, so "more params" has an answer and does not need 5 h.
#
# Seed sd on comparable arms is ~0.010-0.013, so a 1-seed arm against a 3-seed control
# detects a difference of ~0.05 -- far below the ~0.24 the selection fix was worth, and
# well inside what "30k epochs solves it" would have to mean. If an arm lands within
# +-0.02 of base, a second seed is not worth 5 h; if it beats base by >0.05, add seeds.


def run_arm(tag, epochs, schedule, d_model, n_layers, seed, args) -> dict | None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = out / f"{tag}_s{seed}.json"
    if rep.exists() and not args.force:
        print(f"[scale] {tag} s{seed}: already done, skipping", flush=True)
        return json.loads(rep.read_text())
    cmd = [sys.executable, str(ROOT / "scripts" / "cyp_net_one.py"),
           "--tag", tag, "--seed", str(seed),
           "--epochs", str(epochs), "--schedule", schedule,
           "--patience", "0",                       # never stop early: see the whole curve
           "--d-model", str(d_model), "--n-layers", str(n_layers),
           "--members", str(args.members),
           "--screen", "all", "--select-on", "v2",  # the corrected label set and surface
           "--out-dir", str(out),
           "--ckpt-every", "250", "--push-every", "0"]
    if args.feat_cache:
        cmd += ["--feat-cache", args.feat_cache]
    print(f"\n[scale] === {tag} s{seed}: {epochs} epochs, {schedule}, "
          f"{d_model}x{n_layers} ===", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT)
    if p.returncode != 0:
        print(f"[scale] {tag} s{seed} FAILED rc={p.returncode}", flush=True)
        return None
    print(f"[scale] {tag} s{seed} done in {(time.time()-t0)/60:.1f} min", flush=True)
    return json.loads(rep.read_text()) if rep.exists() else None


def report(out: Path) -> int:
    import statistics as st
    from collections import defaultdict
    g = defaultdict(list)
    for p in sorted(out.glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if "best_epoch" in j:
            g[j.get("tag", p.stem)].append(j)
    if not g:
        print(f"no runs under {out}")
        return 1
    print(f"\n{'arm':12} {'n':>2} {'budget':>7} {'sched':>8} {'params':>11} "
          f"{'best_ep':>8} {'V2':>8} {'+-':>7} {'min':>7}")
    print("-" * 78)
    rows = {}
    for tag in sorted(g, key=lambda t: g[t][0].get("epochs_budget", 0)):
        rs = g[tag]
        v = [r.get("v2", r.get("valid")) for r in rs]
        v = [x for x in v if isinstance(x, (int, float))]
        if not v:
            continue
        be = st.mean([r["best_epoch"] for r in rs])
        sd = f"{st.stdev(v):.4f}" if len(v) > 1 else ""
        shape = f"{rs[0].get('d_model')}x{rs[0].get('n_layers')}"
        print(f"{tag:12} {len(rs):>2} {rs[0].get('epochs_budget',0):>7} "
              f"{str(rs[0].get('schedule')):>8} {shape:>11} {be:>8.0f} "
              f"{st.mean(v):>8.4f} {sd:>7} {st.mean([r.get('minutes',0) for r in rs]):>7.1f}")
        rows[tag] = st.mean(v)
    # The number that answers the question: what did the extra compute buy?
    if "base1200" in rows:
        b = rows["base1200"]
        print(f"\nmarginal value of 25x the epochs (lower V2 is better):")
        for tag in ("long30k", "sgdr30k", "plateau30k", "big30k"):
            if tag in rows:
                d = rows[tag] - b
                verdict = "better" if d < 0 else "WORSE"
                print(f"  base1200 {b:.4f}  ->  {tag:11} {rows[tag]:.4f}   "
                      f"{d:+.4f}  ({verdict})")
        print("\nFor scale: the selection-bias fix was -0.2426 and placement is worth")
        print("~0.2-0.33 on the board. Anything here under ~0.02 is noise by comparison.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cyp_cache/scale")
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--feat-cache", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="summarise existing runs and exit")
    ap.add_argument("--only", nargs="*", default=None, help="run only these arm tags")
    args = ap.parse_args()
    out = Path(args.out)
    if args.report:
        return report(out)
    for tag, ep, sch, dm, nl, seeds in ARMS:
        if args.only and tag not in args.only:
            continue
        for s in seeds:
            run_arm(tag, ep, sch, dm, nl, s, args)
    return report(out)


if __name__ == "__main__":
    raise SystemExit(main())
