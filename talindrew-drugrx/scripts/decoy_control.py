"""Run a screen against its target and against a decoy, and compare the rankings.

**The question: would this screen have ranked the same compounds the same way against
the wrong protein?** If it would, the ranking is a compound-quality prior — molecular
weight, lipophilicity, atom count — wearing the appearance of a prediction about a
protein somebody cares about, and nothing downstream can tell the two apart because both
produce a sorted list with plausible numbers.

Three runs of the same compounds:

1. **real** — the target as given;
2. **shuffle** — the same residues in a random order. Composition and length preserved
   exactly, fold destroyed. *Does structure matter at all?*
3. **unrelated** — hen lysozyme, a different real well-folded protein. *Does **this**
   structure matter?* This is the sharper arm: a scramble alone could only show that
   unfoldable inputs behave oddly.

    python scripts/decoy_control.py --template dpp4-control \\
        --base-url http://localhost:3000 --cookie "$TD_COOKIE"

**This needs real compute.** Every arm is a full fold-dock-rank cycle, so the backend
must have `FOLDRX_OSS_BASE_URL` pointing at a deployed gateway. No local backend does
(`docs/ai-workspace/PROGRESS.md`), so this script is written, unit-tested through
`drugrx/decoy.py`, and **has not been run against live compute** — which is stated here
rather than discovered by a reader.

Reading the output: `spearman` comes with the band it would occupy under independence,
±1.96/√(n−1). At twenty to thirty compounds that band is ±0.4 or wider, so a moderate
correlation is *not* evidence of target specificity and the verdict says so. Making the
control sharper means screening more compounds, not thinking harder about the number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drugrx.decoy import (  # noqa: E402
    compare_rankings,
    ranking_from_candidates,
    shuffled_target,
    unrelated_target,
)


def submit(session, base: str, name: str, sequence: str, template: str,
           seed_compounds: str = "") -> str:
    """Start one run and return its id. The arms differ in the target and nothing else."""
    body: dict = {"name": name, "sequence": sequence, "kind": "protein",
                  "template": template}
    if seed_compounds:
        # Pin the identical compound set across arms. Letting each arm draw its own
        # would compare two different screens, which is not the experiment.
        body["models"] = {"ligands": "custom-smiles"}
        body["params"] = {"ligands": {"smiles": seed_compounds}}

    response = session.post(f"{base}/api/v1/foldrx/runs", json=body, timeout=60)
    if response.status_code == 402:
        raise SystemExit("402: this account's plan does not allow runs.")
    response.raise_for_status()
    return response.json()["run_id"]


def wait(session, base: str, run_id: str, timeout_s: int = 5400) -> dict:
    """Poll until the run finishes, or say plainly why it cannot."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run = session.get(f"{base}/api/v1/foldrx/runs/{run_id}", timeout=60).json()
        status = run.get("status")
        if status in ("complete", "failed", "stopped"):
            if status != "complete":
                raise SystemExit(f"run {run_id} ended {status}: {run.get('error')}")
            return run
        time.sleep(10)
    raise SystemExit(f"run {run_id} did not finish within {timeout_s}s")


def candidates_of(run: dict) -> list[dict]:
    rounds = run.get("rounds") or []
    if not rounds:
        return []
    return rounds[-1].get("candidates") or []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="dpp4-control")
    ap.add_argument("--sequence", default="",
                    help="target sequence; defaults to the template's own demo")
    ap.add_argument("--base-url", default="http://localhost:3000")
    ap.add_argument("--cookie", default="",
                    help="td_at cookie value for an account that may run")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import httpx

    session = httpx.Client(
        cookies={"td_at": args.cookie} if args.cookie else {},
        follow_redirects=True,
    )
    base = args.base_url.rstrip("/")

    catalog = session.get(f"{base}/api/v1/foldrx/catalog", timeout=60).json()
    template = next((t for t in catalog.get("templates", [])
                     if t["key"] == args.template), None)
    if template is None:
        raise SystemExit(f"no template {args.template!r} in the catalog")

    sequence = args.sequence or (template.get("demo") or {}).get("sequence", "")
    if not sequence:
        raise SystemExit(f"{args.template} has no demo sequence; pass --sequence")

    # The identical compound set across all three arms.
    compounds = (template.get("params", {}).get("ligands", {}) or {}).get("smiles", "")

    arms = {
        "real": sequence,
        "shuffle": shuffled_target(sequence, seed=args.seed),
        "unrelated": unrelated_target(sequence),
    }

    rankings: dict[str, list[str]] = {}
    for arm, seq in arms.items():
        print(f"[decoy] submitting {arm} ({len(seq)} aa)", flush=True)
        run_id = submit(session, base, f"decoy-control {arm}", seq,
                        args.template, compounds)
        run = wait(session, base, run_id)
        rankings[arm] = ranking_from_candidates(candidates_of(run))
        print(f"[decoy] {arm}: {len(rankings[arm])} candidates", flush=True)

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "template": args.template,
        "target_length": len(sequence),
        "arms": {arm: rankings[arm] for arm in arms},
        "comparisons": {
            arm: compare_rankings(rankings["real"], rankings[arm],
                                  top_k=args.top_k).to_json()
            for arm in ("shuffle", "unrelated")
        },
    }

    print()
    for arm, result in report["comparisons"].items():
        print(f"{arm:10s} rho {result['spearman']:+.3f} "
              f"(null band ±{result['null_band']:.3f}, n={result['n']})  "
              f"top-{result['top_k']} overlap {result['top_k_overlap']:.0%}")
        print(f"           {result['verdict']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
