"""Minimum detectable effect for a paired comparison on a small test fold.

H7 was pre-registered as "Spearman ≥ +0.08 **and** a bootstrap CI excluding
zero" at n=134. Those two criteria cannot both bind: at that n the interval is
roughly ±0.10, so an observed +0.08 has a CI spanning zero and the CI is the
binding one. As written it is a test that cannot be passed at its own stated
effect size — a design error, not a conservative choice.

The fix is not a larger threshold plucked out of the air. It is to **pair the
bootstrap properly and then measure what the design can actually resolve.**

Two schemes are compared here:

* **unpaired** — resample each model's predictions independently. Between-molecule
  variance enters twice and the interval is wide.
* **paired** — resample *molecules* and compute both models on the same resample,
  so the molecules that are intrinsically hard cancel between the two arms. This
  is what `stats.py` already does, and it is substantially tighter.

The script simulates a known true effect at the real fold sizes and reports the
smallest effect detected at 80% power. **The registered threshold is then set
from that number**, so what gets registered is something the design can resolve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def simulate(n: int, true_gain: float, draws: int = 400, boots: int = 600,
             base_rho: float = 0.35, seed: int = 0) -> dict:
    """Fraction of simulated experiments whose paired CI excludes zero.

    A latent "true value" is drawn, then two predictors with different noise
    levels so that model B's rank correlation exceeds A's by about `true_gain`.
    """
    rng = np.random.default_rng(seed)
    paired_hits = 0
    unpaired_hits = 0
    observed = []

    for _ in range(draws):
        y = rng.normal(size=n)

        def predictor(rho: float) -> np.ndarray:
            # Correlated-with-y predictor at approximately the target rho.
            return rho * y + np.sqrt(max(1e-9, 1 - rho ** 2)) * rng.normal(size=n)

        a = predictor(base_rho)
        b = predictor(min(0.99, base_rho + true_gain))

        rho_a = spearmanr(y, a).statistic
        rho_b = spearmanr(y, b).statistic
        observed.append(rho_b - rho_a)

        # paired: resample molecules, both models scored on the same draw
        diffs = np.empty(boots)
        for i in range(boots):
            idx = rng.integers(0, n, n)
            diffs[i] = spearmanr(y[idx], b[idx]).statistic - spearmanr(y[idx], a[idx]).statistic
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        paired_hits += (lo > 0)

        # unpaired: independent resamples per arm
        da = np.empty(boots // 2)
        db = np.empty(boots // 2)
        for i in range(boots // 2):
            ia, ib = rng.integers(0, n, n), rng.integers(0, n, n)
            da[i] = spearmanr(y[ia], a[ia]).statistic
            db[i] = spearmanr(y[ib], b[ib]).statistic
        lo_u = np.percentile(db, 2.5) - np.percentile(da, 97.5)
        unpaired_hits += (lo_u > 0)

    return {
        "n": n,
        "true_gain": true_gain,
        "mean_observed_gain": float(np.mean(observed)),
        "paired_power": paired_hits / draws,
        "unpaired_power": unpaired_hits / draws,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=134, help="test fold size (half_life)")
    ap.add_argument("--gains", type=float, nargs="*",
                    default=[0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--out", default="/content/out/mde.json")
    args = ap.parse_args()

    print(f"minimum detectable effect at n={args.n}, 95% CI, target 80% power\n")
    print(f"{'true gain':>10s} {'paired power':>13s} {'unpaired power':>15s}")
    rows = []
    for g in args.gains:
        r = simulate(args.n, g, draws=args.draws)
        rows.append(r)
        print(f"{g:10.2f} {r['paired_power']:13.2f} {r['unpaired_power']:15.2f}", flush=True)

    ok = [r for r in rows if r["paired_power"] >= 0.8]
    mde = min(r["true_gain"] for r in ok) if ok else None
    print(f"\npaired MDE at 80% power: {mde}")
    if mde is None:
        print("no tested effect reaches 80% power at this n — the fold is too small "
              "to resolve any plausible gain, and that is the finding")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n": args.n, "rows": rows, "paired_mde": mde}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
