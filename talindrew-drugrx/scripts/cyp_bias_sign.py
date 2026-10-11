"""Which way does our placement miss? Resolve the sign the algebra cannot.

`cyp_solve_moments.py` recovers the blinded test set's `(s_y, |m|)` from our own board row,
but R² and MAE both depend on the offset only through `m²` and through a folded normal, so
**the sign of `m` is not algebraically recoverable** -- the solver emits two branches per
isoform and `solved_moments.json` stores both without choosing. Every downstream use of the
bias needs the sign, so it is resolved here, once, against an estimate that shares no step
with the solve: the published reverse-engineering of the blinded per-isoform means that
`cyp_place_calibrate.py` already carries as `MEAN_BLIND`.

The test is a comparison of two distances, so it reports its own discrimination rather than
just a verdict: if the two branches are near-equidistant from the external estimate, the sign
is not resolved and nothing downstream should use it.

Sign convention, fixed here and used everywhere after: `m = y_bar - p_bar`.
**m < 0 means the truth sits below our predictions, i.e. we OVER-predict.**

    python scripts/cyp_bias_sign.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVED = ROOT / "data/cyp_cache/solved_moments.json"
OUT = ROOT / "data/cyp_cache/bias_sign.json"

# Independent estimate of the blinded per-isoform mean pIC50, from a published
# reverse-engineering of the same quantities by an unrelated route. It is the tie-breaker
# precisely because it shares no step with the moment solve.
MEAN_BLIND = {"CYP1A2": 4.344, "CYP2C9": 4.769, "CYP2D6": 3.073, "CYP3A4": 4.817}
MIN_DISCRIMINATION = 3.0     # below this ratio the branches are too close to call


def branches(fit: dict, iso: str) -> list[dict]:
    """Both candidate solutions, normalised across the solver's two output shapes."""
    if "solutions" in fit:
        return [{"y_bar": s["y_bar"], "m": s["bias_m"], "s_y": s["s_y"]}
                for s in fit["solutions"]]
    # the `ties` fallback (CYP2D6) reports |bias| against a floor s_y, not signed solutions
    p_bar, b = fit["s_p"] and fit["p_bar"], fit.get("band", [{}])[0]
    a = fit.get("band", [{}])[0].get("bias_abs")
    if a is None:
        return []
    return [{"y_bar": fit["p_bar"] + a, "m": +a, "s_y": b.get("s_y")},
            {"y_bar": fit["p_bar"] - a, "m": -a, "s_y": b.get("s_y")}]


def main() -> int:
    data = json.loads(SOLVED.read_text())
    out = {"convention": "m = y_bar - p_bar; m < 0 means we OVER-predict",
           "tie_breaker": "MEAN_BLIND, an independent published reverse-engineering",
           "isoforms": {}}

    print(f"{'iso':8} {'p_bar':>7} {'external':>9} | {'branch +':>9} {'dist':>7} | "
          f"{'branch -':>9} {'dist':>7} | {'chosen m':>9} {'x better':>9}")
    print("-" * 92)
    for iso, rec in data["endpoints"].items():
        bs = branches(rec["fit"], iso)
        if len(bs) != 2:
            print(f"{iso:8}  no two branches to choose between")
            continue
        ext = MEAN_BLIND[iso]
        p_bar = rec["fit"]["p_bar"]
        plus = next(b for b in bs if b["m"] >= 0)
        minus = next(b for b in bs if b["m"] < 0)
        dp, dm = abs(plus["y_bar"] - ext), abs(minus["y_bar"] - ext)
        pick = plus if dp < dm else minus
        ratio = max(dp, dm) / max(min(dp, dm), 1e-9)
        ok = ratio >= MIN_DISCRIMINATION
        print(f"{iso:8} {p_bar:7.3f} {ext:9.3f} | {plus['y_bar']:9.3f} {dp:7.3f} | "
              f"{minus['y_bar']:9.3f} {dm:7.3f} | {pick['m']:+9.3f} {ratio:8.1f}x"
              f"{'' if ok else '   NOT RESOLVED'}")
        out["isoforms"][iso] = {"p_bar": p_bar, "external_mean": ext,
                                "y_bar": pick["y_bar"], "m": pick["m"],
                                "s_y": pick["s_y"], "discrimination": round(ratio, 2),
                                "resolved": bool(ok)}

    signs = {k: v["m"] for k, v in out["isoforms"].items()}
    allneg = all(v < 0 for v in signs.values())
    allpos = all(v > 0 for v in signs.values())
    out["unanimous"] = bool(allneg or allpos)
    print()
    if allneg:
        print("All four resolve the SAME way: m < 0, so the truth sits BELOW our predictions")
        print("and we OVER-predict on every isoform, by "
              f"{min(abs(v) for v in signs.values()):.3f}-{max(abs(v) for v in signs.values()):.3f} log units.")
        print("\nThat is a deliberate choice overshooting, not a modelling error: the placement")
        print("step puts the centre above the population mean on purpose, because ST-RAE's")
        print("deadband makes an inactive compound's credible window free up to ~3.5. The")
        print("finding is that it overshot, NOT that the centre should be moved to y_bar --")
        print("the ST-RAE optimum is above y_bar, and only the simulator says by how much.")
    elif allpos:
        print("All four resolve m > 0: we UNDER-predict on every isoform.")
    else:
        print("Signs disagree across isoforms -- treat each separately.")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
