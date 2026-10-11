r"""Could a router keep `both`'s tail gain without its bulk cost? Ceiling first, free.

The tail experiment was about to buy 60 seeds to decide whether `both` beats `none`
in the most-active quintile. Raised in review, and correct: **the leaderboard scores
the macro.** A confirmed tail gain that leaves the macro flat is, by arithmetic, paid
for out of the bulk bins, and shipping auxiliary heads on that basis moves the rank
by nothing. The version that would be actionable is a *routed* model that keeps the
tail gain without the bulk cost — so the experiment that matters is that model
against `none`, scored on the macro.

That experiment needs no GPU. Both arms' validation predictions are already saved
per seed, on the same rows, so every router over them can be evaluated now.

Three things are computed, cheapest and most damning first:

* **The oracle ceiling.** Per molecule, take whichever arm's prediction has the lower
  soft-threshold error. No router can beat this — it uses the labels. If the ceiling
  is close to `none`, no realisable router helps and the whole direction is dead
  without spending anything.
* **A realisable router**: switch to `both` where the *predicted* value exceeds a
  threshold. Thresholds are scanned, and the scan is reported as a scan — the best
  one is selected on the data it is scored on, so it is itself an optimistic bound,
  just a much tighter one than the oracle.
* **A simple average** of the two arms, which needs no threshold at all and is the
  baseline any router must beat to justify its complexity.

Reported on the macro, because that is what is scored, and per bin, because that is
where the mechanism lives.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error, st_rae


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/cyp_cache/cyp_aux")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    runs: dict[str, dict[int, Path]] = {}
    for f in sorted(Path(args.dir).glob("*.json")):
        j = json.loads(f.read_text())
        npz = f.with_suffix(".npz")
        if npz.exists():
            runs.setdefault(j["tag"].replace("aux-", ""), {})[int(j["seed"])] = npz
    seeds = sorted(set(runs["none"]) & set(runs["both"]))
    print(f"paired on {len(seeds)} shared seeds: {seeds}\n")

    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    def macro(pred_of_head: dict[str, np.ndarray], idx: np.ndarray) -> float:
        return float(np.nanmean([
            st_rae(pred_of_head[c], y[c][idx], lo[c][idx], hi[c][idx]) for c in PRIMARY]))

    variants: dict[str, list[float]] = {}
    taus = [4.0, 4.5, 5.0, 5.5, 6.0]
    for s in seeds:
        zn, zb = np.load(runs["none"][s]), np.load(runs["both"][s])
        assert np.array_equal(zn["va_index"], zb["va_index"]), "arms differ in split"
        idx = zn["va_index"]
        pn, pb = zn["val"], zb["val"]

        none_h = {c: pn[:, i] for i, c in enumerate(PRIMARY)}
        both_h = {c: pb[:, i] for i, c in enumerate(PRIMARY)}
        variants.setdefault("none", []).append(macro(none_h, idx))
        variants.setdefault("both", []).append(macro(both_h, idx))
        variants.setdefault("mean(none,both)", []).append(
            macro({c: 0.5 * (none_h[c] + both_h[c]) for c in PRIMARY}, idx))

        # Oracle: per molecule keep whichever arm's error is smaller. Uses labels.
        orac = {}
        for i, c in enumerate(PRIMARY):
            en = soft_threshold_error(pn[:, i], lo[c][idx], hi[c][idx])
            eb = soft_threshold_error(pb[:, i], lo[c][idx], hi[c][idx])
            orac[c] = np.where(eb < en, pb[:, i], pn[:, i])
        variants.setdefault("ORACLE ceiling", []).append(macro(orac, idx))

        # Realisable: route on the predicted value, which is available at inference.
        for t in taus:
            routed = {}
            for i, c in enumerate(PRIMARY):
                use_both = pb[:, i] >= t
                routed[c] = np.where(use_both, pb[:, i], pn[:, i])
            variants.setdefault(f"route@pred>={t}", []).append(macro(routed, idx))

    base = variants["none"]
    print(f"{'variant':20s} {'macro':>8s} {'vs none':>10s} {'se':>8s} {'t':>7s} "
          f"{'wins':>6s}")
    for name, vals in variants.items():
        d = [v - b for v, b in zip(vals, base)]
        m = st.mean(d)
        se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 and any(d) else 0.0
        t = m / se if se else float("nan")
        wins = sum(1 for x in d if x < 0)
        mark = "" if name == "none" else f"{m:+10.4f} {se:8.4f} {t:+7.2f} {wins:4d}/{len(d)}"
        print(f"{name:20s} {st.mean(vals):8.4f} {mark}")

    print("\nLower is better; 'vs none' negative means the variant beats the control.")
    print("The ORACLE row uses the labels to pick per molecule, so no router can beat it:")
    print("if it is not clearly better than none, routing cannot pay for itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
