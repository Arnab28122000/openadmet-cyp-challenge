"""Phase 7: inverse-variance weighting, because the label is a point estimate with known noise.

`findings/11` refuted the censoring frame: 2.211 is the **median** of a measured 303-compound
distribution (mean 2.3822, sd 0.4435, min 1.906), and **P(y* > 2.211) = 0.4983**. So the screen
label is not a bound, it is a *location with a measured dispersion*, and the pipeline asserts that
dispersion is zero. Under squared loss the efficient treatment of a point label with known noise
variance is not an estimator -- it is a weight, `w_i prop to 1/sigma_i^2`.

Everything needed is already measured and in the release:

    floor pseudo-label      sigma = 0.4435                             (n = 303, measured)
    DR labels               sigma from (conf_high - conf_low)/3.92     median 0.0869, rms 0.2632
    assay reproducibility   sigma_assay ~ 0.3 ASSUMED (see below)

    w_floor / w_drc = (sigma_d / sigma_floor)^2  =  0.038  (median sigma_d)
                                        =  0.352  (rms sigma_d)
                                        =  0.496  (rms, with sigma_assay in quadrature)

`PSEUDO_W["inactive"] = 1.0` is therefore **2x to 26x too high**, and the bracket is DERIVED
rather than swept -- which matters, because re-reading an old sweep would trip
`feedback-slices-share-noise-resamples-dont`. The `down` arm of `cyp_censored_floor.py` already
sat at w = 0.2, *below* this bracket, and gained rho on all four isoforms at t = 1.92, 10/16.

Two further defects the same derivation exposes, both free:

* **`PSEUDO_W` is ordered inversely to the measured precisions.** `inactive` gets 1.0 and
  `weak`/`active` get 0.5, yet `weak`/`active` are q10-q90 *windows* from the calibration table --
  demonstrably wider than the floor label's 0.4435. The more precise label is down-weighted.
* **`allr_floor` returns the MEDIAN while the training loss is SQUARED ERROR**, whose
  risk-minimising constant is the **MEAN**, 2.3822. The median is L1-optimal. A **+0.171**
  location correction that is wrong for a stated reason and costs nothing.

ARMS. Each changes exactly one thing against `base`, plus one combined arm.

    base      status quo: floor 2.2111, w_inactive 1.0, w_weak/active 0.5, w_drc 1.0
    meanfloor floor -> 2.3822 (the L2-optimal constant). Location only.
    gls       w_inactive -> 0.496 (the loose, most generous end of the derived bracket)
    gls_tight w_inactive -> 0.038 (the strict end, median sigma_d, no sigma_assay)
    rowwise   gls PLUS per-row DR weights 1/(sigma_assay^2 + sigma_CI,i^2), normalised to mean 1
    both      meanfloor + gls

PRE-REGISTERED, written before the run.

1. **The bar, identical to `findings/07`'s so the numbers are comparable: macro drho >= +0.005 AND
   positive on >= 5 of 6 seeds.**
2. **Honest expected value, stated first: I expect this to MISS.** The `down` arm at w = 0.2 gave
   macro +0.0037 at 5/6 -- below bar -- and `gls` at 0.496 is a *less* aggressive
   down-weighting than 0.2, so the naive expectation is a *smaller* gain, somewhere in
   +0.001..+0.004. `gls_tight` at 0.038 is more aggressive than 0.2 and is the arm with a real
   chance. If the response is monotone in w then the ordering must be
   `gls (0.496) < down (0.2) < gls_tight (0.038)`, and that ordering is itself the test of whether
   the GLS derivation is picking up anything the sweep did not.
3. **The manipulation check, which `findings/07` lacked (`AGENTS.md` trap 8).** The quantity being
   varied is the *effective sample size* of the pseudo-labels, so print **Kish ESS** `(sumw)^2/sumw^2`
   for every arm (`feedback-weighting-that-concentrates-is-restriction`). An arm whose ESS does
   not move has not varied what its claim is about.
4. **`meanfloor` is a CORRECTNESS fix, not a lever, and is graded separately.** Its prediction is
   |drho| < 0.003 with sign unknown: moving the constant 0.171 toward the DR mean (4.78) could help
   (less distortion of the low end) or hurt (less separation). It is submitted only if it is
   non-negative, because it is defensible on the loss argument regardless of size -- but a
   *negative* result on it would be interesting and must be reported as such.
5. **NOT CLAIMED: sigma_assay = 0.3.** It is the one quantity here that is assumed rather than
   measured; a DR curve's credible interval is its *fit precision*, not its accuracy. The
   `gls`/`gls_tight` pair brackets that assumption, which is why both are run.

Priced at **0.35% of absolute accuracy against a 13.6% requirement** (`PLAN.md`), i.e. this is a
correctness matter with a derivation, not a board play. Said before the run.

    python scripts/cyp_gls_weights.py --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_borda_decode import N_MEMBERS  # noqa: E402
from cyp_decide_vs_affine import PSEUDO_W, gbm, short  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import CYPS, INACTIVE_R, _col, screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

#: Measured, not assumed. Re-derived at runtime and asserted against these.
FLOOR_MEDIAN = 2.21113
SIGMA_ASSAY = 0.3          # ASSUMED. The one soft number here; bracketed by gls / gls_tight.
ARMS = ("base", "meanfloor", "gls", "gls_tight", "rowwise", "both")


def latent_floor_stats(frame) -> dict[str, float]:
    """The distribution `allr_floor` reduces to a median. Measured, so the arms can assert it."""
    ys = []
    for c in CYPS:
        m = frame[_col(c)].notna() & (2.0 ** frame[f"log2fc_{c}"] >= INACTIVE_R)
        ys.append(frame.loc[m, _col(c)])
    y = np.concatenate([v.to_numpy(float) for v in ys])
    y = y[np.isfinite(y)]
    return dict(n=int(y.size), mean=float(y.mean()), median=float(np.median(y)),
                sd=float(y.std(ddof=1)), lo=float(y.min()),
                p_above=float((y > FLOOR_MEDIAN).mean()))


def drc_sigma(frame, col: str) -> np.ndarray:
    """Per-row DR label sd from the release's own credible interval. NaN where absent.

    `col` is the FULL column name (as in `PRIMARY`), not the bare isoform. Passing the bare name
    and letting this re-suffix it was the first bug here -- it produced
    `CYP1A2_pIC50_direct_inhibition_pIC50_direct_inhibition_conf_low`.
    """
    lo = frame[f"{col}_conf_low"].to_numpy(float)
    hi = frame[f"{col}_conf_high"].to_numpy(float)
    return (hi - lo) / 3.92


def kish(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    w = w[w > 0]
    return float(w.sum() ** 2 / (w * w).sum()) if w.size else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/gls_weights.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    frame = load(workers=8).frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs = screen_labels(frame)
    stats = latent_floor_stats(frame)
    print("=" * 94)
    print("THE MEASURED LATENT FLOOR DISTRIBUTION (findings/11), re-derived here:")
    print(f"   n {stats['n']}  mean {stats['mean']:.4f}  median {stats['median']:.4f}  "
          f"sd {stats['sd']:.4f}  min {stats['lo']:.3f}  P(y*>c) {stats['p_above']:.4f}")
    assert abs(stats["median"] - FLOOR_MEDIAN) < 1e-3, "floor median moved; re-read it"
    sig_floor = stats["sd"]
    mean_floor = stats["mean"]

    parts = []
    for col in PRIMARY:
        s = drc_sigma(frame, col)
        parts.append(s[np.isfinite(s) & frame[col].notna().to_numpy()])
    sd_all = np.concatenate(parts)
    sd_med, sd_rms = float(np.median(sd_all)), float(np.sqrt((sd_all ** 2).mean()))
    # The scalar GLS ratio is w_floor divided by the MEAN DR WEIGHT, i.e.
    #     ratio = (1/sig_floor^2) / mean(1/sig_i^2),
    # which is dominated by the most PRECISE DR rows. Using a ratio of sd summaries instead is
    # wrong in a way that matters: rms(sigma) is inflated by a tail of wide credible intervals
    # and gave 0.81, against 0.21 for the correct form -- a 4x difference in the thing being
    # pre-registered. Caught by the two forms disagreeing.
    w_bar_tight = float(np.mean(1.0 / sd_all ** 2))
    w_bar_loose = float(np.mean(1.0 / (SIGMA_ASSAY ** 2 + sd_all ** 2)))
    w_tight = (1.0 / sig_floor ** 2) / w_bar_tight
    w_loose = (1.0 / sig_floor ** 2) / w_bar_loose
    print(f"   DR label sd: median {sd_med:.4f}  rms {sd_rms:.4f}  (n {sd_all.size})")
    print(f"   DERIVED w_floor/w_drc = (1/sig_floor^2) / mean(1/sig_i^2):")
    print(f"      tight (CI only)                   {w_tight:.4f}")
    print(f"      loose (sigma_assay {SIGMA_ASSAY} in quadrature)  {w_loose:.4f}")
    print(f"   PSEUDO_W['inactive'] is {PSEUDO_W['inactive']} -> too high by "
          f"{PSEUDO_W['inactive']/w_loose:.1f}x to {PSEUDO_W['inactive']/w_tight:.1f}x")
    print("=" * 94)

    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[gls] features {X.shape}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[gls] seed {seed}", flush=True)
        for c in PRIMARY:
            lab = labs[c]
            y = frame[c].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            base_y = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
            idx = np.flatnonzero(drc | ps)
            inact = (kind == "inactive") & ps
            okv = np.isfinite(y[va]); vv = va[okv]; yv = y[va][okv]
            if len(idx) < 50 or okv.sum() < 60:
                continue
            sd_row = drc_sigma(frame, c)
            # Pre-generate the bootstrap index sets ONCE per (seed, member) and reuse them across
            # every arm, so each contrast is paired on the resample as well as on the seed.
            rng = np.random.default_rng(1000 + seed)
            boots = [rng.choice(idx, size=len(idx), replace=True) for _ in range(N_MEMBERS)]

            for arm in ARMS:
                t0 = time.time()
                yy = base_y.copy()
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                if arm in ("meanfloor", "both"):
                    yy = np.where(inact, mean_floor, yy)
                if arm in ("gls", "both"):
                    w = np.where(inact, w_loose, w)
                elif arm == "gls_tight":
                    w = np.where(inact, w_tight, w)
                elif arm == "rowwise":
                    w = np.where(inact, w_loose, w)
                    pr = 1.0 / (SIGMA_ASSAY ** 2 + np.nan_to_num(sd_row, nan=sd_med) ** 2)
                    pr = pr / pr[drc].mean()        # normalise so DR rows average weight 1
                    w = np.where(drc, pr, w)
                ess = kish(w[idx])
                preds = []
                for bs in boots:
                    preds.append(gbm(seed * 100 + len(preds)).fit(
                        X[bs], yy[bs], sample_weight=w[bs]).predict(X[vv]))
                P = np.vstack(preds)
                p = P.mean(axis=0)
                rec = dict(seed=seed, iso=c, arm=arm, n=int(okv.sum()),
                           rho=float(spearmanr(p, yv).statistic),
                           ess=ess, ess_frac=ess / len(idx),
                           w_inact=float(w[inact].mean()) if inact.any() else 0.0,
                           floor_y=float(yy[inact].mean()) if inact.any() else 0.0,
                           secs=round(time.time() - t0, 1))
                rows.append(rec)
                print(f"   {short(c)} {arm:10}: rho {rec['rho']:.4f}  ESS {ess:7.1f} "
                      f"({rec['ess_frac']:.3f})  w_inact {rec['w_inact']:.3f}  "
                      f"floor {rec['floor_y']:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1
    seeds = sorted({r["seed"] for r in rows})

    def cell(iso, arm, seed, key="rho"):
        v = [r[key] for r in rows if r["iso"] == iso and r["arm"] == arm and r["seed"] == seed]
        return v[0] if v else np.nan

    print("\n" + "=" * 94)
    print("3. MANIPULATION CHECK (trap 8): did each arm move the effective sample size?")
    print(f"   {'arm':11} {'w_inact':>8} {'mean Kish ESS':>14} {'ESS/n':>8} {'floor y':>9}")
    for arm in ARMS:
        e = [r["ess"] for r in rows if r["arm"] == arm]
        f = [r["ess_frac"] for r in rows if r["arm"] == arm]
        wi = [r["w_inact"] for r in rows if r["arm"] == arm]
        fy = [r["floor_y"] for r in rows if r["arm"] == arm]
        print(f"   {arm:11} {np.mean(wi):8.3f} {np.mean(e):14.1f} {np.mean(f):8.3f} "
              f"{np.mean(fy):9.3f}")

    print("\n" + "=" * 94)
    print("4. THE PRE-REGISTERED BAR: macro drho >= +0.005 AND positive on >= 5 of 6 seeds")
    print("=" * 94)
    summary = {}
    for arm in ARMS[1:]:
        per, parts = {}, []
        for iso in PRIMARY:
            d = np.array([cell(iso, arm, s) - cell(iso, "base", s) for s in seeds])
            d = d[np.isfinite(d)]
            if not d.size:
                continue
            t = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                 if len(d) > 1 and d.std(ddof=1) else 0.0)
            per[iso] = (float(d.mean()), float(t), int((d > 0).sum()), len(d))
            parts.append(d)
        macro = np.array([np.mean([x[i] for x in parts]) for i in range(len(parts[0]))])
        mt = (macro.mean() / (macro.std(ddof=1) / np.sqrt(len(macro)))
              if macro.std(ddof=1) else 0.0)
        verdict = ("CLEARS BAR" if macro.mean() >= 0.005 and (macro > 0).sum() >= 5
                   else "below bar")
        print(f"\n   arm = {arm}")
        print(f"   {'iso':8} {'d rho':>9} {'t':>7} {'wins':>8}")
        for iso, (m, t, wins, n) in per.items():
            print(f"   {short(iso):8} {m:+9.4f} {t:7.2f} {wins:4d}/{n:<3d}")
        print(f"   {'MACRO':8} {macro.mean():+9.4f} {mt:7.2f} "
              f"{int((macro>0).sum()):4d}/{len(macro):<3d}  -> {verdict}")
        summary[arm] = dict(macro=float(macro.mean()), t=float(mt),
                            wins=int((macro > 0).sum()), n=len(macro),
                            per={k: v for k, v in per.items()}, verdict=verdict)

    print("\n" + "=" * 94)
    print("5. THE MONOTONICITY TEST (pre-registered item 2). If the response is monotone in w,")
    print("   gls(0.496) < down(0.2, +0.0037 from findings/07) < gls_tight(0.038).")
    g, gt = summary.get("gls", {}).get("macro"), summary.get("gls_tight", {}).get("macro")
    if g is not None and gt is not None:
        ok = g < 0.0037 < gt
        print(f"   gls {g:+.4f} | down +0.0037 (prior run) | gls_tight {gt:+.4f}"
              f"   -> {'MONOTONE as predicted' if ok else 'NOT monotone'}")
        print("   A non-monotone result means the response is not a simple function of w, and")
        print("   the GLS derivation is picking up something the old sweep did not -- or noise.")
    print("\n   meanfloor is graded SEPARATELY as a correctness fix, prediction |drho| < 0.003.")
    print("   A negative result on it is interesting and is reported as such.")
    print("\n   Priced at 0.35% of absolute accuracy against a 13.6% requirement. A near miss is")
    print("   a miss (feedback-a-magnitude-bar-is-not-a-significance-bar).")
    Path(str(args.out).replace(".json", "_summary.json")).write_text(
        json.dumps(dict(summary=summary, floor=stats, w_tight=w_tight, w_loose=w_loose,
                        sd_med=sd_med, sd_rms=sd_rms), indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
