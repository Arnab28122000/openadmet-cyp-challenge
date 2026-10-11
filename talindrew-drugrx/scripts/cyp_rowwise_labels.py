"""Phase 8b: a per-row pseudo-label and per-row weight, both functions of the continuous readout.

ARNAB'S HYPOTHESIS, redirected by its own gate. The ask was a learned per-prediction weight
matrix, a bank of such weightings, and an ensemble over them. `scripts/cyp_weight_predictability.py`
(Phase 8a) tested the two things that decide whether that can work, on the 303 compounds where the
pseudo-label error is actually observable:

    Q1  does the error depend on the continuous readout r = 2^log2fc ?
        PASS. Spearman(r, |err|) -0.187 (p = 0.0011); Spearman(r, SIGNED err) -0.377 (p < 1e-4).
        And it holds WITHIN isoform, so it is not a between-isoform artefact:
        CYP1A2 -0.614 (n=42), CYP2C9 -0.500 (n=11), CYP3A4 -0.234 (n=250).

    Q2  is the error predictable from the 3091 FEATURES, out of fold?
        FAIL. oracle in-fold +0.9949, cross-fitted +0.1274, R^2 -0.048, 1/5 folds at the bar.
        An oracle-to-cross-fitted gap of +0.87 is the signature of the family that closed in
        findings/04-05 -- a convincing in-sample lie.

So the per-row object is licensed as a function of **r** (1-D) and gated out as a function of
**X** (3091-D). And the dominant effect is on the LABEL, not the weight: across terciles of r the
true mean y* runs 2.5901 / 2.3029 / 2.2535, a 0.34 spread -- 77% of the floor population's sd --
while the pipeline assigns all of them the single constant 2.2111.

That constant scores **R^2 = -0.1492** against the floor population: the value in production is
worse than the trivial mean of the very distribution it is meant to summarise.

THE ARMS. Each changes one thing against `base`, except the combined and bank arms.

    base        status quo: label 2.2111, w_inactive 1.0
    meanfloor   label 2.3822 for every inactive row (uniform; the L2-optimal CONSTANT)
    rowlabel    label g(r_i), g log-linear in log10(r), CROSS-FITTED per split
    rowiso      label g(r_i), g isotonic decreasing, CROSS-FITTED per split
    rowboth     rowlabel + per-row weight 1/(sigma_assay^2 + sigma^2(r_i))
    bank        ARNAB'S BANK, done properly: each ensemble member k gets its OWN calibration
                g_k fitted on a bootstrap resample of the in-split dual-labelled compounds, so
                the members differ in their label CALIBRATION as well as their row resample.
                The ensemble then averages over calibration uncertainty -- which is proper
                multiple imputation over the calibration FUNCTION, and unlike the draw-based arm
                of findings/11 it has a downstream consumer: the point estimate itself.

LEAKAGE, handled explicitly. The 303 dual-labelled compounds carry real dose-response labels, so
they enter training as `drc` rows and some of them land in the validation split. g is therefore
fitted **only on dual-labelled compounds inside the training split**, refitted per seed. Fitting g
on all 303 would leak validation labels into the training labels of other rows, and the effect
would look like a gain.

PRE-REGISTERED, written before the run.

1. **Bar, identical to findings/07 and Phase 7 so the numbers are comparable: macro drho >= +0.005
   AND positive on >= 5 of 6 seeds.**
2. **Honest expected value, stated first.** r explains only **5.3%** of the variance of y*
   (log-linear R^2 0.0526; isotonic 0.1773 in-sample, which I expect to shrink hard under
   cross-fitting). The per-row shift has sd **0.112** around a mean of **+0.097**, against the
   uniform meanfloor arm's constant +0.171. So most of the available gain is the UNIFORM mean
   correction, and the r-dependent part is a small addition on top. I expect:
        meanfloor  >  base          (a real correctness fix, +0.002..+0.004)
        rowlabel  ~=  meanfloor     (same location, slightly better shape)
        bank      >=  rowlabel      (averaging over calibration uncertainty)
   and I expect **the macro bar to be MISSED**, as it was by every member of this family so far.
   Said before the run so it cannot be re-framed afterwards.
3. **The ordering IS the test.** If `rowlabel` does not beat `base`, then r carries no usable
   label information despite Q1, and the whole per-row story is dead. If `rowlabel` beats
   `meanfloor`, the r-dependence is worth more than the uniform shift and the per-row object is
   vindicated on its own terms.
4. **Manipulation check** (trap 8): print the realised mean and sd of the applied label shift per
   arm, plus Kish ESS. An arm whose shift sd is ~0 has not varied what its claim is about.
5. **NOT CLAIMED: transfer to CYP2D6.** The dual-labelled sample is 250 CYP3A4 / 42 CYP1A2 /
   11 CYP2C9 / **0 CYP2D6**, and CYP2D6 carries 60.5% of the mass. g is therefore fitted on a
   pooled sample dominated by CYP3A4 and applied to CYP2D6 across a selection boundary. The
   per-isoform table is the thing to read; a macro gain driven entirely by CYP3A4 (1.9% mass) is
   a different result from one that reaches CYP2D6.

    python scripts/cyp_rowwise_labels.py --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_borda_decode import N_MEMBERS  # noqa: E402
from cyp_decide_vs_affine import PSEUDO_W, gbm, short  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import CYPS, INACTIVE_R, _col, screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

FLOOR = 2.21113
SIGMA_ASSAY = 0.3
ARMS = ("base", "meanfloor", "rowlabel", "rowiso", "rowboth", "bank")


def dual_sample(frame):
    """Every (compound, isoform) cell with BOTH a DR curve and a no-inhibition readout.

    Returns (compound_index, readout r, true pIC50). These are the only rows in the release where
    the pseudo-label error is observable, and they are what g is fitted on.
    """
    ci, rr, yy = [], [], []
    for c in CYPS:
        r = 2.0 ** frame[f"log2fc_{c}"].to_numpy(float)
        y = frame[_col(c)].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(r) & (r >= INACTIVE_R)
        ci.append(np.flatnonzero(m)); rr.append(r[m]); yy.append(y[m])
    return np.concatenate(ci), np.concatenate(rr), np.concatenate(yy)


def fit_loglin(r, y):
    """y* = a + b*log10(r). Two parameters, which is all n=303 supports."""
    A = np.vstack([np.ones_like(r), np.log10(r)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda q: beta[0] + beta[1] * np.log10(np.clip(q, INACTIVE_R, None))


def fit_iso(r, y):
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(r, y)
    return lambda q: ir.predict(np.clip(q, INACTIVE_R, None))


def fit_sigma(r, y, g):
    """sigma(r) by binning the absolute residual of g. Three bins; n=303 supports no more."""
    res = np.abs(y - g(r))
    edges = np.quantile(r, [1 / 3, 2 / 3])
    sd = []
    for t in range(3):
        m = np.digitize(r, edges) == t
        sd.append(float(res[m].std(ddof=1)) if m.sum() > 3 else float(res.std(ddof=1)))
    sd = np.maximum(np.array(sd), 1e-3)

    def sig(q):
        return sd[np.digitize(np.asarray(q, float), edges)]
    return sig


def kish(w):
    w = np.asarray(w, float); w = w[w > 0]
    return float(w.sum() ** 2 / (w * w).sum()) if w.size else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/rowwise_labels.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    frame = load(workers=8).frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs = screen_labels(frame)
    d_ci, d_r, d_y = dual_sample(frame)
    print("=" * 96)
    print(f"Phase 8b. Dual-labelled cells n = {d_ci.size}; g is fitted ONLY on those inside each")
    print("training split, refitted per seed, because they carry real DR labels and some land in")
    print("validation. Fitting g on all 303 would leak.")
    g_all = fit_loglin(d_r, d_y)
    print(f"   pooled log-linear: g(0.90) = {float(g_all(np.array([0.90]))[0]):.4f}, "
          f"g(1.5) = {float(g_all(np.array([1.5]))[0]):.4f};  constant in production = {FLOOR}")
    print(f"   constant R^2 against this population: "
          f"{1 - ((d_y - FLOOR)**2).sum()/((d_y - d_y.mean())**2).sum():+.4f}  (0 = the mean)")
    print("=" * 96)

    X = np.asarray(featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                             counts=True, desc_mode="rdkit2d", extra_fps=True), float)
    print(f"[row] features {X.shape}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        in_tr = np.isin(d_ci, tr)
        if in_tr.sum() < 30:
            print(f"[row] seed {seed}: only {int(in_tr.sum())} dual rows in split -- skipping")
            continue
        gr, gy = d_r[in_tr], d_y[in_tr]
        g_lin, g_iso = fit_loglin(gr, gy), fit_iso(gr, gy)
        sig_r = fit_sigma(gr, gy, g_lin)
        print(f"\n[row] seed {seed}  dual rows in split {int(in_tr.sum())}  "
              f"g(0.90)={float(g_lin(np.array([0.90]))[0]):.3f} "
              f"g(1.5)={float(g_lin(np.array([1.5]))[0]):.3f}", flush=True)

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
            iso_short = c.split("_")[0]
            r_row = 2.0 ** frame[f"log2fc_{iso_short}"].to_numpy(float)
            okv = np.isfinite(y[va]); vv = va[okv]; yv = y[va][okv]
            if len(idx) < 50 or okv.sum() < 60 or inact.sum() < 20:
                continue
            rng = np.random.default_rng(1000 + seed)
            boots = [rng.choice(idx, size=len(idx), replace=True) for _ in range(N_MEMBERS)]
            # One bootstrap of the dual sample per member, for the `bank` arm. Drawn once so
            # every arm sees the same member-level randomness where it applies.
            brng = np.random.default_rng(7000 + seed)
            bank_g = []
            for _ in range(N_MEMBERS):
                bs = brng.choice(len(gr), size=len(gr), replace=True)
                bank_g.append(fit_loglin(gr[bs], gy[bs]))

            for arm in ARMS:
                t0 = time.time()
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                shifts = np.zeros(0)
                preds = []
                for mi, bs in enumerate(boots):
                    yy = base_y.copy()
                    if arm == "meanfloor":
                        yy[inact] = float(d_y[in_tr].mean())
                    elif arm in ("rowlabel", "rowboth"):
                        yy[inact] = g_lin(r_row[inact])
                    elif arm == "rowiso":
                        yy[inact] = g_iso(r_row[inact])
                    elif arm == "bank":
                        yy[inact] = bank_g[mi](r_row[inact])
                    ww = w
                    if arm == "rowboth":
                        s = sig_r(r_row[inact])
                        ww = w.copy()
                        ww[inact] = 1.0 / (SIGMA_ASSAY ** 2 + s ** 2)
                        ww[inact] = ww[inact] / ww[inact].mean() * (
                            1.0 / (SIGMA_ASSAY ** 2 + float(np.mean(sig_r(r_row[inact]))) ** 2))
                    if mi == 0:
                        shifts = yy[inact] - FLOOR
                    preds.append(gbm(seed * 100 + mi).fit(
                        X[bs], yy[bs], sample_weight=ww[bs]).predict(X[vv]))
                P = np.vstack(preds)
                p = P.mean(axis=0)
                rec = dict(seed=seed, iso=c, arm=arm, n=int(okv.sum()),
                           rho=float(spearmanr(p, yv).statistic),
                           shift_mean=float(shifts.mean()), shift_sd=float(shifts.std(ddof=1)),
                           ess=kish(w[idx]), n_inact=int(inact.sum()),
                           secs=round(time.time() - t0, 1))
                rows.append(rec)
                print(f"   {short(c)} {arm:10}: rho {rec['rho']:.4f}  shift "
                      f"{rec['shift_mean']:+.4f} +/- {rec['shift_sd']:.4f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1
    seeds = sorted({r["seed"] for r in rows})

    def cell(iso, arm, seed, key="rho"):
        v = [r[key] for r in rows if r["iso"] == iso and r["arm"] == arm and r["seed"] == seed]
        return v[0] if v else np.nan

    print("\n" + "=" * 96)
    print("MANIPULATION CHECK (trap 8): did each arm actually move the label?")
    print(f"   {'arm':11} {'mean shift':>11} {'sd of shift':>12}")
    for arm in ARMS:
        sm = [r["shift_mean"] for r in rows if r["arm"] == arm]
        ss = [r["shift_sd"] for r in rows if r["arm"] == arm]
        print(f"   {arm:11} {np.mean(sm):+11.4f} {np.mean(ss):12.4f}")
    print("   base must be 0.0000 +/- 0.0000; meanfloor a constant shift with sd 0; the row")
    print("   arms a nonzero sd. An arm with sd 0 has not varied what its claim is about.")

    print("\n" + "=" * 96)
    print("THE PRE-REGISTERED BAR: macro drho >= +0.005 AND positive on >= 5 of 6 seeds")
    print("=" * 96)
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
        if not parts:
            continue
        macro = np.array([np.mean([x[i] for x in parts]) for i in range(len(parts[0]))])
        mt = (macro.mean() / (macro.std(ddof=1) / np.sqrt(len(macro)))
              if macro.std(ddof=1) else 0.0)
        verdict = ("CLEARS BAR" if macro.mean() >= 0.005 and (macro > 0).sum() >= 5
                   else "below bar")
        print(f"\n   arm = {arm}")
        print(f"   {'iso':8} {'d rho':>9} {'t':>7} {'wins':>8}   {'mass%':>6}")
        mass = {"CYP2D6": 60.5, "CYP1A2": 29.5, "CYP2C9": 14.9, "CYP3A4": 1.9}
        for iso, (m, t, wins, n) in per.items():
            print(f"   {short(iso):8} {m:+9.4f} {t:7.2f} {wins:4d}/{n:<3d}   "
                  f"{mass.get(short(iso), 0):5.1f}%")
        print(f"   {'MACRO':8} {macro.mean():+9.4f} {mt:7.2f} "
              f"{int((macro>0).sum()):4d}/{len(macro):<3d}  -> {verdict}")
        summary[arm] = dict(macro=float(macro.mean()), t=float(mt),
                            wins=int((macro > 0).sum()), n=len(macro), verdict=verdict,
                            per={short(k): v for k, v in per.items()})

    print("\n" + "=" * 96)
    print("THE ORDERING TEST (pre-registered item 3)")
    print("=" * 96)
    mf = summary.get("meanfloor", {}).get("macro")
    rl = summary.get("rowlabel", {}).get("macro")
    bk = summary.get("bank", {}).get("macro")
    if mf is not None and rl is not None:
        print(f"   meanfloor {mf:+.4f} | rowlabel {rl:+.4f} | bank "
              f"{bk if bk is None else f'{bk:+.4f}'}")
        if rl <= 0:
            print("   rowlabel does NOT beat base: r carries no usable label information")
            print("   despite Q1, and the per-row story is dead. Reported as such.")
        elif rl > mf:
            print("   rowlabel BEATS meanfloor: the r-dependence is worth more than the uniform")
            print("   shift, and the per-row object is vindicated on its own terms.")
        else:
            print("   rowlabel does not beat meanfloor: the gain is the UNIFORM mean correction,")
            print("   and the r-dependent part adds nothing. The honest fix is then a one-line")
            print("   constant change, not a per-row model.")
    print("\n   NOT CLAIMED: transfer to CYP2D6. g is fitted on 250 CYP3A4 / 42 CYP1A2 /")
    print("   11 CYP2C9 / 0 CYP2D6 and applied to CYP2D6 across a selection boundary.")
    print("   Read the per-isoform table: a macro gain driven by CYP3A4 (1.9% mass) is a")
    print("   different result from one that reaches CYP2D6 (60.5%).")
    Path(str(args.out).replace(".json", "_summary.json")).write_text(
        json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
