"""Why is CYP2D6's sigma-hat NEGATIVELY correlated with error? Test the mechanism, don't assert it.

`cyp_sigma_quality.py` found that sigma-hat tracks |residual| positively on three isoforms
(+0.086 / +0.129 / +0.107) and **negatively on CYP2D6** (-0.164 ensemble, -0.157 fitted),
consistent across seeds and both estimators. Compounds the model is more uncertain about have
SMALLER errors there. The hypothesis offered at the time -- a censoring artefact from the floor --
was flagged untested, and this script is the test.

The first half of that hypothesis is **already refuted** by the labels alone: the DRC pIC50
distributions have no censoring spike at all (0.1-0.2% of labels sit on the lowest value, mode
mass 0.2-0.5%, on every isoform). There is nothing compressed against a floor in the scored
labels, and the scored compounds are all `kind == "drc"` anyway.

The floor that DOES exist is in the **training pseudo-labels**: `screen_labels` assigns every
screen-inactive compound the constant **2.211**, and its mass varies 29x across isoforms --

    CYP2D6 2648 inactive vs 53 active   |   CYP3A4 91 inactive vs 2200 active
    CYP1A2 1290 / 771                   |   CYP2C9 654 / 1687

-- which orders almost exactly with the anomaly, CYP2D6 extreme on both. `PSEUDO_W["inactive"]`
is **1.0**, the same weight as a real measurement, so on CYP2D6 a single repeated value is ~61%
of the training signal.

That gives a mechanism with range (`feedback-negatives-need-range`) and, more importantly, a
CAUSAL test rather than another correlation. Three arms differing only in the floor's weight:

* **full**  -- inactive weight 1.0. The status quo; must reproduce the anomaly or nothing else counts.
* **down**  -- inactive weight 0.2.
* **none**  -- inactive rows dropped from training entirely.

If the point mass causes the anomaly, `r(sigma, |err|)` on CYP2D6 must move toward zero/positive
as the weight falls, monotonically, and the three control isoforms must move far less. If it does
not move, the mechanism is **refuted** and the anomaly needs a different explanation.

A mediation decomposition runs alongside on the `full` arm, because the cheapest explanation is
that the anomaly is not about uncertainty at all but about the prediction LEVEL: if sigma-hat is
large where p is small, and |err| is small where p is small, then r(sigma, |err|) is negative with
no floor story required. The partial correlation r(sigma, |err| | p) settles that -- if the
anomaly vanishes when p is controlled for, it was never a statement about uncertainty.

rho(p, y) is reported for every arm too, because deleting 61% of CYP2D6's training signal is
expected to COST accuracy. An intervention that fixes the diagnostic by breaking the model is not
a fix, and this programme prices interventions in what survives (`feedback-score-interventions-
on-what-survives-postprocessing`).

    python scripts/cyp_sigma_anomaly.py --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_borda_decode import N_MEMBERS  # noqa: E402
from cyp_decide_vs_affine import PSEUDO_W, gbm, short  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

#: The three arms: the weight given to screen-"inactive" rows, all pinned at y = 2.211.
#: `None` drops them from the training index entirely rather than weighting them to zero, so the
#: bootstrap draws from a smaller pool instead of resampling rows that contribute nothing.
ARMS = {"full": 1.0, "down": 0.2, "none": None}


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Spearman correlation of a and b with c partialled out, via residualised ranks."""
    ra, rb, rc = (rankdata(v).astype(float) for v in (a, b, c))
    X = np.vstack([np.ones_like(rc), rc]).T
    res = [v - X @ np.linalg.lstsq(X, v, rcond=None)[0] for v in (ra, rb)]
    sa, sb = res[0].std(), res[1].std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(res[0], res[1])[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/sigma_anomaly.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    frame = load(workers=8).frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs = screen_labels(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[anom] features {X.shape}", flush=True)

    # The floor's share of each isoform's training signal -- the range the test needs.
    share = {}
    for c in PRIMARY:
        k = labs[c]["kind"].to_numpy()
        share[c] = float((k == "inactive").sum() / max((k != "none").sum(), 1))
    print("[anom] screen-inactive share of labelled rows: "
          + "  ".join(f"{short(c)} {share[c]*100:.0f}%" for c in PRIMARY), flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[anom] seed {seed}", flush=True)
        for c in PRIMARY:
            lab = labs[c]
            y = frame[c].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
            okv = np.isfinite(y[va])
            yv = y[va][okv]
            if (drc | ps).sum() < 50 or okv.sum() < 60:
                continue
            for arm, wt in ARMS.items():
                t0 = time.time()
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                keep = drc | ps
                if wt is None:
                    keep = keep & ~(ps & (kind == "inactive"))
                else:
                    w = np.where(ps & (kind == "inactive"), wt, w)
                idx = np.flatnonzero(keep)
                if len(idx) < 50:
                    continue
                rng = np.random.default_rng(1000 + seed)
                preds = []
                for m in range(N_MEMBERS):
                    bs = rng.choice(idx, size=len(idx), replace=True)
                    preds.append(gbm(seed * 100 + m).fit(
                        X[bs], yy[bs], sample_weight=w[bs]).predict(X[va])[okv])
                P = np.vstack(preds)
                p, s = P.mean(axis=0), P.std(axis=0, ddof=1)
                err = np.abs(yv - p)
                rec = dict(seed=seed, iso=c, arm=arm, n_train=int(len(idx)), n=int(okv.sum()),
                           r_sig_err=float(spearmanr(s, err).statistic),
                           r_sig_p=float(spearmanr(s, p).statistic),
                           r_err_p=float(spearmanr(err, p).statistic),
                           r_partial=partial_spearman(s, err, p),
                           rho=float(spearmanr(p, yv).statistic),
                           secs=round(time.time() - t0, 1))
                rows.append(rec)
                print(f"   {short(c)} {arm:5}: r(sig,|err|) {rec['r_sig_err']:+.3f}   "
                      f"partial|p {rec['r_partial']:+.3f}   r(sig,p) {rec['r_sig_p']:+.3f}   "
                      f"r(|err|,p) {rec['r_err_p']:+.3f}   rho {rec['rho']:.4f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1

    def mean(iso, arm, key):
        v = [r[key] for r in rows if r["iso"] == iso and r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    print("\n" + "=" * 92)
    print("=== DOSE-RESPONSE: does removing the 2.211 point mass move the anomaly? ===")
    print(f"{'iso':8} {'floor%':>7} | " + " ".join(f"{'r(sig,|err|) ' + a:>18}" for a in ARMS)
          + f" | {'full->none':>11}")
    for c in PRIMARY:
        vals = [mean(c, a, "r_sig_err") for a in ARMS]
        print(f"{short(c):8} {share[c]*100:6.0f}% | "
              + " ".join(f"{v:18.3f}" for v in vals)
              + f" | {vals[-1] - vals[0]:+11.3f}")
    print("\n  The mechanism predicts CYP2D6 moves UP toward zero as the floor's weight falls,")
    print("  and that the other three -- which have far less floor mass -- move much less.")
    print("  If CYP2D6 does not move, the point mass is NOT the cause and the mechanism is dead.")

    print("\n=== MEDIATION on the status-quo arm: is this about sigma, or just about p? ===")
    print(f"{'iso':8} {'r(sig,|err|)':>13} {'partial | p':>12} {'r(sig,p)':>9} {'r(|err|,p)':>11}")
    for c in PRIMARY:
        print(f"{short(c):8} {mean(c,'full','r_sig_err'):13.3f} {mean(c,'full','r_partial'):12.3f} "
              f"{mean(c,'full','r_sig_p'):9.3f} {mean(c,'full','r_err_p'):11.3f}")
    print("\n  If the partial collapses toward zero, the anomaly is a statement about the")
    print("  PREDICTION LEVEL, not about uncertainty, and no floor story is needed at all.")

    print("\n=== THE PRICE: what deleting the floor costs in rank accuracy ===")
    print(f"{'iso':8} " + " ".join(f"{'rho ' + a:>11}" for a in ARMS) + f" {'full->none':>11}")
    for c in PRIMARY:
        v = [mean(c, a, "rho") for a in ARMS]
        print(f"{short(c):8} " + " ".join(f"{x:11.4f}" for x in v) + f" {v[-1]-v[0]:+11.4f}")
    print("\n  An intervention that repairs the diagnostic while destroying rho is not a fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
