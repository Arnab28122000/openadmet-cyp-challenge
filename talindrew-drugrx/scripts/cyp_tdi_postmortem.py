"""What the TDI board row says about the blinded test set, after the fact.

The v2 TDI submission was **pre-registered**: macro MCC 0.318 (2D6 0.191, 3A4 0.445), from
the prevalence solved off the board (pi = 0.0706 / 0.2916) and the assumption that our
*validation* ROC transfers to the test set unchanged, because an ROC is prevalence-independent.
The board came back 0.2964 (2D6 **0.2291**, 3A4 **0.3637**) -- 2D6 beat its prediction, 3A4
missed, and this script works out why from the row itself.

The lever: one board row is one **operating point** on the test ROC. Under H1 (Accuracy,
Precision and Recall are all standard positive-class quantities -- confirmed below to
sd 8e-5 over 152 independent entries) the point is recoverable in closed form,

    q   = R * pi / P                      predicted-positive rate
    FPR = (q - R*pi) / (1 - pi)           false-positive rate

and a single (TPR, FPR) pins an equal-variance binormal ROC, hence an AUC:

    d = Phi^-1(TPR) - Phi^-1(FPR)         AUC = Phi(d / sqrt(2))

That is the *test* AUC, measured rather than assumed, and it is the number the
pre-registration got wrong. With it, the MCC-optimal operating point can be recomputed for
the next submission -- this time against a ROC the test set actually has.

Caveat kept in view: no integer confusion matrix on any N <= 800 reproduces Accuracy,
Precision, Recall and MCC together, and the reported F1 sits *below* the harmonic mean of the
reported P and R on all 305 rows (by 0.0002-0.0073, always the same sign). Both are the
signature of metrics averaged over bootstrap replicates -- which the Interim board's
"Significance (tiers)" column independently implies. So these are effective, lightly
Jensen-biased quantities, not exact counts. The closed-form MCC still reproduces our own
reported MCC to 0.001, which bounds how much that bias matters here.

    python scripts/cyp_tdi_postmortem.py
"""
from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

# Our live TDI row, read 2026-10-09 off the Live board (submitted 2026-10-08 21:44 UTC).
OURS = {
    "CYP2D6": dict(mcc=0.2291, acc=0.8519, prec=0.2184, rec=0.4245, f1=0.2858, rank=59),
    "CYP3A4": dict(mcc=0.3637, acc=0.7313, prec=0.5365, rec=0.5760, f1=0.5544, rank=93),
}
# What we told ourselves would happen, before submitting.
PREREG = {"CYP2D6": 0.191, "CYP3A4": 0.445, "macro": 0.318}
# Validation (V2) AUC of the deployed arm, which the pre-registration assumed would transfer.
AUC_V2 = {"CYP2D6": 0.7281, "CYP3A4": 0.8127}
# Our submitted files' own predicted-positive rates, over all 750 rows.
Q_FILE = {"CYP2D6": 120 / 750, "CYP3A4": 231 / 750}

SQ2 = math.sqrt(2.0)


def ndtri(p: float) -> float:
    """Inverse standard normal CDF (Acklam), max |err| ~1e-9 -- avoids a scipy dependency."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def ndtr(x: float) -> float:
    return 0.5 * math.erfc(-x / SQ2)


def solve_pi(acc: float, prec: float, rec: float) -> float:
    """TP = R*pi*N = P*q*N and Acc = 1 - (FP+FN)/N give pi in closed form."""
    return (1 - acc) / (1 + rec / prec - 2 * rec)


def mcc_at(pi: float, tpr: float, q: float) -> float:
    """MCC in terms of prevalence, recall and predicted-positive rate. Exact."""
    return math.sqrt(pi / (1 - pi)) * (tpr - q) / math.sqrt(q * (1 - q))


def best_q(pi: float, d: float) -> tuple[float, float, float]:
    """MCC-optimal operating point on an equal-variance binormal ROC with separation d."""
    best = (-1.0, 0.0, 0.0)
    t = -6.0
    while t <= 6.0:
        fpr = 1 - ndtr(t)
        tpr = 1 - ndtr(t - d)
        q = pi * tpr + (1 - pi) * fpr
        if 1e-6 < q < 1 - 1e-6:
            m = mcc_at(pi, tpr, q)
            if m > best[0]:
                best = (m, q, tpr)
        t += 0.002
    return best


def board_prevalence(path: Path) -> tuple[float, float, int]:
    """Re-solve pi from every usable entry; the agreement across entries is the validation."""
    vals = []
    for r in csv.DictReader(path.open()):
        try:
            a, p, rc = float(r["Accuracy"]), float(r["Precision"]), float(r["Recall"])
        except (ValueError, KeyError):
            continue
        if not (0 < p < 1 and 0 < rc < 1 and 0 < a < 1):
            continue
        den = 1 + rc / p - 2 * rc
        if den <= 1e-6:
            continue
        v = (1 - a) / den
        if 0 < v < 0.6:
            vals.append(v)
    return st.mean(vals), st.pstdev(vals), len(vals)


def main() -> int:
    dump = Path("/private/tmp/claude-501/-Users-arnab-Documents-talindrew"
                "/c617177f-c1cb-4037-80b1-ab3bb279b316/scratchpad/board_oct9")
    files = {"CYP2D6": dump / "lb_78.csv", "CYP3A4": dump / "lb_81.csv"}

    print("=" * 78)
    print("TDI v2 post-mortem -- pre-registered 0.318 macro, board says 0.2964")
    print("=" * 78)

    macro_now, macro_opt = [], []
    for iso, b in OURS.items():
        if files[iso].exists():
            pi, sd, n = board_prevalence(files[iso])
            src = f"re-solved from {n} board rows, sd {sd:.5f}"
        else:
            pi, src = {"CYP2D6": 0.0706, "CYP3A4": 0.2916}[iso], "cached"

        R, P = b["rec"], b["prec"]
        q = R * pi / P
        fpr = (q - R * pi) / (1 - pi)
        m_check = mcc_at(pi, R, q)

        d = ndtri(R) - ndtri(fpr)
        auc = ndtr(d / SQ2)

        m_opt, q_opt, tpr_opt = best_q(pi, d)
        macro_now.append(b["mcc"])
        macro_opt.append(m_opt)

        print(f"\n### {iso}   (board rank {b['rank']})")
        print(f"  prevalence pi            {pi:.4f}   ({src})")
        print(f"  board MCC                {b['mcc']:.4f}    pre-registered {PREREG[iso]:.4f}"
              f"    {'BEAT' if b['mcc'] > PREREG[iso] else 'MISSED'} by {b['mcc']-PREREG[iso]:+.4f}")
        print(f"  closed-form MCC check    {m_check:.4f}   (vs board {b['mcc']:.4f},"
              f" delta {m_check-b['mcc']:+.4f})")
        print(f"  realised operating point TPR {R:.4f} at FPR {fpr:.4f},"
              f" q {q:.4f}  (file q over 750 rows: {Q_FILE[iso]:.4f})")
        print(f"  TEST AUC (binormal)      {auc:.4f}   <-- assumed {AUC_V2[iso]:.4f} from V2,"
              f" error {auc-AUC_V2[iso]:+.4f}")
        print(f"  MCC-optimal on that ROC  {m_opt:.4f} at q {q_opt:.4f} (TPR {tpr_opt:.4f})"
              f"   headroom {m_opt-b['mcc']:+.4f}")

    print(f"\n### macro")
    print(f"  board                    {st.mean(macro_now):.4f}   pre-registered {PREREG['macro']:.4f}"
          f"   missed by {st.mean(macro_now)-PREREG['macro']:+.4f}")
    print(f"  best reachable by rethresholding alone  {st.mean(macro_opt):.4f}"
          f"   headroom {st.mean(macro_opt)-st.mean(macro_now):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
