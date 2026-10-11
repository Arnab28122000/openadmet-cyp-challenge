"""Labels for the compounds the dose-response stage never measured.

**Every dose-response label in the release is a selected one.** The challenge ships a
single-concentration screen (50 µM, log2 fold-change of signal) for 4,376 compounds on
all four isoforms, and a dose-response curve (DRC) exists for a compound/isoform only
if it was *promoted* out of that screen. The promotion rule differs per isoform —
measured on the release, not assumed (2026-10-08):

| isoform | DRC'd / screened | what decides promotion |
|---|---|---|
| CYP1A2 | 1,412 / 4,376 | rises with inhibition: 100% below log2FC −1.5, 11–24% for weak readouts |
| CYP2C9 | 1,285 / 4,376 | ~one third, **flat** in log2FC — effectively random |
| CYP2D6 | 1,493 / 4,376 | **a hard threshold**: DRC iff log2FC < ~−0.6; the 2,882 others sit at median **+0.38** |
| CYP3A4 | 1,805 / 4,376 | if anything enriched for *weak* readouts |

So a model trained on DRC labels alone has never seen a CYP2D6 non-inhibitor, and it
over-predicts every unselected compound. The test set is exactly that — 750 analogs of
CYP1A2/2C9/3A4 hits, scored on all four isoforms — and the two isoforms whose promotion
depends on activity (1A2, 2D6) are the two this programme scored worst on (0.88 and
1.46 on the board, against 0.52/0.57 for the other two). The programme had recorded
"every label we hold is promoted"; the non-promoted compounds were in the
single-concentration file the whole time.

## What a single 50 µM readout says about pIC50

Calibrated on the compounds that carry both (`calibrate`). With `r = 2**log2fc` the
fraction of signal left, DRC pIC50 is monotone in `r` on every isoform, and a compound
showing no inhibition at 50 µM (`r >= 0.9`) has a DRC pIC50 of **~2.0–2.9 with an
interval of ~[1.05, 3.5]** — the floor window the metric already treats as free. Those
compounds become *interval* labels, not point labels: the loss is zero anywhere inside,
which is the honest statement of what one readout knows.

For readouts in between, the label is the **[10th, 90th] percentile of DRC pIC50 among
compounds with the same readout** (per isoform where the bin holds >= `MIN_BIN`
compounds, pooled over isoforms otherwise) — a predictive interval for the compound's
potency given only its screen value, not the much narrower credible interval of a curve
nobody fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
# Bin edges on r = 2**log2fc (fraction of signal remaining at 50 µM).
R_EDGES = (0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf)
INACTIVE_R = 0.9          # at or above this, the readout is "no meaningful inhibition"
MIN_BIN = 20              # fewer DRC'd compounds than this in a bin -> use the pooled bin
MIN_FLOOR = 25            # below this the floor window is not calibratable; raise, never NaN
#: The floor label is issued in two branches split at this readout. Not a new constant -- 1.0 is
#: already an edge in R_EDGES. Selected out-of-fold over a fitted slope and the pooled constant;
#: see `allr_floor`.
FLOOR_SPLIT_R = 1.0
#: Below this many compounds in a branch, fall back to the pooled median rather than estimate a
#: branch from a noisy half-sample.
MIN_FLOOR_BRANCH = 40
KINDS = ("drc", "inactive", "weak", "active", "none")


def _col(c: str) -> str:
    return f"{c}_pIC50_direct_inhibition"


def calibrate(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-isoform (and pooled) DRC pIC50 statistics by single-readout bin.

    Uses only compounds that have *both* a DRC label and a screen readout, so it is
    fitted on exactly the population the labels came from. Returns one table per
    isoform plus ``"pooled"``, indexed by bin number, with n / y_med / y_q10 / y_q90 /
    lo_med / hi_med.
    """
    rows = []
    for c in CYPS:
        y = frame[_col(c)]
        r = 2.0 ** frame[f"log2fc_{c}"]
        m = y.notna() & r.notna()
        rows.append(pd.DataFrame({
            "iso": c, "bin": np.digitize(r[m], R_EDGES[1:-1]),
            "y": y[m], "lo": frame.loc[m, f"{_col(c)}_conf_low"],
            "hi": frame.loc[m, f"{_col(c)}_conf_high"]}))
    allr = pd.concat(rows, ignore_index=True)

    def stats(d: pd.DataFrame) -> pd.DataFrame:
        g = d.groupby("bin")
        return pd.DataFrame({
            "n": g.size(),
            "y_med": g["y"].median(),
            "y_q10": g["y"].quantile(0.10),
            "y_q90": g["y"].quantile(0.90),
            "lo_med": g["lo"].median(),
            "hi_med": g["hi"].median(),
        })

    out = {c: stats(allr[allr.iso == c]) for c in CYPS}
    out["pooled"] = stats(allr)
    return out


def screen_labels(frame: pd.DataFrame, calib: dict[str, pd.DataFrame] | None = None,
                  ) -> dict[str, pd.DataFrame]:
    """Interval labels for every screened compound that has no DRC on an isoform.

    Returns, per primary column, a frame aligned to ``frame``'s rows with ``kind``
    (one of `KINDS`), ``y`` (a point used only by point losses and diagnostics),
    ``lo`` and ``hi``. Rows that carry a DRC keep ``kind == "drc"`` and NaN screen
    values, so a caller can never mistake a pseudo-label for a measurement.
    """
    calib = calib or calibrate(frame)
    pooled = calib["pooled"]
    # One floor label for every isoform: pooled over the r >= 0.9 bins, because
    # CYP2D6 has no DRC'd compound there at all (that is the selection rule).
    floor = allr_floor(frame)
    out: dict[str, pd.DataFrame] = {}
    for c in CYPS:
        y = frame[_col(c)]
        lf = frame[f"log2fc_{c}"]
        r = 2.0 ** lf
        b = pd.Series(np.digitize(r.fillna(np.nan), R_EDGES[1:-1]), index=frame.index)
        res = pd.DataFrame({"kind": "none", "y": np.nan, "lo": np.nan, "hi": np.nan},
                           index=frame.index)
        res.loc[y.notna(), "kind"] = "drc"
        need = y.isna() & lf.notna()
        tab = calib[c]
        rv = r.to_numpy()
        for i in np.flatnonzero(need.to_numpy()):
            bi = int(b.iloc[i])
            # "No meaningful inhibition at 50 uM" is a property of the READOUT, not of whether
            # some DRC'd compound happened to land in that bin. An earlier version asked the
            # latter, and on an isoform whose promotion rule excludes the weak bins entirely --
            # which is exactly CYP2D6, where no compound above r = 0.6 has a curve -- a plainly
            # inactive compound then fell through to the nearest *populated* bin and was
            # labelled **active**. The pooled table hid it, because the other three isoforms
            # fill those bins. Caught by `test_every_no_inhibition_compound_gets_the_floor_window`.
            if rv[i] >= INACTIVE_R:
                # Two-branch floor, split at the existing R_EDGES boundary r = 1.0. See
                # `allr_floor` for the out-of-fold comparison that selected medians over a
                # fitted slope and over the pooled constant.
                fy = floor["y_lo"] if rv[i] < FLOOR_SPLIT_R else floor["y_hi"]
                res.iloc[i, :] = ["inactive", fy, floor["lo"], floor["hi"]]
                continue
            src = tab if (bi in tab.index and tab.loc[bi, "n"] >= MIN_BIN) else pooled
            if bi not in src.index:
                # No DRC'd compound anywhere landed in this readout bin, so there is nothing
                # to calibrate against. Fall back to the nearest populated bin rather than
                # raising or inventing a number: the tables are monotone in the readout, so
                # the neighbour is the closest honest statement available. (CYP2D6 has no
                # DRC'd compound at all above r = 0.6 — that *is* its promotion rule — so this
                # branch is reached on the real release, not only on a fixture.)
                if len(src.index) == 0:
                    continue
                bi = int(min(src.index, key=lambda b: abs(b - bi)))
            row = src.loc[bi]
            kind = "weak" if R_EDGES[bi] >= 0.7 else "active"
            res.iloc[i, :] = [kind, row["y_med"], row["y_q10"], row["y_q90"]]
        out[_col(c)] = res
    return out


def allr_floor(frame: pd.DataFrame) -> dict[str, float]:
    """The pooled label of a compound showing no inhibition at 50 µM.

    Medians over every DRC'd compound with ``r >= INACTIVE_R`` on any isoform. The
    interval bounds are the medians of the *credible-interval* bounds there: for a
    compound below the lowest tested dose the DRC interval is an absolute window, and
    it is this window the metric charges against.

    SPLIT AT r = 1.0, and why this one and not a fitted model. The 303 compounds here are a
    *measured* sample of what the floor label should be, and the readout carries real information
    about it -- Spearman(r, signed error) is -0.377, p < 1e-4, holding within isoform. A
    log-linear fit of y on log10(r) was tried and rejected: its slope has a Kish effective n of
    **5.7** against a nominal 303 (CYP1A2 is 13.9% of rows and carries 85.7% of the leverage),
    it is biased +0.17 locally over 82% of the rows it is applied to, and it extrapolates one
    compound at r = 23.27 to a pIC50 of 0.576. Measured out-of-fold against the single constant,
    5-fold x 20 shuffles:

        rule                       OOF MSE    OOF MAE
        constant median (old)       +0.00%     +0.00%
        constant mean              -12.87%     +8.51%   <- wins L2, LOSES L1
        split at 1.0, means        -15.67%     +5.16%   <- wins L2, LOSES L1
        split at 1.0, MEDIANS       -4.40%     -1.44%   <- the only rule winning BOTH

    Both losses matter because the submission is 0.30 x XGBoost (squared error on the point) +
    0.70 x a net whose interval hinge is L1-like, so a rule that wins one and loses the other is
    not defensible. 1.0 is already an edge in ``R_EDGES``, so this introduces no new constant.

    ``y`` remains the POOLED median for backward compatibility -- `tests/test_shard_parity.py`
    pins a shard constant against it, and it is the reference point every recorded shift is
    measured from. ``y_lo``/``y_hi`` are the two branches and are what `screen_labels` issues.
    """
    ys, los, his, rs = [], [], [], []
    for c in CYPS:
        r = 2.0 ** frame[f"log2fc_{c}"]
        m = frame[_col(c)].notna() & (r >= INACTIVE_R)
        ys.append(frame.loc[m, _col(c)])
        los.append(frame.loc[m, f"{_col(c)}_conf_low"])
        his.append(frame.loc[m, f"{_col(c)}_conf_high"])
        rs.append(r[m])
    y, lo, hi = pd.concat(ys), pd.concat(los), pd.concat(his)
    r_all = pd.concat(rs)
    keep = np.isfinite(y.to_numpy(float))
    y, r_kept = y[keep], r_all.to_numpy(float)[keep]
    lo, hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
    if len(y) < MIN_FLOOR:
        # Returning NaN here would make every inactive pseudo-label NaN, which the trainers
        # mask out — so the selection fix would silently become a no-op and the run would
        # look fine. Fail loudly instead; the release gives 303 such compounds.
        raise ValueError(
            f"only {len(y)} compounds carry both a dose-response label and a readout showing "
            f"no inhibition at 50 uM (need {MIN_FLOOR}); the floor window cannot be "
            "calibrated, and a NaN floor would disable the screen labels without erroring.")
    lo_branch = r_kept < FLOOR_SPLIT_R
    y_arr = y.to_numpy(float)
    y_lo = float(np.median(y_arr[lo_branch])) if lo_branch.sum() >= MIN_FLOOR_BRANCH else None
    y_hi = float(np.median(y_arr[~lo_branch])) if (~lo_branch).sum() >= MIN_FLOOR_BRANCH else None
    if y_lo is None or y_hi is None:
        # Too few in a branch to estimate it; fall back to the pooled median for both so the
        # behaviour degrades to the previous release rather than to a noisy half-sample.
        y_lo = y_hi = float(y.median())
    return {"y": float(y.median()), "lo": float(lo.median()), "hi": float(hi.median()),
            "n": int(len(y)), "y_lo": y_lo, "y_hi": y_hi,
            "n_lo": int(lo_branch.sum()), "n_hi": int((~lo_branch).sum()),
            "split_at": float(FLOOR_SPLIT_R)}


def summary(labels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts of each label kind per isoform — printed by every caller."""
    return pd.DataFrame({k: v["kind"].value_counts() for k, v in labels.items()}
                        ).reindex(list(KINDS)).fillna(0).astype(int)
