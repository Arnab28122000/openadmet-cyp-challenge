"""How much of our board Spearman is the model, and how much is an easier test set?

Measured 2026-10-09. The answer is the second one, mostly, and it overturned a premise this
programme had been acting on: that the deployed ensemble beat a plain LightGBM by 0.204 of rho
on CYP1A2. Of that, 0.029 is the ensemble and 0.175 is the test set.

The output is a per-isoform **lift**, `board rho - validation rho`, which is the calibration
from our validation surface onto the board. Use it to forecast a modelling change rather than
guessing at it, and re-run it after any submission that changes the model (not placement --
placement is monotone and cannot move rho, so the lift is stable across placement rounds).

The premise under test: a plain LightGBM falls 0.204 and 0.128 of rho short of the board on
CYP1A2 and CYP2C9, which has been read as "the deployed ensemble does something large there".
But the LightGBM's rho is measured on our scaffold validation and the board's on the blinded
test set, and those are different populations -- the test set is measurably easier (median
nearest-neighbour Tanimoto 0.581 against 0.439).

`final-net_s*.npz` holds the production net's own validation predictions and the row indices
they belong to, so the ensemble and the LightGBM can finally be scored on the SAME compounds.

  ensemble val rho ~= board rho  -> the ensemble really is that much better
  ensemble val rho ~= lgbm rho   -> the gap was the easier test set, and there is no mystery
"""
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/arnab/Documents/talindrew/talindrew-drugrx")
from drugrx.cyp.data import PRIMARY, load

BOARD = {"CYP1A2": 0.7597, "CYP2C9": 0.8084, "CYP2D6": 0.5004, "CYP3A4": 0.7685}
LGBM = {"CYP1A2": 0.5558, "CYP2C9": 0.6801, "CYP2D6": 0.4424, "CYP3A4": 0.7764}
BASE = "/Users/arnab/Documents/talindrew/talindrew-drugrx/data/cyp_cache/final_v4screen/nets"


def main():
    frame = load(workers=8).frame.reset_index(drop=True)
    per = {c: [] for c in PRIMARY}
    for s in (0, 1, 2):
        z = np.load(f"{BASE}/final-net_s{s}.npz", allow_pickle=True)
        val, idx = z["val"], z["va_index"]
        for k, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)[idx]
            m = np.isfinite(y)
            if m.sum() > 20:
                per[c].append(float(spearmanr(val[m, k], y[m]).statistic))

    print(f"\nProduction net, Spearman on its OWN saved validation predictions "
          f"({len(per[PRIMARY[0]])} seeds, n={int(m.sum())}+ per isoform):\n")
    print(f"{'iso':8} {'prod net val':>13} {'lgbm val':>9} {'board':>8} | "
          f"{'net - lgbm':>11} {'board - net':>12}")
    print("-" * 70)
    for c in PRIMARY:
        iso = c[:6]
        v = float(np.mean(per[c]))
        sd = float(np.std(per[c], ddof=1)) if len(per[c]) > 1 else float("nan")
        print(f"{iso:8} {v:8.4f}+-{sd:.3f} {LGBM[iso]:9.4f} {BOARD[iso]:8.4f} | "
              f"{v-LGBM[iso]:+11.4f} {BOARD[iso]-v:+12.4f}")
    mv = float(np.mean([np.mean(per[c]) for c in PRIMARY]))
    ml = float(np.mean(list(LGBM.values())))
    mb = float(np.mean(list(BOARD.values())))
    print(f"{'MACRO':8} {mv:13.4f} {ml:9.4f} {mb:8.4f} | {mv-ml:+11.4f} {mb-mv:+12.4f}")
    print("\n'net - lgbm' is the ensemble's REAL contribution, both on the same compounds.")
    print("'board - net' is what the easier blinded test set adds on top, for free.")


if __name__ == "__main__":
    main()
