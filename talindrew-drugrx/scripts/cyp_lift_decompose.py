"""What actually causes the validation->board lift? Label SPREAD, not easier chemistry.

The lift (board rho minus validation rho) is +0.202 / +0.123 / +0.065 / -0.009 on the four
isoforms, measured over 12 seeds. The standing explanation is the analog expansion: the blinded
set is 96.9% within NN 0.4 of training, so it is "easier" than a scaffold split.

**That explanation lost its support on 2026-10-09.** If the lift came from the similarity
regime, then scoring validation *in* that regime should close it -- and that is exactly what the
`near` / `matched` surfaces do. On 12 fresh seeds they are no closer to the board than the plain
split (`cyp_split_match_paired.py`, every CI crossing zero). So the lift is real and, as of that
measurement, unexplained.

This script tests the remaining candidate, which needs no new chemistry: **Spearman rises with
the spread of the true labels at fixed absolute prediction error.** For a bivariate-normal pair,
`r = 2 sin(pi rho / 6)` and the residual sd after an *optimal* affine map is

    sigma* = s_y * sqrt(1 - r^2)

`sigma*` is the placement-free measure of how well the model predicts: unlike the raw residual
sd, it does not move when we deliberately shrink the submitted spread (our k = 0.66 against a
squared-error optimum of 0.77), so it is the only accuracy figure comparable across the two sets.

`s_y` for the blinded set comes from inverting our own board row (`cyp_solve_moments.py`);
`s_y` for validation is measured directly (`cyp_surface_anatomy.py`). If spread is the whole
story, then `sigma*` transfers and the board's rho is predictable from validation's `sigma*`
plus the blinded `s_y` alone.

**CYP2D6 is excluded**, and that is not a convenience: `cyp_solve_moments.py` REJECTS its
rho-inverted r (0.518) because the reported MAE 1.1827 requires `s_y >= 1.3253`, implying
r = 0.8075 instead. Its board row is internally inconsistent under this model, so its `s_y` is
the least trustworthy of the four and cannot carry a test of the hypothesis.

    python scripts/cyp_lift_decompose.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cyp_cache"
EXCLUDE = ("CYP2D6",)  # inversion rejected by its own MAE check -- see docstring


def r_from_rho(rho: float) -> float:
    """Pearson r implied by Spearman rho under bivariate normality."""
    return float(2.0 * np.sin(np.pi * np.asarray(rho, float) / 6.0))


def rho_from_r(r: float) -> float:
    """Inverse of `r_from_rho`."""
    return float(6.0 / np.pi * np.arcsin(np.clip(r, -2.0, 2.0) / 2.0))


def sigma_star(s_y: float, rho: float) -> float:
    """Residual sd after an optimal affine map: placement-free model accuracy."""
    r = r_from_rho(rho)
    return float(s_y * np.sqrt(max(0.0, 1.0 - r * r)))


def main() -> int:
    moments = json.loads((CACHE / "solved_moments.json").read_text())
    match = json.loads((CACHE / "split_match_board_prod_s3_14.json").read_text())
    anat = json.loads((CACHE / "surface_anatomy_s3_14.json").read_text())

    isos = sorted({r["iso"] for r in match})
    seeds = sorted({r["seed"] for r in match})
    print(f"[lift] validation over {len(seeds)} seeds; blinded moments from the board row "
          f"read {moments['board_read']}\n")

    rows = []
    for c in isos:
        short = c.split("_")[0]
        fit = moments["endpoints"][short]["fit"]
        board = moments["endpoints"][short]["board"]
        # Two shapes: a clean Gaussian solve carries `solutions`; a row whose rho-inverted r the
        # MAE check rejects falls back to `mode: "ties"` with a feasible `band`.
        if "solutions" in fit:
            s_y_board, flag = float(fit["solutions"][0]["s_y"]), ""
        else:
            s_y_board = float(fit["band"][0]["s_y"])
            flag = (f"mode={fit.get('mode')}: rho-implied r {fit.get('rho_implied_r')} rejected, "
                    f"band implies r {fit['band'][0].get('r_implied')}")
        rho_board = float(board["rho"])
        rho_val = float(np.mean([r["rho_all"] for r in match if r["iso"] == c]))
        s_y_val = float(np.mean([r["sd_all"] for r in anat if r["iso"] == c]))
        floor = float(np.mean([r["frac_floor_all"] for r in anat if r["iso"] == c]))
        rows.append(dict(
            iso=short, excluded=short in EXCLUDE, flag=flag, frac_floor=floor,
            rho_val=rho_val, rho_board=rho_board, lift=rho_board - rho_val,
            s_y_val=s_y_val, s_y_board=s_y_board, ratio=s_y_board / s_y_val,
            sig_val=sigma_star(s_y_val, rho_val), sig_board=sigma_star(s_y_board, rho_board),
        ))

    print("=== label spread, and the model's placement-free accuracy on each set ===")
    print(f"{'iso':8} {'rho val':>8} {'rho brd':>8} {'lift':>8} | {'s_y val':>8} {'s_y brd':>8} "
          f"{'ratio':>6} | {'sig* val':>9} {'sig* brd':>9} {'sig* brd/val':>13}")
    for r in rows:
        mark = "  <- excluded" if r["excluded"] else ""
        print(f"{r['iso']:8} {r['rho_val']:8.4f} {r['rho_board']:8.4f} {r['lift']:+8.4f} | "
              f"{r['s_y_val']:8.3f} {r['s_y_board']:8.3f} {r['ratio']:6.2f} | "
              f"{r['sig_val']:9.3f} {r['sig_board']:9.3f} {r['sig_board'] / r['sig_val']:13.2f}"
              f"{mark}")

    keep = [r for r in rows if not r["excluded"]]
    print("\n=== the test: predict board rho from validation accuracy + blinded spread alone ===")
    print("predicted rho assumes sigma* transfers unchanged, so only the label spread differs")
    print(f"{'iso':8} {'rho brd actual':>15} {'rho brd predicted':>18} {'error':>8} "
          f"{'raw val rho error':>18}")
    err_pred, err_raw = [], []
    for r in keep:
        r_pred = float(np.sqrt(max(0.0, 1.0 - (r["sig_val"] / r["s_y_board"]) ** 2)))
        rho_pred = rho_from_r(r_pred)
        e, e0 = rho_pred - r["rho_board"], r["rho_val"] - r["rho_board"]
        err_pred.append(abs(e)); err_raw.append(abs(e0))
        print(f"{r['iso']:8} {r['rho_board']:15.4f} {rho_pred:18.4f} {e:+8.4f} {e0:+18.4f}")
    print(f"\n{'mean |error|':8} {'':15} {np.mean(err_pred):18.4f} {'':8} "
          f"{np.mean(err_raw):18.4f}")
    print(f"  spread correction cuts the validation->board error "
          f"{np.mean(err_raw) / np.mean(err_pred):.1f}x "
          f"({np.mean(err_raw):.3f} -> {np.mean(err_pred):.3f}) on {len(keep)} isoforms")

    print("\n=== what this says about the 'easier test set' story ===")
    worse = [r for r in keep if r["sig_board"] > r["sig_val"]]
    print(f"  sigma* is LARGER on the board (model predicts it WORSE) on {len(worse)}/{len(keep)}"
          f" isoforms, by {np.mean([r['sig_board'] / r['sig_val'] - 1 for r in keep]) * 100:.0f}%"
          " on average.")
    print("  So the blinded set is not easier to PREDICT. Its rho is higher because its labels")
    print(f"  are {np.mean([r['ratio'] for r in keep]):.2f}x more spread out, which buys rank")
    print("  accuracy at the same absolute error.")

    print("\n=== why the NN-matched surface could never have closed the lift ===")
    print("  If the lift is label spread, a surface helps only insofar as it moves s_y toward")
    print("  the blinded set's. The `near` subset barely moves it, and on CYP3A4 moves it the")
    print("  WRONG WAY -- which is the mechanism behind the null in cyp_split_match_paired.py.")
    print(f"  {'iso':8} {'s_y ratio NEEDED':>17} {'delivered by near':>19} {'% of the way':>13}")
    for r in keep:
        sd_near = float(np.mean([a["sd_near"] for a in anat if a["iso"].startswith(r["iso"])]))
        got = sd_near / r["s_y_val"]
        frac = (got - 1.0) / (r["ratio"] - 1.0) * 100.0 if r["ratio"] != 1.0 else float("nan")
        print(f"  {r['iso']:8} {r['ratio']:17.2f} {got:19.2f} {frac:12.0f}%")
    print("  So it reweights WHICH compounds are scored while leaving the label spread alone.")
    print("  It was the wrong variable, not merely an underpowered correction.")

    print("\n=== the standing caveat on this whole calculation: TIES ===")
    print("  `r = 2 sin(pi rho / 6)` assumes a continuous bivariate normal and NO ties. Censored")
    print("  labels at the assay floor are exactly ties, so the map is biased where they are")
    print("  common -- and that is how CYP2D6's board row failed its own MAE check.")
    print(f"  {'iso':8} {'validation fraction at/below pIC50 4':>38}")
    for r in rows:
        print(f"  {r['iso']:8} {r['frac_floor']:38.3f}" + ("  <- excluded" if r["excluded"] else ""))
    print("  CYP3A4 carries 41% ties, so its sigma* is the least reliable of the three kept,")
    print("  and it is also the isoform where the prediction below errs most. Treat the")
    print("  decomposition as directional on CYP3A4 and quantitative on CYP2C9.")
    for r in rows:
        if r["flag"]:
            print(f"  NOTE {r['iso']}: {r['flag']}")

    # The reframing this whole calculation buys: state the gap in pIC50 accuracy, not in rho.
    print("\n=== the gap to the field, in pIC50 accuracy instead of rho ===")
    print(f"  {'iso':8} {'our brd rho':>11} {'field best':>11} {'d rho':>8} | "
          f"{'our sig*':>9} {'needed':>8} {'reduction':>10} {'assay floor':>12}")
    FIELD_BEST = {"CYP1A2": 0.8561, "CYP2C9": 0.8808, "CYP2D6": 0.5700, "CYP3A4": 0.8741}
    #: assay noise as a fraction of LABEL VARIANCE, measured from the challenge's per-label _std
    NOISE_VAR_FRAC = {"CYP1A2": 0.052, "CYP2C9": 0.095, "CYP2D6": 0.067, "CYP3A4": 0.102}
    reductions = []
    for r in keep:
        fb = FIELD_BEST[r["iso"]]
        need = sigma_star(r["s_y_board"], fb)
        floor = r["s_y_board"] * np.sqrt(NOISE_VAR_FRAC[r["iso"]])
        reductions.append(1.0 - need / r["sig_board"])
        print(f"  {r['iso']:8} {r['rho_board']:11.4f} {fb:11.4f} {fb - r['rho_board']:+8.4f} | "
              f"{r['sig_board']:9.3f} {need:8.3f} {(1 - need / r['sig_board']) * 100:9.0f}% "
              f"{floor:12.3f}")
    print(f"\n  Closing the gap needs a {np.mean(reductions) * 100:.0f}% cut in absolute error, "
          f"sigma* {np.mean([r['sig_board'] for r in keep]):.3f} -> "
          f"{np.mean([sigma_star(r['s_y_board'], FIELD_BEST[r['iso']]) for r in keep]):.3f} pIC50.")
    print("  The leaders are well ABOVE the assay floor too, so the gap is real headroom and")
    print("  not a physics violation -- but it is ~0.16 pIC50 of accuracy, not a tweak.")
    # Why the lever table is empty, in one line.
    c = next(r for r in keep if r["iso"] == "CYP3A4")
    dock = sigma_star(c["s_y_board"], c["rho_board"] + 0.0068)
    print(f"\n  For scale: docking's +0.0068 rho on CYP3A4 is sigma* {c['sig_board']:.3f} -> "
          f"{dock:.3f}, a {(1 - dock / c['sig_board']) * 100:.1f}% accuracy gain")
    print(f"  against the {(1 - sigma_star(c['s_y_board'], FIELD_BEST['CYP3A4']) / c['sig_board']) * 100:.0f}%"
          " needed. THAT is why ten levers at +-0.005 rho were never going to close it.")

    out = CACHE / "lift_decompose.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\n[lift] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
