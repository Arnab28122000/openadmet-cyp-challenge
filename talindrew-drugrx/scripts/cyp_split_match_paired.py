"""Re-read `cyp_split_match_board.py`'s output as a PAIRED comparison of surfaces.

`cyp_split_match_board.py` asks which held-out surface -- `all`, `near` (NN >= 0.50) or
`matched` (importance-weighted to the test's NN histogram) -- reports a rho closest to the
board's. On 3 seeds it declared itself UNDERPOWERED: the surfaces were separated by 0.0336
against an unpaired macro standard error of up to 0.0354.

**That power check uses the wrong standard error.** The three surfaces are scored on nested
subsets of the *same* held-out compounds, from the *same* fitted model, on the *same* split.
A seed that gives a high rho gives a high rho on all three, so the errors are strongly
positively correlated and the standard error of the *difference* is far smaller than that of
either term. The quantity of interest is a paired difference and must be tested as one.

This is the same defect this programme found in the ensemble lever ("the +0.0154 was
unpaired; properly paired it is macro -0.0066"), with the sign reversed: there pairing
destroyed an effect, here it rescues a comparison that was written off as unreadable.

Two estimators are reported because they answer different questions and the original script
used only the first:

    mean-then-abs   |mean_seeds(rho) - board|, macro over isoforms -- the error of the
                    seed-AVERAGED estimate. This is what the original printed.
    abs-then-mean   mean_seeds|rho - board|, macro over isoforms -- the error of a SINGLE
                    seed. This is the one that pairs cleanly over seeds, so it carries
                    the significance test.

### Scope, stated up front because it is easy to overclaim

This measures how well a surface predicts the board's rho **level**. It is NOT by itself a
verdict on comparing two *arms* on that surface: a surface can be badly offset and still rank
arms correctly, because a shared offset cancels in a difference -- a point this programme
already recorded when correcting the validation->board lift. The `signed` section below is
what bears on arm ranking: a surface whose signed error is a near-CONSTANT across isoforms is
usable for selection despite being biased, while one whose error varies across isoforms
distorts any comparison that is pooled or macro-averaged over them.

    python scripts/cyp_split_match_paired.py --in data/cyp_cache/split_match_board.json
    python scripts/cyp_split_match_paired.py --in A.json B.json --confirm-seeds 3 4 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURF_ALL = ("all", "near", "match", "prodmatch")
BOOT = 10000


def short(col: str) -> str:
    return col.split("_")[0]


def load_rows(paths: list[str]) -> list[dict]:
    """Merge result files, de-duplicating on (seed, iso); later files win."""
    by_key: dict[tuple, dict] = {}
    for p in paths:
        for row in json.loads(Path(p).read_text()):
            by_key[(row["seed"], row["iso"])] = row
    return list(by_key.values())


def per_seed_macro_abs(rows, isos, seeds, surf) -> dict[str, np.ndarray]:
    """abs-then-mean: for each seed, the macro mean over isoforms of |rho - board|."""
    out = {}
    for k in surf:
        vals = []
        for s in seeds:
            e = [abs(r[f"rho_{k}"] - r["board"]) for r in rows
                 if r["seed"] == s and r["iso"] in isos and np.isfinite(r[f"rho_{k}"])]
            vals.append(float(np.mean(e)) if e else np.nan)
        out[k] = np.array(vals, float)
    return out


def mean_then_abs(rows, isos, surf) -> dict[str, float]:
    """The original script's estimator: average rho over seeds first, then take the error."""
    out = {}
    for k in surf:
        errs = []
        for c in isos:
            rs = [r for r in rows if r["iso"] == c]
            if not rs:
                continue
            errs.append(abs(float(np.nanmean([r[f"rho_{k}"] for r in rs])) - rs[0]["board"]))
        out[k] = float(np.mean(errs))
    return out


def paired(d: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired summary of a per-seed difference, with a bootstrap over seeds."""
    d = d[np.isfinite(d)]
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    draws = rng.choice(d, size=(BOOT, n), replace=True).mean(axis=1) if n > 1 else np.array([mean])
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    return dict(n=n, mean=mean, sd=sd, se=se,
                t=mean / se if se and np.isfinite(se) and se > 0 else float("nan"),
                lo=lo, hi=hi, wins=int((d > 0).sum()))


def report(tag, rows, isos, seeds, rng, surf) -> None:
    print(f"\n{'='*78}\n{tag}  --  {len(seeds)} seeds, {len(isos)} isoforms\n{'='*78}")

    mta = mean_then_abs(rows, isos, surf)
    print("\nmean-then-abs (the original script's estimator; no significance test attaches)")
    print("   " + "  ".join(f"{k} {mta[k]:.4f}" for k in surf)
          + f"   spread {max(mta.values()) - min(mta.values()):.4f}")

    ps = per_seed_macro_abs(rows, isos, seeds, surf)
    print("\nabs-then-mean, per seed (macro |rho - board|; lower is a better surface)")
    print(f"   {'seed':>5} " + " ".join(f"{k:>10}" for k in surf) + "   best")
    for i, s in enumerate(seeds):
        row = {k: ps[k][i] for k in surf}
        best = min(row, key=lambda k: row[k] if np.isfinite(row[k]) else np.inf)
        print(f"   {s:>5} " + " ".join(f"{row[k]:10.4f}" for k in surf) + f"   {best}")
    print("   " + f"{'mean':>5} " + " ".join(f"{np.nanmean(ps[k]):10.4f}" for k in surf))

    print("\npaired differences against 'all' (positive => that surface is WORSE than 'all')")
    for k in [x for x in surf if x != "all"]:
        st = paired(ps[k] - ps["all"], rng)
        print(f"   {k:>6} - all: {st['mean']:+.4f}  se {st['se']:.4f}  t {st['t']:5.2f}  "
              f"95% CI [{st['lo']:+.4f}, {st['hi']:+.4f}]  worse on {st['wins']}/{st['n']} seeds")


    # Signed error is what bears on arm ranking: a constant offset cancels in a difference,
    # a per-isoform-varying one does not.
    print("\nsigned error (board - validation rho), per isoform, mean +- sd over seeds")
    print(f"   {'iso':9} " + " ".join(f"{k:>16}" for k in surf))
    signed = {k: [] for k in surf}
    for c in isos:
        rs = [r for r in rows if r["iso"] == c and r["seed"] in seeds]
        if not rs:
            continue
        cells = []
        for k in surf:
            v = np.array([r["board"] - r[f"rho_{k}"] for r in rs], float)
            v = v[np.isfinite(v)]
            signed[k].append(float(v.mean()))
            cells.append(f"{v.mean():+.3f} +-{v.std(ddof=1) if len(v) > 1 else float('nan'):.3f}")
        print(f"   {short(c):9} " + " ".join(f"{c2:>16}" for c2 in cells))
    print(f"\n   {'spread across isoforms (sd of the signed offset) -- lower is more constant,':<60}")
    print(f"   {'and a constant offset is harmless when ranking arms':<60}")
    for k in surf:
        v = np.array(signed[k], float)
        print(f"     {k:>6}: offsets {np.array2string(v, precision=3, floatmode='fixed')}  "
              f"mean {v.mean():+.4f}  sd across isoforms {v.std(ddof=1):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+",
                    default=[str(ROOT / "data/cyp_cache/split_match_board.json")])
    ap.add_argument("--confirm-seeds", type=int, nargs="*", default=None,
                    help="report these seeds separately as a confirmation set")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = ap.parse_args()

    rows = load_rows(args.inputs)
    isos = sorted({r["iso"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    rng = np.random.default_rng(args.seed)
    surf = [k for k in SURF_ALL if any(np.isfinite(r.get(f"rho_{k}", np.nan)) for r in rows)]
    print(f"[paired] {len(rows)} rows from {len(args.inputs)} file(s); "
          f"seeds {seeds}; isoforms {[short(c) for c in isos]}; surfaces {surf}")

    if args.confirm_seeds:
        disc = [s for s in seeds if s not in args.confirm_seeds]
        conf = [s for s in seeds if s in args.confirm_seeds]
        if disc:
            report("DISCOVERY seeds (already looked at)", rows, isos, disc, rng, surf)
        if conf:
            report("CONFIRMATION seeds (never looked at before this run)", rows, isos, conf, rng, surf)
        if disc and conf:
            report("ALL seeds pooled", rows, isos, seeds, rng, surf)
    else:
        report("all seeds in the input", rows, isos, seeds, rng, surf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
