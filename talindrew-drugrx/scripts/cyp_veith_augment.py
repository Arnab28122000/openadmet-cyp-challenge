r"""Veith's continuous AC50 as extra ROWS on the primary head. Runs on Colab.

`cyp_veith_continuous.py` rebuilt Veith from AID 1851 with its potency intact: **13,586 compounds,
31,674 measurements**, against the challenge's 1,412/1,285/1,493/2,335. The programme had been
reading TDC's binarised mirror, so the recorded Veith result (+0.0029 +/- 0.0032) was measured on a
0/1 flag and says nothing about the continuous label.

## Scope, set by what was measurable and not by what would be convenient

The offset between Veith and the challenge assay, on compounds carrying both labels:

| endpoint | n | mean | sd | r |
|---|---|---|---|---|
| CYP1A2 | 34 | -0.817 | 0.554 | 0.765 |
| CYP2C9 | **6** | — | — | not estimable |
| CYP2D6 | 48 | -0.406 | 0.419 | **0.789** |
| CYP3A4 | **10** | -0.827 | 0.832 | 0.237 |

**So augmentation is applied to CYP1A2 and CYP2D6 only.** Borrowing an offset for CYP2C9 (6 pairs)
or CYP3A4 (10 pairs, r = 0.237) would be asserting a correction outside the range it was measured
over, which is the error this programme has made three times. The two endpoints that *are*
estimable include CYP2D6 — which is 74% of the leaderboard gap — so the scope costs nothing that
matters.

## How a borrowed row is charged

The metric's loss is an interval hinge: zero inside `[lo, hi]`, Huber outside. Challenge rows have
real credible intervals. Veith rows do not, and inventing a point target for them would assert a
precision we do not have -- we know the challenge-equivalent value only to about the offset's
residual sd.

So a Veith row is given a **pseudo-interval of `y_corrected +/- offset_sd`**, where
`y_corrected = veith_pAC50 + mean_offset`. That is the same loss, charging zero exactly over the
band we genuinely cannot resolve, and it is honest about the borrowing rather than hiding it in a
weight. For CYP2D6 that band is +/-0.419 against a label spread of ~0.92.

## Arms

* `none` -- challenge rows only. The baseline **within this harness**.
* `rows` -- Veith rows added with the measured offset and pseudo-interval. The primary arm.
* `rows-raw` -- Veith rows added with **no** offset correction. The control that shows whether the
  offset is doing the work, rather than the extra rows alone.
* `rows-hq` -- as `rows`, but only Veith measurements whose `Curve_Description` is a *complete*
  curve, discarding "single point of activity" and "poor fit". Tests whether the quality
  annotation the binarised mirror discarded is worth anything.

Scored on the **incumbent's own 922 validation rows** against its own pooled cached predictions,
so the baseline is the real thing. Note the standing caveat: this harness sits ~0.2 above the
incumbent, so arm-vs-arm differences are internally valid while absolute levels are not
comparable to it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIG_ID = "veith-augment-v1"
# Measured, not assumed. Only endpoints with enough paired compounds to estimate an offset.
OFFSET = {"CYP1A2": dict(mean=-0.817, sd=0.554, n=34),
          "CYP2D6": dict(mean=-0.406, sd=0.419, n=48)}
HQ = ("Complete curve; high efficacy", "Complete curve; partial efficacy")
ALL_EPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["none", "rows-raw", "rows-all", "rows-w"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--head-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--borrow-weight", type=float, default=0.3,
                    help="loss weight on BORROWED rows for the `rows-w` arm. CYP3A4 came back "
                         "DECIDED worse (+0.0425, t=+4.16) having been given no rows while 11k "
                         "borrowed rows shaped the shared trunk; down-weighting rebalances the "
                         "gradient an unaugmented endpoint competes against.")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--veith", default="/content/out/veith/veith_continuous.parquet")
    ap.add_argument("--rep-from", default="/content/baseline_pool.npz")
    ap.add_argument("--featcache", default="/content/featcache")
    ap.add_argument("--out", default="/content/out/augment")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    from drugrx.cyp.data import PRIMARY, load
    from drugrx.cyp.metric import METRIC_ID, interval_loss, st_rae
    from drugrx.data.splits import scaffold_split
    from drugrx.features import featurise, standardise_descriptors

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[aug] device {dev} · {METRIC_ID} · {CONFIG_ID}", flush=True)

    fr = load(workers=args.workers).frame
    ve = pd.read_parquet(args.veith)
    ve = ve[~ve["key"].isin(set(fr["key"]))]          # borrowed rows only; no challenge duplicates
    print(f"[aug] challenge {len(fr)} · veith {len(ve)} after removing challenge overlap",
          flush=True)

    # One feature matrix over challenge + borrowed rows, computed once.
    all_smiles = fr["smiles"].astype(str).tolist() + ve["smiles"].astype(str).tolist()
    X = featurise(all_smiles, workers=args.workers, counts=True, desc_mode="rdkit2d",
                  extra_fps=True, cache=args.featcache)
    n_ch = len(fr)
    print(f"[aug] features {X.shape} (challenge rows 0..{n_ch-1})", flush=True)

    Y = np.full((len(all_smiles), 4), np.nan)
    LO = np.full_like(Y, np.nan); HI = np.full_like(Y, np.nan)
    Y[:n_ch] = fr[list(PRIMARY)].to_numpy(float)
    LO[:n_ch] = fr[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    HI[:n_ch] = fr[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    def veith_block(use_offset: bool, hq_only: bool, endpoints: tuple):
        """Borrowed labels as (y, lo, hi) with a pseudo-interval of +/- the offset's residual sd."""
        y = np.full((len(ve), 4), np.nan); lo = y.copy(); hi = y.copy()
        for j, c in enumerate(PRIMARY):
            iso = c.split("_")[0]
            col = f"veithc_{iso}"
            if col not in ve.columns or iso not in endpoints:
                continue
            if use_offset and iso not in OFFSET:
                continue      # an offset can only be APPLIED where it was estimable...
            # ...but the no-offset arm needs no offset, so nothing is excluded from it. The
            # original scope (CYP1A2 + CYP2D6 only) was justified by offset estimability and
            # was therefore wrong for `rows-raw`: it starved CYP2C9 and CYP3A4 of rows while
            # 11k rows pulled the shared trunk toward the other two, and CYP3A4 came back
            # DECIDED worse (+0.0425, t=+4.16, 1/10) having been given nothing.
            v = ve[col].to_numpy(float)
            if hq_only:
                # Raise rather than silently behave like `rows`. The build aggregates by key
                # and currently drops Curve_Description; a missing column must break the arm,
                # not quietly turn it into a duplicate of another one.
                if "Curve_Description" not in ve.columns:
                    raise SystemExit("rows-hq needs Curve_Description in the veith parquet; "
                                     "re-run cyp_veith_continuous.py carrying it through the "
                                     "aggregation")
                v = np.where(ve["Curve_Description"].isin(HQ).to_numpy(), v, np.nan)
            off = OFFSET[iso]["mean"] if use_offset else 0.0
            # Pseudo-interval half-width: the measured residual sd where we have one,
            # else the median of those we do (0.49) rather than a guess at zero.
            w = OFFSET[iso]["sd"] if iso in OFFSET else 0.49
            yc = v + off
            y[:, j] = yc; lo[:, j] = yc - w; hi[:, j] = yc + w
        return y, lo, hi

    pool = np.load(args.rep_from)
    rep_rows = pool["va_index"]
    base_val = pool["val"]
    MASK_CH = np.isfinite(Y[:n_ch]) & np.isfinite(LO[:n_ch]) & np.isfinite(HI[:n_ch])

    def macro_on(p, idx):
        per = {}
        for j, c in enumerate(PRIMARY):
            k = MASK_CH[idx, j]
            if k.sum() > 20:
                per[c] = st_rae(p[k, j], Y[idx][k, j], LO[idx][k, j], HI[idx][k, j])
        return float(np.mean(list(per.values()))), per

    m_base, per_base = macro_on(base_val, rep_rows)
    print(f"[aug] INCUMBENT on these {len(rep_rows)} rows: {m_base:.4f}  "
          + " ".join(f"{c[3:9]} {per_base[c]:.4f}" for c in PRIMARY), flush=True)

    results = {"INCUMBENT-cached": [dict(seed=-1, rep=m_base, per=per_base)]}
    pooled: dict[str, list] = {}

    for seed in range(args.seeds):
        rest = np.setdiff1d(np.arange(n_ch), rep_rows)
        s_tr, s_fit, s_more = scaffold_split([all_smiles[i] for i in rest],
                                             frac_train=0.80, frac_valid=0.15, seed=seed)
        tr_ch = np.concatenate([rest[s_tr], rest[s_more]])
        fit = rest[s_fit]
        for arm in args.arms:
            # Scope is per-arm and explicit. `rows`/`rows-raw` keep the original
            # CYP1A2+CYP2D6 scope so the 10-seed result above is reproduced exactly;
            # `rows-all` borrows every endpoint, which is only defensible without an offset.
            use_off = arm == "rows"
            eps = ALL_EPS if arm == "rows-all" else tuple(OFFSET)
            if arm == "none":
                tr_idx, yA, loA, hiA = tr_ch, Y, LO, HI
            else:
                vy, vlo, vhi = veith_block(use_off, arm == "rows-hq", eps)
                yA, loA, hiA = Y.copy(), LO.copy(), HI.copy()
                yA[n_ch:], loA[n_ch:], hiA[n_ch:] = vy, vlo, vhi
                have = np.isfinite(vy).any(axis=1)
                tr_idx = np.concatenate([tr_ch, n_ch + np.nonzero(have)[0]])
            mA = (np.isfinite(yA) & np.isfinite(loA) & np.isfinite(hiA)).astype(np.float32)
            if arm == "rows-w":
                mA[n_ch:] *= args.borrow_weight

            _, stt = standardise_descriptors(X[tr_idx])
            Z, _ = standardise_descriptors(X, stats=stt)
            Z = torch.as_tensor(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0
                                              ).astype(np.float32), device=dev)
            lot = torch.as_tensor(np.nan_to_num(loA), dtype=torch.float32, device=dev)
            hit = torch.as_tensor(np.nan_to_num(hiA), dtype=torch.float32, device=dev)
            mt = torch.as_tensor(mA, device=dev)

            torch.manual_seed(seed); np.random.seed(seed)
            blocks, w = [], Z.shape[1]
            for _ in range(args.head_layers):
                blocks += [nn.Linear(w, args.head_hidden), nn.ReLU(), nn.Dropout(args.dropout)]
                w = args.head_hidden
            net = nn.Sequential(*blocks, nn.Linear(w, 4)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1),
                                                             eta_min=args.lr * 0.05)
            best_fit, best_rep, best_per, best_ep, since, best_pred = math.inf, None, None, 0, 0, None
            t0 = time.time()
            for ep in range(args.epochs):
                net.train()
                perm = np.random.permutation(len(tr_idx))
                for b in range(0, len(perm), args.batch):
                    ii = torch.as_tensor(tr_idx[perm[b:b + args.batch]], device=dev)
                    loss = interval_loss(net(Z[ii]), lot[ii], hit[ii], mt[ii])
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                sch.step()
                if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
                    net.eval()
                    with torch.no_grad():
                        pf = net(Z[torch.as_tensor(fit, device=dev)]).float().cpu().numpy()
                        pv = net(Z[torch.as_tensor(rep_rows, device=dev)]).float().cpu().numpy()
                    m_f, _ = macro_on(pf, fit)
                    m_r, per_r = macro_on(pv, rep_rows)
                    if m_f < best_fit - 1e-6:
                        best_fit, best_rep, best_per, best_ep, since = m_f, m_r, per_r, ep + 1, 0
                        best_pred = pv.copy()
                    else:
                        since += args.eval_every
                        if since >= args.patience:
                            break
            pooled.setdefault(arm, []).append(best_pred)
            results.setdefault(arm, []).append(
                dict(seed=seed, rep=best_rep, per=best_per, fit=best_fit, stop_epoch=best_ep,
                     n_train=int(len(tr_idx)), secs=time.time() - t0))
            print(f"[aug] seed {seed} {arm:<9s} n_tr {len(tr_idx):5d} REP {best_rep:.4f} "
                  f"@{best_ep}  "
                  + " ".join(f"{c[3:9]} {best_per.get(c, float('nan')):.4f}" for c in PRIMARY)
                  + f"  {time.time()-t0:.0f}s", flush=True)
            (out / "results.json").write_text(json.dumps(
                dict(metric_id=METRIC_ID, config_id=CONFIG_ID, offsets=OFFSET,
                     args=vars(args), results=results), indent=1, default=float))

    print(f"\n{'arm':<14s} {'POOLED':>9s} {'vs none':>9s} " +
          " ".join(f"{c[3:9]:>9s}" for c in PRIMARY))
    print(f"{'INCUMBENT':<14s} {m_base:9.4f} {'':>9s} " +
          " ".join(f"{per_base[c]:9.4f}" for c in PRIMARY))
    psum, base_pool = {}, None
    for arm in args.arms:
        ok = [q for q in pooled.get(arm, []) if q is not None]
        if not ok:
            continue
        mp, pp = macro_on(np.mean(ok, axis=0), rep_rows)
        psum[arm] = dict(macro=mp, per=pp, n=len(ok))
        if arm == "none":
            base_pool = mp
        print(f"{arm:<14s} {mp:9.4f} "
              f"{(mp - base_pool) if base_pool is not None else float('nan'):+9.4f} " +
              " ".join(f"{pp.get(c, float('nan')):9.4f}" for c in PRIMARY))

    print(f"\npaired against `none` — CYP2D6 is the pre-registered endpoint")
    for arm in args.arms:
        if arm == "none" or arm not in results:
            continue
        print(f"  {arm}:")
        for c in list(PRIMARY) + ["MACRO"]:
            key = (lambda r: r["rep"]) if c == "MACRO" else (lambda r: r["per"].get(c, np.nan))
            d = np.array([key(r) for r in results[arm]]) - \
                np.array([key(r) for r in results["none"]])
            d = d[np.isfinite(d)]
            sd = d.std(ddof=1) if len(d) > 1 else float("nan")
            t = d.mean() / (sd / math.sqrt(len(d))) if sd and sd > 0 else float("nan")
            print(f"    {c[:34]:<34s} {d.mean():+.4f} ± {sd:.4f}  t={t:+.2f}  "
                  f"{int((d < 0).sum())}/{len(d)} better  "
                  f"{'DECIDED' if abs(t) > 2.5 else ''}")
    (out / "results.json").write_text(json.dumps(
        dict(metric_id=METRIC_ID, config_id=CONFIG_ID, offsets=OFFSET, args=vars(args),
             results=results, pooled=psum), indent=1, default=float))
    print(f"\n[aug] wrote {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
