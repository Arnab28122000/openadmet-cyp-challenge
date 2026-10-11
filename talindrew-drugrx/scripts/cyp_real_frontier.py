"""What is the ACTUAL best macro sigma* anyone on this board achieves, and how far are we?

This exists because the programme has been quoting a 22% target derived from a leader sigma* of
0.567, and two research strands have now priced levers against it. Before another hour is spent
against that number it should be re-derived from the board rather than inherited.

Two traps it is built to avoid:

* **Per-isoform bests do not compose.** The best rho on each isoform is achieved by DIFFERENT
  entrants, so stacking them describes a competitor who does not exist. The honest ceiling is the
  best macro actually achieved by a single row.
* **A macro of rho is not a macro of accuracy.** sigma* = s_y*sqrt(1-r^2) is per isoform, with a
  DIFFERENT s_y each. Averaging rho across isoforms and converting once is not the same as
  converting per isoform and averaging, because the map is non-linear and the s_y differ by 1.7x.
  The second is the correct one.

    python scripts/cyp_real_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyp_board_fetch import boards, fetch  # noqa: E402

CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
US = "talindrew"
#: Blinded-truth sd per isoform, from the moment inversion. s_y varies 1.7x across isoforms,
#: which is exactly why sigma* must be computed per isoform and then averaged.
S_Y = {"CYP1A2": 1.439, "CYP2C9": 0.966, "CYP2D6": 1.660, "CYP3A4": 1.160}


def sigma_star(rho: float, s_y: float) -> float:
    r = 2.0 * np.sin(np.pi * rho / 6.0)
    return float(s_y * np.sqrt(max(0.0, 1.0 - r * r)))


def main() -> int:
    bs = boards(fetch())
    per = {}
    for iso in CYPS:
        key = next(k for k in bs if "Live" in k and "Regression" in k and iso in k)
        b = bs[key]
        i_rho = next(i for i, h in enumerate(b["headers"]) if "Spearman" in h)
        i_user = b["headers"].index("Username")
        for r in b["data"]:
            try:
                rho = float(r[i_rho])
            except (TypeError, ValueError):
                continue
            per.setdefault(str(r[i_user]), {})[iso] = rho

    full = {u: v for u, v in per.items() if len(v) == 4}
    print(f"{len(per)} entrants, {len(full)} with all four isoforms scored\n")

    macro_sig, macro_rho = {}, {}
    for u, v in full.items():
        macro_sig[u] = float(np.mean([sigma_star(v[i], S_Y[i]) for i in CYPS]))
        macro_rho[u] = float(np.mean([v[i] for i in CYPS]))

    order = sorted(macro_sig, key=lambda u: macro_sig[u])
    ours = macro_sig.get(US)
    best_u = order[0]

    print("=== the real frontier: best MACRO sigma* actually achieved by a single entrant ===")
    print(f"{'rank':>4} {'entrant':26} {'macro sig*':>11} {'macro rho':>10}  " +
          "  ".join(f"{i[3:]:>6}" for i in CYPS))
    for n, u in enumerate(order[:8], 1):
        tag = u if len(u) <= 26 else u[:23] + "..."
        print(f"{n:4d} {tag:26} {macro_sig[u]:11.4f} {macro_rho[u]:10.4f}  " +
              "  ".join(f"{full[u][i]:6.3f}" for i in CYPS))
    if ours is not None:
        our_rank = order.index(US) + 1
        print(f"{our_rank:4d} {US:26} {ours:11.4f} {macro_rho[US]:10.4f}  " +
              "  ".join(f"{full[US][i]:6.3f}" for i in CYPS))

    print("\n=== the target, re-derived ===")
    if ours is None:
        print("  our row not scored on all four; cannot compare")
        return 1
    real_cut = 1.0 - macro_sig[best_u] / ours
    print(f"  ours              macro sigma* {ours:.4f}")
    print(f"  best real entrant macro sigma* {macro_sig[best_u]:.4f}  ({best_u})")
    print(f"  REAL CUT REQUIRED: {100*real_cut:.1f}%   (rank {order.index(US)+1} of {len(full)})")

    # The composite competitor who does not exist.
    synth = {i: max(full[u][i] for u in full) for i in CYPS}
    synth_sig = float(np.mean([sigma_star(synth[i], S_Y[i]) for i in CYPS]))
    print(f"\n  For contrast, the IMPOSSIBLE composite (best rho on each isoform, achieved by")
    print(f"  different entrants): " + "  ".join(f"{i[3:]} {synth[i]:.3f}" for i in CYPS))
    print(f"  its macro sigma* would be {synth_sig:.4f}, a {100*(1-synth_sig/ours):.1f}% cut.")
    print("  Quoting that as the target prices levers against a competitor who does not exist.")

    print("\n=== where OUR error actually sits (share of summed sigma*) ===")
    tot = sum(sigma_star(full[US][i], S_Y[i]) for i in CYPS)
    print(f"  {'iso':9} {'our sig*':>9} {'share':>7} {'best rho':>9} {'cut to best':>12}")
    for i in CYPS:
        s = sigma_star(full[US][i], S_Y[i])
        bi = max(full[u][i] for u in full)
        print(f"  {i:9} {s:9.3f} {100*s/tot:6.1f}% {bi:9.4f} "
              f"{100*(1 - sigma_star(bi, S_Y[i])/s):11.1f}%")
    print("\n  Two different questions, and they have different answers:")
    print("   * 'share' = where our absolute error lives -> what a better model would reduce most")
    print("   * 'cut to best' = how far the field is ahead -> what is demonstrably achievable")
    print("  A cell can dominate the error AND be nearly maxed out. That is not a contradiction;")
    print("  it means that cell is hard for everyone and is not where the field's edge comes from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
