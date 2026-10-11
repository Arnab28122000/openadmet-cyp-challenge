"""Route each ADMET endpoint to whichever model actually wins it.

H3 settled that neither model dominates: on TDC's official test folds the
multi-task net wins 10 endpoints and per-endpoint LightGBM wins 11. The split is
not arbitrary — the net wins where the endpoint is about whole-molecule
behaviour (distribution, transport, clearance, channel binding) and LightGBM
wins where it is about specific substructures (the CYP inhibition panel,
mutagenicity, and the solubility/lipophilicity/permeability regressions).

So the honest thing to serve is neither model alone. This builds the routing
table **from the committed measurements**, not from a hand-written preference,
so it changes when the evidence changes and cannot drift from it.

**The routing table is itself a test-set decision**, which is a real caveat: by
choosing the per-endpoint winner on test we spend a little of that test set's
independence. It is one bit per endpoint — far less than tuning a model against
it — but it is not nothing, and an endpoint whose two models are within noise
should route by the simpler model rather than by a coin-flip margin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from drugrx.data.tdc import ADMET_TASKS, LOWER_IS_BETTER, TASK_BY_KEY

#: Below this relative margin the two models are treated as tied, and the tie is
#: broken toward LightGBM — it is cheaper to serve, needs no GPU, and is the
#: model whose failure modes are better understood. "The deep model wins by
#: 0.3%" is not a reason to put a GPU in the serving path.
TIE_MARGIN = 0.01


@dataclass(frozen=True)
class Route:
    endpoint: str
    model: str            # "nn" | "lightgbm"
    metric: str
    nn: float
    lightgbm: float
    margin: float         # relative, positive means the winner is ahead
    tied: bool
    reason: str


def _relative_margin(metric: str, nn: float, gbm: float) -> float:
    """How far apart the two are, as a fraction, in the metric's own direction."""
    scale = max(abs(nn), abs(gbm), 1e-9)
    raw = (gbm - nn) if metric in LOWER_IS_BETTER else (nn - gbm)
    return raw / scale


def build_routes(eval_json: str | Path, baseline_json: str | Path) -> list[Route]:
    """Build the table from a run's test evaluation and the committed baseline."""
    ev = json.loads(Path(eval_json).read_text())
    nn_scores = {r["task"]: r["nn"] for r in ev["rows"]}

    base_blob = json.loads(Path(baseline_json).read_text())
    gbm_scores = {
        r["task"]: float(r["official_split"]["value"])
        for r in base_blob.get("results", [])
        if "official_split" in r
    }

    routes: list[Route] = []
    for task in ADMET_TASKS:
        if task.key not in nn_scores or task.key not in gbm_scores:
            continue
        nn, gbm = nn_scores[task.key], gbm_scores[task.key]
        margin = _relative_margin(task.metric, nn, gbm)
        tied = abs(margin) < TIE_MARGIN

        if tied:
            model = "lightgbm"
            reason = (f"within {TIE_MARGIN:.0%} ({margin:+.1%}); tie broken toward LightGBM "
                      "— no GPU in the serving path for a margin this small")
        elif margin > 0:
            model, reason = "nn", f"multi-task net ahead by {margin:.1%}"
        else:
            model, reason = "lightgbm", f"LightGBM ahead by {-margin:.1%}"

        routes.append(Route(task.key, model, task.metric, nn, gbm, abs(margin), tied, reason))
    return routes


def write_table(routes: list[Route], out: str | Path) -> None:
    lines = [
        "# ADMET routing table",
        "",
        "Which model serves each endpoint, derived from the committed test-fold measurements "
        "(`runs/h3/eval.json` and `runs/baseline/results.json`) by `drugrx/hybrid.py`. "
        "Regenerate it rather than editing it.",
        "",
        f"Ties (within {TIE_MARGIN:.0%}) go to LightGBM: it needs no GPU and its failure modes are "
        "better understood, so a sub-1% win is not worth a GPU in the serving path.",
        "",
        "| endpoint | serves | metric | net | LightGBM | margin | why |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(routes, key=lambda r: (r.model, -r.margin)):
        lines.append(
            f"| {r.endpoint} | **{r.model}** | {r.metric} | {r.nn:.4f} | {r.lightgbm:.4f} | "
            f"{r.margin:.1%}{' (tied)' if r.tied else ''} | {r.reason} |"
        )

    n_nn = sum(1 for r in routes if r.model == "nn")
    lines += [
        "",
        f"**{n_nn} endpoints served by the net, {len(routes) - n_nn} by LightGBM.**",
        "",
        "Caveat carried from the leakage audit: both columns are official-split numbers, and 47% of "
        "that test set has a close training analogue. The *comparison* is sound; the absolute values "
        "are optimistic for genuinely new chemistry.",
        "",
        "Second caveat: choosing the winner per endpoint on test spends a little of that test set's "
        "independence — one bit per endpoint. That is why ties go to the simpler model rather than "
        "to the better number.",
        "",
    ]
    Path(out).write_text("\n".join(lines))


def routing_dict(routes: list[Route]) -> dict[str, str]:
    """`{endpoint: "nn" | "lightgbm"}`, for the serving layer to consume."""
    return {r.endpoint: r.model for r in routes}
