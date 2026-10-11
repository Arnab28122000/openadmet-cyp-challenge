"""The router: load the bundle, dispatch per endpoint, attach the domain.

This is the module the serving container imports, and it is the whole reason
Phase 4 is not just "upload the weights". Everything this programme established
lives outside the tensor file:

* **gradient boosting serves most endpoints**, because the ensemble's advantage
  over it is statistically undecided nearly everywhere — so a single-model predict
  surface would contradict the one decision in this programme with an effect size
  behind it;
* **a prediction without its applicability domain is not a prediction** we have any
  business producing, so `in_domain` and the nearest-neighbour similarity that
  decided it ride on every value;
* **the ExpansionRx heads are a different measurement scale** from any TDC target
  and live in their own namespace, so that comparing `ExpansionRx_CLint` with
  `Clearance_Microsome_AZ` is obviously wrong rather than merely discouraged.

Two things it deliberately does not do:

**It invents no interval for gradient boosting.** The ensemble has a member spread
and that is a real, if imperfect, uncertainty; a booster has nothing comparable,
and manufacturing one from residuals would be the "estimate that reads as a
measurement" the FoldRx registry already forbids. `interval` is `None` there, and a
reader can tell the difference.

**It never substitutes one endpoint for another.** If an endpoint's model is missing
from the bundle it is absent from the output, not filled from a neighbour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# **LightGBM must be imported before torch, and this line is load-bearing.**
# Both ship their own libomp, and on macOS arm64 the second one to load aborts
# the process outright — `Fatal Python error: Aborted` inside LightGBM's
# `__init_from_np2d`, with no Python traceback to read. Measured: `import
# lightgbm, torch` is fine, `import torch, lightgbm` dies. Since this module is
# the serving entry point and reaches for both, it fixes the order here rather
# than hoping every caller gets it right. Do not move this below the torch
# imports, and do not make it lazy.
import lightgbm as lgb  # noqa: F401  (imported for its side effect on load order)
import numpy as np

from drugrx.domain import DomainReference, membership

#: ExpansionRx's own column names carry spaces and a `>` ("Caco-2 Permeability
#: Papp A>B"), which make poor response keys and worse test selectors. These are
#: stable slugs; the original assay name travels in the metadata, because the
#: label is the thing a reader needs and the slug is the thing code needs.
#:
#: Two of these name properties that TDC also measures — `MLM_CLint` against
#: `Clearance_Microsome_AZ`, `Caco2_Papp_AB` against `Caco2_Wang`. That is exactly
#: the confusion the `ExpansionRx_` prefix exists to prevent: different
#: laboratories, different protocols, panels that share **zero** molecules, so no
#: offset between them is estimable and no conversion exists.
ERX_SLUGS: dict[str, str] = {
    "LogD": "LogD",
    "KSOL": "KineticSolubility",
    "HLM CLint": "HLM_CLint",
    "MLM CLint": "MLM_CLint",
    "Caco-2 Permeability Papp A>B": "Caco2_Papp_AB",
    "Caco-2 Permeability Efflux": "Caco2_Efflux",
}


def erx_key(head: str) -> str:
    """`ExpansionRx_<slug>`, falling back to a sanitised form for a new column."""
    slug = ERX_SLUGS.get(head)
    if slug is None:
        slug = "".join(c if (c.isalnum() or c == "_") else "_" for c in head).strip("_")
    return f"ExpansionRx_{slug}"


#: ±1.96σ. Named because it is a choice: the member spread is not a calibrated
#: predictive interval, and this turns it into one only under an assumption the
#: spread has not been shown to satisfy. It is reported as a spread-derived band,
#: labelled as such, rather than as a confidence interval.
Z95 = 1.96


@dataclass
class EnsembleBundle:
    """The neural checkpoint, with everything needed to reproduce its featuriser."""

    params: dict
    buffers: dict
    in_dim: int
    n_tasks: int
    tdc_tasks: list[str]
    erx_heads: list[str]
    feature_stats: tuple[np.ndarray, np.ndarray]
    label_norm: dict[str, tuple[float, float]]
    architecture: dict
    extra_fps: bool


@dataclass
class Router:
    """One endpoint, one model, chosen by the measured routing table."""

    gbm: dict = field(default_factory=dict)
    gbm_meta: dict = field(default_factory=dict)
    domain: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    ensemble: EnsembleBundle | None = None
    #: endpoint key → display name, and the reverse. Output is keyed by display
    #: name because that is the vocabulary admet-ai already uses.
    names: dict = field(default_factory=dict)

    # ── loading ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Router:
        root = Path(path)
        meta = json.loads((root / "gbm" / "meta.json").read_text())["endpoints"]

        gbm, domain, names = {}, {}, {}
        for key, info in meta.items():
            booster = root / "gbm" / f"{key}.txt"
            if booster.exists():
                gbm[key] = lgb.Booster(model_file=str(booster))
            reference = root / "domain" / f"{key}.npz"
            if reference.exists():
                with np.load(reference) as handle:
                    domain[key] = DomainReference.from_npz(
                        key, handle["packed"], handle["meta"],
                        handle["calibration_sims"]
                        if "calibration_sims" in handle else None)
            names[key] = info["name"]

        routing = {}
        routing_file = root / "routing.json"
        if routing_file.exists():
            blob = json.loads(routing_file.read_text())
            routing = {r["task"]: r["serves"] for r in blob.get("rows", [])}

        ensemble = None
        checkpoint = root / "ensemble.pt"
        if checkpoint.exists():
            ensemble = cls._load_ensemble(checkpoint, device)

        # **An endpoint routed to the ensemble with no ensemble present must fail
        # loudly.** Falling back to the booster would silently serve a model the
        # routing table says is worse there, and the output would look identical.
        wanted = [k for k, v in routing.items() if v == "ensemble"]
        if wanted and ensemble is None:
            raise ValueError(
                f"routing.json sends {wanted} to the ensemble but no ensemble.pt "
                "is in the bundle"
            )
        return cls(gbm=gbm, gbm_meta=meta, domain=domain, routing=routing,
                   ensemble=ensemble, names=names)

    @staticmethod
    def _load_ensemble(path: Path, device: str) -> EnsembleBundle:
        import torch

        blob = torch.load(path, map_location=device, weights_only=False)
        stats = blob["feature_stats"]
        # `extra_fps` is read from the checkpoint when it is there and inferred
        # from the width when it is not — older checkpoints predate the field, and
        # getting this wrong makes every prediction wrong while still being a number.
        widths = {2264: False, 3091: True}
        extra = blob.get("extra_fps")
        if extra is None:
            if blob["in_dim"] not in widths:
                raise ValueError(
                    f"checkpoint in_dim {blob['in_dim']} is neither 2264 nor 3091 "
                    "and carries no extra_fps field, so its featuriser is unknown")
            extra = widths[blob["in_dim"]]
        return EnsembleBundle(
            params={k: v.to(device) for k, v in blob["params"].items()},
            buffers={k: v.to(device) for k, v in blob["buffers"].items()},
            in_dim=int(blob["in_dim"]),
            n_tasks=int(blob["n_tasks"]),
            tdc_tasks=list(blob["tdc_tasks"]),
            erx_heads=list(blob.get("erx_heads") or []),
            feature_stats=(np.array(stats["mean"], dtype=np.float32),
                           np.array(stats["std"], dtype=np.float32)),
            label_norm={k: tuple(v) for k, v in blob["label_norm"].items()},
            architecture=dict(blob["architecture"]),
            extra_fps=bool(extra),
        )

    # ── prediction ────────────────────────────────────────────────────────

    def served_by(self, key: str) -> str:
        """Which model answers for this endpoint.

        Defaults to gradient boosting when the routing table is silent, because
        that is the cheaper model whose failure modes are better understood, and
        because "undecided" must never resolve to the GPU model by omission.
        """
        return "ensemble" if self.routing.get(key) == "ensemble" else "gbm"

    def predict(self, smiles: list[str], device: str = "cpu",
                workers: int | None = None) -> list[dict]:
        """One row per input molecule, aligned, with metadata on every value."""
        from drugrx.features import featurise

        rows: list[dict] = [{"endpoints": {}, "endpoint_meta": {}} for _ in smiles]
        if not smiles:
            return rows

        wanted = [k for k in self.gbm_meta if self.served_by(k) == "gbm" and k in self.gbm]
        ens_keys = [k for k in self.gbm_meta if self.served_by(k) == "ensemble"]

        # ── gradient boosting, grouped by arm so each featuriser runs once ──
        by_arm: dict[str, list[str]] = {}
        for key in wanted:
            by_arm.setdefault(self.gbm_meta[key]["arm"], []).append(key)

        for arm, keys in by_arm.items():
            x = featurise(smiles, workers=workers, **_arm_options(arm))
            for key in keys:
                values = np.asarray(self.gbm[key].predict(x), dtype=float)
                self._emit(rows, key, values, None, "gbm", smiles)

        # ── the ensemble, for the endpoints that earned it ──────────────────
        if ens_keys and self.ensemble is not None:
            mean, spread = self._ensemble_predict(smiles, device, workers)
            for key in ens_keys:
                if key not in self.ensemble.tdc_tasks:
                    continue
                i = self.ensemble.tdc_tasks.index(key)
                self._emit(rows, key, mean[:, i], spread[:, i], "ensemble", smiles)

            # The auxiliary heads, in their own namespace, always from the
            # ensemble (nothing else has them) and never as a TDC endpoint.
            for j, head in enumerate(self.ensemble.erx_heads):
                col = len(self.ensemble.tdc_tasks) + j
                if col >= mean.shape[1]:
                    continue
                name = erx_key(head)
                for row_index in range(len(smiles)):
                    rows[row_index]["endpoints"][name] = round(float(mean[row_index, col]), 4)
                    rows[row_index]["endpoint_meta"][name] = {
                        "served_by": "ensemble",
                        "in_domain": None,
                        "nn_similarity": None,
                        "assay": head,
                        "scale": "ExpansionRx assay, z-scored on their panel",
                        "comparable_to_tdc": False,
                        "note": "a different lab's measurement; no offset to any "
                                "TDC endpoint is estimable — the two panels share "
                                "zero molecules",
                    }
        return rows

    def _ensemble_predict(self, smiles: list[str], device: str,
                          workers: int | None) -> tuple[np.ndarray, np.ndarray]:
        import torch

        from drugrx.features import featurise, standardise_descriptors
        from drugrx.fast_train import predict_ensemble

        bundle = self.ensemble
        assert bundle is not None
        x = featurise(smiles, workers=workers, counts=True, desc_mode="rdkit2d",
                      extra_fps=bundle.extra_fps)
        if x.shape[1] != bundle.in_dim:
            raise ValueError(
                f"featurised to {x.shape[1]} but the checkpoint wants {bundle.in_dim}")
        x, _ = standardise_descriptors(x, stats=bundle.feature_stats)

        base = _plain_mlp(bundle).to("meta")
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        with torch.no_grad():
            preds = predict_ensemble(base, bundle.params, bundle.buffers, xt)
        mean = preds.mean(0).float().cpu().numpy()
        spread = preds.std(0).float().cpu().numpy()

        # De-normalise regressions and squash classifications, endpoint by
        # endpoint — the checkpoint's outputs are in z-space for regressions and
        # logits for classifications, and mixing those up produces a plausible
        # number on the wrong scale.
        from drugrx.data.tdc import TASK_BY_KEY

        for i, key in enumerate(bundle.tdc_tasks):
            task = TASK_BY_KEY.get(key)
            if task is None:
                continue
            if task.task == "regression":
                mu, sd = bundle.label_norm.get(key, (0.0, 1.0))
                mean[:, i] = mean[:, i] * sd + mu
                spread[:, i] = spread[:, i] * sd
            else:
                logit = mean[:, i]
                probability = 1.0 / (1.0 + np.exp(-logit))
                # Spread on the probability scale, to first order. Stated as an
                # approximation because that is what it is.
                spread[:, i] = spread[:, i] * probability * (1.0 - probability)
                mean[:, i] = probability
        return mean, spread

    def _emit(self, rows: list[dict], key: str, values: np.ndarray,
              spread: np.ndarray | None, served_by: str, smiles: list[str]) -> None:
        name = self.names.get(key, key)
        info = self.gbm_meta.get(key, {})
        reference = self.domain.get(key)
        domain_rows = (membership(reference, smiles) if reference is not None
                       else [{"in_domain": None, "nn_similarity": None} for _ in smiles])

        for index in range(len(smiles)):
            value = float(values[index])
            meta = {
                "served_by": served_by,
                "in_domain": domain_rows[index].get("in_domain"),
                "nn_similarity": domain_rows[index].get("nn_similarity"),
                # Where this molecule sits among the ones the reported scores were
                # computed on. This is the graded signal; `in_domain` is a floor.
                "measured_percentile": domain_rows[index].get("measured_percentile"),
                "domain_threshold": (round(reference.threshold, 4)
                                     if reference is not None else None),
                "metric": info.get("metric"),
                "task": info.get("task"),
                "interval": None,
            }
            if spread is not None:
                half = Z95 * float(spread[index])
                meta["interval"] = [round(value - half, 4), round(value + half, 4)]
                meta["interval_basis"] = "ensemble member spread x1.96, not a calibrated interval"
            rows[index]["endpoints"][name] = round(value, 4)
            rows[index]["endpoint_meta"][name] = meta


def _arm_options(arm: str) -> dict:
    """The featuriser settings for an arm name, kept in one place.

    These must match `baseline.ARMS` exactly: a booster fitted on one feature
    layout and fed another produces numbers, not errors.
    """
    if arm == "maplight":
        return {"counts": True, "desc_mode": "rdkit2d", "extra_fps": True}
    if arm == "count-rdkit2d":
        return {"counts": True, "desc_mode": "rdkit2d", "extra_fps": False}
    raise ValueError(f"unknown arm {arm!r}")


def _plain_mlp(bundle: EnsembleBundle):
    """The architecture the checkpoint was trained with.

    Defined here rather than imported from `scripts/train_fast.py`, because the
    serving container ships the package and not the scripts directory — and a
    serve path that depends on a scripts import is a serve path that breaks the
    first time the image is slimmed.
    """
    import torch.nn as nn

    arch = bundle.architecture
    d_model = int(arch["d_model"])
    layers: list[nn.Module] = []
    d = bundle.in_dim
    for _ in range(int(arch["n_layers"])):
        layers += [nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU(),
                   nn.Dropout(float(arch["dropout"]))]
        d = d_model

    class PlainMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(d_model, bundle.n_tasks)

        def forward(self, x):
            return self.head(self.body(x))

    return PlainMLP()
