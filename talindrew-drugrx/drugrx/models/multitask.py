"""A shared encoder with one head per ADMET endpoint.

Three encoders behind one interface, so Phase 2 can swap them without touching
the training loop:

* ``fingerprint`` — ECFP4 + RDKit descriptors through an MLP. Cheap, and the
  honest control: if the transformer cannot beat this, the transformer is not
  earning its compute.
* ``chemberta``   — a pretrained SMILES transformer (DeepChem/ChemBERTa-77M-MTR),
  fine-tuned. The default: with ~1k labels per endpoint, pretraining is where
  the signal has to come from.
* ``recursive``   — the TRM-style control for Phase 2: a small block applied
  repeatedly instead of a deep stack. Recursion is a **hypothesis here**, not a
  foundation — independent replications found TRM's puzzle results collapse
  without its task-ID embedding (see docs/drugrx-training/PLAYBOOK.md).

The loss masks missing labels, which is most cells, and weights tasks by learned
uncertainty (Kendall & Gal): with endpoints whose scales and difficulties differ
by orders of magnitude, fixed equal weights let the loudest task dominate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrx.data.tdc import AdmetTask


# ─────────────────────────────────────────────────────────────────────
# encoders
# ─────────────────────────────────────────────────────────────────────

class FingerprintEncoder(nn.Module):
    """ECFP4 bits + descriptors → a dense vector."""

    def __init__(self, in_dim: int, d_model: int = 512, n_layers: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)]
            d = d_model
        self.net = nn.Sequential(*layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.net(x)


class RecursiveEncoder(nn.Module):
    """One block applied `depth` times, TRM-style, with a residual each pass.

    The parameter count is that of a single block regardless of `depth`, which
    is the whole claim being tested: does recursion buy what extra layers buy,
    at a fraction of the parameters?

    Deep supervision is available by returning every intermediate state, so the
    trainer can supervise each pass rather than only the last.
    """

    def __init__(self, in_dim: int, d_model: int = 512, depth: int = 6,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.block = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.depth = depth
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, return_all: bool = False, **_: object):
        h = self.inp(x)
        states = []
        for _ in range(self.depth):
            h = self.norm(h + self.block(h))
            if return_all:
                states.append(h)
        return states if return_all else h


class ChemBertaEncoder(nn.Module):
    """A pretrained SMILES transformer, mean-pooled over real tokens.

    Mean pooling over the attention mask rather than the CLS token: these
    checkpoints were pretrained with a regression head rather than a sentence
    objective, so CLS carries less than the average does.
    """

    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR",
                 dropout: float = 0.1, freeze_layers: int = 0) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)
        self.out_dim = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        if freeze_layers:
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
            for layer in self.backbone.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                **_: object) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.dropout(pooled)


# ─────────────────────────────────────────────────────────────────────
# the model
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HeadSpec:
    key: str
    kind: str  # "regression" | "binary"


class MultiTaskAdmet(nn.Module):
    """Shared encoder, one small head per endpoint, masked multi-task loss."""

    def __init__(
        self,
        tasks: tuple[AdmetTask, ...],
        encoder: nn.Module,
        head_hidden: int = 256,
        dropout: float = 0.1,
        learn_task_weights: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.specs = [HeadSpec(t.key, t.task) for t in tasks]
        d = encoder.out_dim

        self.heads = nn.ModuleDict({
            spec.key: nn.Sequential(
                nn.Linear(d, head_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for spec in self.specs
        })

        # Kendall & Gal: learn log σ² per task and weight by 1/(2σ²), with
        # log σ as a regulariser so the model cannot simply shrink every task to
        # nothing. Initialised at 0 (σ=1), i.e. equal weights.
        self.log_var = (
            nn.Parameter(torch.zeros(len(self.specs))) if learn_task_weights else None
        )

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(**batch)
        if isinstance(h, list):  # deep supervision from RecursiveEncoder
            h = h[-1]
        return {spec.key: self.heads[spec.key](h).squeeze(-1) for spec in self.specs}

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Masked, uncertainty-weighted multi-task loss.

        `y` and `mask` are (batch, n_tasks). A task with no labelled example in
        the batch contributes nothing — and must not contribute a zero, which
        would drag its learned weight around for no reason.
        """
        total = y.new_zeros(())
        parts: dict[str, float] = {}
        n_used = 0

        for i, spec in enumerate(self.specs):
            m = mask[:, i]
            if not bool(m.any()):
                continue
            pred = preds[spec.key][m]
            target = y[m, i]

            if spec.kind == "binary":
                raw = F.binary_cross_entropy_with_logits(pred, target)
            else:
                raw = F.mse_loss(pred, target)

            if self.log_var is not None:
                lv = self.log_var[i]
                weighted = 0.5 * torch.exp(-lv) * raw + 0.5 * lv
            else:
                weighted = raw

            total = total + weighted
            parts[spec.key] = float(raw.detach())
            n_used += 1

        if n_used == 0:
            # Possible with a tiny batch and a sparse table; a zero with no
            # graph would break backward, so keep it connected.
            return total * 0.0 + sum(p.sum() * 0.0 for p in self.parameters()), parts

        return total / n_used, parts


def build_encoder(arch: str, in_dim: int = 0, **kwargs) -> nn.Module:
    if arch == "fingerprint":
        return FingerprintEncoder(in_dim=in_dim, **kwargs)
    if arch == "recursive":
        return RecursiveEncoder(in_dim=in_dim, **kwargs)
    if arch == "chemberta":
        return ChemBertaEncoder(**kwargs)
    raise ValueError(f"unknown arch {arch!r}")


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
