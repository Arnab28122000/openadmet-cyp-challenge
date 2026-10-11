"""The decoy-target control: would this screen rank the same compounds against the wrong protein?

**A ranking that does not move when the target is replaced is not about the target.**
It is a compound-quality prior — molecular weight, lipophilicity, how many atoms a
scoring function has to work with — wearing the appearance of a prediction about a
protein someone cares about. Nothing downstream of the pipeline can tell the two apart,
because both produce a sorted list with plausible numbers.

This is the same discipline the programme already applies to its own measurements: a
null-vs-null arm that must centre on 0.5 (T3), and a spike-recovery positive control
that must find a planted answer (T4). A detector certified only against the null cannot
distinguish "no effect" from "blind", and a screening pipeline validated only on targets
where it looks reasonable cannot distinguish "working" from "scoring the ligands".

It also answers, by measuring rather than citing, an external claim this programme chose
not to carry: that Boltz-2 retains high enrichment when the binding site is mutated
(`research/06-prospective-screening-reality.md` §4). We could not verify it. We can test
the consequence directly on our own pipeline, which is better than believing it.

## Two decoys, because they ask different questions

* **`shuffle`** — the same residues in a random order. Composition and length are
  preserved exactly, so anything that depends only on amino-acid content is unchanged;
  the fold is destroyed. This asks *does structure matter at all?* A shuffled sequence
  should fold to a low-confidence blob, and a screen that ranks identically against a
  blob is not using the structure it was given.
* **`unrelated`** — a different, real, well-folded protein. This asks the sharper
  question: *does **this** structure matter?* It controls for the possibility that the
  shuffle arm only shows that unfoldable inputs behave oddly.

Both are needed. A high correlation on `shuffle` alone is suggestive; a high correlation
on `unrelated` is the finding.

## Reading the number

`spearman` between the two candidate orderings, with the sampling band it would occupy
under independence — ±1.96/√(n−1), which at the 20-to-30-compound sets this pipeline
runs is **±0.4 or wider**. That band is the reason this module reports it rather than
leaving the comparison to a glance: at n = 20 a correlation of 0.35 is indistinguishable
from zero, and reporting it as "some target specificity" would be inventing a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

#: A real, well-folded, unrelated protein for the `unrelated` arm. Hen egg-white
#: lysozyme: 129 residues, one of the most-solved structures in the PDB, and
#: biologically unrelated to every target in the playground's catalogue. It is already
#: a demo target in `templates.py`, so the pipeline is known to handle it.
LYSOZYME = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
    "WWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQA"
    "WIRGCRL"
)


def shuffled_target(sequence: str, seed: int = 0) -> str:
    """The same residues, reordered. Composition and length exactly preserved.

    Deterministic in `seed`, because a control that cannot be reproduced is an
    anecdote. Note what this does *not* do: it does not try to avoid producing
    something foldable. Scrambling a sequence essentially never yields a stable fold,
    and testing for that would be inventing a criterion we cannot evaluate.
    """
    residues = list(sequence)
    np.random.default_rng(seed).shuffle(residues)
    return "".join(residues)


def unrelated_target(_sequence: str) -> str:
    """A different real protein, independent of the one being screened."""
    return LYSOZYME


@dataclass(frozen=True)
class RankComparison:
    """How much a ranking moved when the target was replaced."""

    n: int
    spearman: float
    #: ±1.96/sqrt(n-1): where this statistic sits under independence. At the compound
    #: counts this pipeline runs it is wide, and quoting the correlation without it
    #: would turn noise into a claim.
    null_band: float
    #: Fraction of the real run's top-k that the decoy run also puts in its top-k.
    top_k: int
    top_k_overlap: float
    #: Did the compound sets even match? A comparison across different molecules is
    #: not a comparison.
    shared: int

    @property
    def distinguishable(self) -> bool:
        """Is the correlation outside the band it would occupy by chance?"""
        return abs(self.spearman) > self.null_band

    @property
    def verdict(self) -> str:
        """The reading, stated so it cannot be softened after the fact.

        The uncomfortable outcome has its own name. A screen whose ranking survives
        having its target replaced is reporting a property of the compounds, and that
        is a finding about the pipeline rather than a failure of the experiment.
        """
        if self.shared < 5:
            return "too few shared compounds to compare"
        if not self.distinguishable:
            return "indistinguishable from chance — the design cannot resolve it at this n"
        if self.spearman >= 0.8:
            return "ranking is target-independent: this is a compound-quality prior"
        if self.spearman >= 0.4:
            return "ranking is substantially target-independent"
        return "ranking is target-specific"

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "shared": self.shared,
            "spearman": round(self.spearman, 4),
            "null_band": round(self.null_band, 4),
            "distinguishable": self.distinguishable,
            "top_k": self.top_k,
            "top_k_overlap": round(self.top_k_overlap, 4),
            "verdict": self.verdict,
        }


def compare_rankings(real: Sequence[str], decoy: Sequence[str],
                     top_k: int = 10) -> RankComparison:
    """Compare two candidate orderings, both given best-first.

    The inputs are compound **ids** in rank order, not scores: scores from two runs are
    not on a common scale and comparing them directly would measure the scale rather
    than the order.
    """
    shared = [cid for cid in real if cid in set(decoy)]
    n = len(shared)
    if n < 2:
        return RankComparison(n=n, spearman=float("nan"), null_band=float("inf"),
                              top_k=top_k, top_k_overlap=float("nan"), shared=n)

    real_rank = {cid: i for i, cid in enumerate(real)}
    decoy_rank = {cid: i for i, cid in enumerate(decoy)}
    a = np.array([real_rank[c] for c in shared], dtype=float)
    b = np.array([decoy_rank[c] for c in shared], dtype=float)

    # Spearman is Pearson on the ranks; computed directly so this module needs no
    # scipy, which the serving image does not carry.
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    rho = float((a * b).sum() / denom) if denom else float("nan")

    k = min(top_k, len(real), len(decoy))
    overlap = (len(set(real[:k]) & set(decoy[:k])) / k) if k else float("nan")

    return RankComparison(
        n=n, spearman=rho, null_band=1.96 / math.sqrt(n - 1),
        top_k=k, top_k_overlap=overlap, shared=n,
    )


def ranking_from_candidates(candidates: Iterable[dict],
                            score_key: str = "composite_score") -> list[str]:
    """Compound ids, best first, from a FoldRx `rank` stage's candidate list.

    Sorted here rather than trusted: the engine already sorts, but a control that
    assumes its input is ordered would silently compare two arbitrary orders if that
    ever changed.
    """
    rows = [c for c in candidates if c.get("id") is not None]
    rows.sort(key=lambda c: -(float(c.get(score_key) or 0.0)))
    return [str(c["id"]) for c in rows]
