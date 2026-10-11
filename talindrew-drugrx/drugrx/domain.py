"""Applicability domain: is this query molecule inside the training distribution?

Every prediction this programme serves needs this attached, because the honest
unit of output is not a number but *a number plus whether we have any business
producing it*. The machinery already existed for audits — `nearest_neighbour_
similarity` and the background calibration that retired the Avalon row — and this
module is the same computation made (a) reusable at inference time, (b) packed
into something an artifact can carry, and (c) calibrated per endpoint.

**The threshold is calibrated, never fixed.** A raw "nearest neighbour above
Tanimoto 0.3" is not interpretable on its own: it depends on fingerprint density
and on how much chemistry the reference set contains. Avalon's background sits at
0.546 against Morgan's 0.265, which is how a 99.8%-above-0.4 figure came to look
like contamination when it was baseline (T6). The same trap applies here with an
extra twist — the reference set *size* moves the background too. A 12,000-molecule
CYP training set gives an ordinary drug-like molecule a much higher nearest
neighbour than a 201-molecule `half_life` set does, so one cut cannot serve both.

**What it is calibrated *against* is the part that took two attempts.** The obvious
choice — a high percentile of a random drug-like background — is calibrated and
still useless. Measured on our own data it lands near Tanimoto 0.53, while the mean
test-fold molecule sits at 0.25–0.42 from its own training set and only 27% of test
molecules clear even 0.4 (`runs/leakage_calibrated.json`). So that bar would flag
**roughly nine out of ten of the molecules our published scores were computed on**,
and a warning on nine rows in ten is indistinguishable from no warning at all. It is
the same failure as a badge on every row, arrived at by a defensible route.

So the threshold is a low quantile of the **validation** fold's own
nearest-neighbour distribution, and "in domain" means: **this molecule is at least
as close to the training set as all but the least typical few per cent of the
molecules this model's reported performance was measured on.** That is a statement a
reader can act on — out of domain means *our numbers do not describe molecules like
this one* — and it makes the flag rare enough to be worth reading.

**But the binary is a floor, and the percentile is the signal.** Moving the anchor
did not produce a clean separation, it moved the failure: a bar at the validation
5th percentile lands at 0.175 on caco2, *below* that endpoint's random-background
median of 0.259 — so most ordinary chemistry would now read in-domain. Neither
anchor works alone because **the two populations genuinely overlap**: the measured
shift of test-fold similarity over background is only +0.019 to +0.180
(`runs/leakage_calibrated.json`). There is no cut here that admits what we measured
on and excludes what we did not, and a binary that claims otherwise is asserting a
separation the data does not contain.

So every row also carries `measured_percentile` — where the query sits *within* the
distribution of molecules the scores were computed on. That is graded, it degrades
honestly, and it is what a caller should rank on. `in_domain` stays as the
conservative floor it can honestly be: **we have no evidence at all this far out.**

Validation rather than test on purpose: the threshold is then a property of a fold we
do not report scores on, which leaves the **test** fold's in-domain rate as an honest
measurement rather than a definition. No selection among alternatives happens on
either fold; the quantile is fixed up front (T7).

The random background is still computed and carried, because it is the context that
makes a similarity number readable at all, and the raw similarity is always returned
so the fixed-0.3 coverage figures in H9/H10 stay comparable with whatever this reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Morgan r=2, 2048 bits — the same fingerprint every similarity claim in this
#: programme has been made with. Changing it invalidates the comparison with
#: H9's 18% and H10's 11.2%, so it is a module constant rather than a parameter.
N_BITS = 2048
RADIUS = 2

#: The quantile of the **validation** fold's nearest-neighbour distribution that
#: sets the bar. At 5, a molecule is out of domain when it is less like the
#: training set than 95% of the molecules this model's scores were measured on.
#: It is a choice, named here so it reads as one, fixed before the run (T7), and
#: recorded in every reference built.
DEFAULT_QUANTILE = 5.0

#: Kept so older bundles and the audit scripts that predate the change still load.
DEFAULT_PERCENTILE = DEFAULT_QUANTILE


def fingerprints(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Packed binary Morgan fingerprints, and the indices that produced them.

    Returns `(packed, kept)` where `packed` is `(n_kept, N_BITS // 8)` uint8 and
    `kept` indexes back into `smiles`. **Unparseable molecules are reported, not
    silently dropped**: a caller aligning predictions to rows needs to know which
    rows vanished, and the audit layer's own history says so — a silently dropped
    row is more dangerous than a featureless one.

    Binary rather than count fingerprints, because Tanimoto over bits is what
    every similarity number in this programme already means.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows, kept = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        rows.append(np.packbits(gen.GetFingerprintAsNumPy(mol).astype(bool)))
        kept.append(i)
    if not rows:
        return np.zeros((0, N_BITS // 8), dtype=np.uint8), np.zeros(0, dtype=int)
    return np.vstack(rows).astype(np.uint8), np.array(kept, dtype=int)


def _popcount(packed: np.ndarray) -> np.ndarray:
    """Bits set per row. `np.bitwise_count` needs numpy ≥ 2.0."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(packed).sum(-1).astype(np.int32)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    return table[packed].sum(-1).astype(np.int32)


def max_tanimoto(query: np.ndarray, reference: np.ndarray,
                 chunk: int = 512) -> np.ndarray:
    """Nearest-neighbour Tanimoto from each query row to any reference row.

    Done over packed bits so a reference set of 12,000 molecules is 3 MB and the
    whole thing is integer arithmetic. Chunked over queries because the
    intermediate is `(chunk, n_ref, bytes)` and an unchunked 2,500-query call
    against a 12,000-molecule reference would ask for 7 GB.
    """
    if len(query) == 0 or len(reference) == 0:
        return np.zeros(len(query), dtype=np.float32)

    ref_bits = _popcount(reference)
    out = np.zeros(len(query), dtype=np.float32)
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        q_bits = _popcount(block)
        inter = _popcount(block[:, None, :] & reference[None, :, :])
        union = q_bits[:, None] + ref_bits[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        out[start:start + chunk] = sim.max(axis=1)
    return out


@dataclass(frozen=True)
class DomainReference:
    """One endpoint's training set, packed, with its calibrated threshold.

    `threshold` is the `quantile`-th percentile of the **validation** fold's
    nearest-neighbour similarity to this training set. A query at or above it is at
    least as close to the training data as all but the least typical `quantile`% of
    the molecules the model's reported performance was measured on.

    The `background_*` fields describe a random drug-like sample against the same
    reference. They do not set the threshold; they are the context that makes a
    similarity number readable, and the gap between `background_median` and
    `threshold` is the honest measure of how much proximity the training set buys.
    """

    endpoint: str
    packed: np.ndarray
    threshold: float
    quantile: float
    calibration_n: int
    calibration_median: float
    background_n: int
    background_mean: float
    background_median: float
    #: The calibration fold's own similarities, sorted. Carried in full — a few
    #: hundred floats — because the **percentile is the real signal and the binary
    #: is only a floor**. Neither population separates cleanly from the other (see
    #: the module docstring), so a query's position *within* the distribution of
    #: molecules we measured on says far more than which side of a cut it fell.
    calibration_sims: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    @property
    def percentile(self) -> float:
        """Back-compatible alias; older bundles called this `percentile`."""
        return self.quantile

    @property
    def n_reference(self) -> int:
        return int(len(self.packed))

    def to_npz(self) -> dict:
        """Arrays and scalars for `np.savez`, prefixed by endpoint by the caller."""
        return {
            "packed": self.packed,
            "calibration_sims": self.calibration_sims,
            "meta": np.array([
                self.threshold, self.quantile, self.background_n,
                self.background_mean, self.background_median,
                self.calibration_n, self.calibration_median,
            ], dtype=np.float64),
        }

    @classmethod
    def from_npz(cls, endpoint: str, packed: np.ndarray, meta: np.ndarray,
                 calibration_sims: np.ndarray | None = None) -> DomainReference:
        # Older bundles carry five fields and no calibration set. They load, with
        # the calibration fields absent rather than invented — a zero there would
        # read as "calibrated on nothing", which is exactly what it was.
        return cls(
            endpoint=endpoint, packed=packed, threshold=float(meta[0]),
            quantile=float(meta[1]), background_n=int(meta[2]),
            background_mean=float(meta[3]), background_median=float(meta[4]),
            calibration_n=int(meta[5]) if len(meta) > 5 else 0,
            calibration_median=float(meta[6]) if len(meta) > 6 else float("nan"),
            calibration_sims=(np.zeros(0, np.float32) if calibration_sims is None
                              else np.asarray(calibration_sims, dtype=np.float32)),
        )


def build_reference(endpoint: str, train_smiles: list[str],
                    background_smiles: list[str],
                    calibration_smiles: list[str],
                    quantile: float = DEFAULT_QUANTILE) -> DomainReference:
    """Pack a training set and calibrate its in-domain threshold.

    `calibration_smiles` is the **validation** fold — the population the threshold
    is defined against, and deliberately not the test fold, so the test fold's
    in-domain rate stays a measurement rather than a definition.

    `background_smiles` is ordinary drug-like chemistry with this endpoint's own
    training identities removed (`background_excluding`). It does not set the
    threshold; it is reported alongside so a similarity number can be read at all.
    """
    packed, _ = fingerprints(train_smiles)
    bg_packed, _ = fingerprints(background_smiles)
    cal_packed, _ = fingerprints(calibration_smiles)
    if len(packed) == 0:
        raise ValueError(f"{endpoint}: no parseable training molecules")
    if len(bg_packed) == 0:
        raise ValueError(f"{endpoint}: no parseable background molecules")
    if len(cal_packed) < 20:
        # Below this the quantile is a single molecule's similarity, which is a
        # threshold set by one compound rather than by a distribution.
        raise ValueError(
            f"{endpoint}: only {len(cal_packed)} calibration molecules; "
            "the quantile would not be estimable"
        )

    bg_sims = max_tanimoto(bg_packed, packed)
    cal_sims = max_tanimoto(cal_packed, packed)
    return DomainReference(
        endpoint=endpoint,
        packed=packed,
        threshold=float(np.percentile(cal_sims, quantile)),
        quantile=float(quantile),
        calibration_n=int(len(cal_sims)),
        calibration_median=float(np.median(cal_sims)),
        background_n=int(len(bg_sims)),
        background_mean=float(bg_sims.mean()),
        background_median=float(np.median(bg_sims)),
        calibration_sims=np.sort(cal_sims.astype(np.float32)),
    )


def membership(reference: DomainReference, query_smiles: list[str]) -> list[dict]:
    """One row per query molecule, aligned to the input, never shorter.

    An unparseable molecule gets `in_domain=False` with `reason="unparseable"`
    rather than being dropped — the caller is zipping these onto compounds and a
    length change would silently misalign every row after the bad one.
    """
    packed, kept = fingerprints(query_smiles)
    sims = max_tanimoto(packed, reference.packed)

    rows: list[dict] = [
        {"in_domain": False, "nn_similarity": None, "measured_percentile": None,
         "reason": "unparseable"}
        for _ in query_smiles
    ]
    calibration = reference.calibration_sims
    for position, index in enumerate(kept):
        sim = float(sims[position])
        rows[int(index)] = {
            # A floor, not a verdict. See `measured_percentile`.
            "in_domain": bool(sim >= reference.threshold),
            "nn_similarity": round(sim, 4),
            # **The number worth reading.** Where this molecule sits among the
            # molecules the model's reported scores were computed on: 40 means
            # "closer to the training set than 40% of them". It degrades to None
            # rather than to a default on an older reference that carries no
            # calibration distribution.
            # `side="right"` so a tie counts as "at least as close as". With
            # `side="left"` a query identical to a training molecule scored 52nd
            # percentile on a fixture where half the calibration set was also at
            # 1.0 — which is what happens whenever a set contains near-duplicates,
            # and ADMET datasets are full of them.
            "measured_percentile": (
                round(float(np.searchsorted(calibration, sim, side="right")
                            / len(calibration) * 100), 1)
                if len(calibration) else None
            ),
            # The fixed cut H9 and H10 reported, carried alongside so those
            # figures stay comparable with anything served from here.
            "above_0_3": bool(sim >= 0.3),
        }
    return rows


def background_excluding(candidates: list[str], reference: list[str],
                         key_of: dict[str, str]) -> list[str]:
    """Candidate background molecules with the reference set's identities removed.

    **This is the step that makes a calibrated threshold mean anything**, and it is
    here rather than in `ship_router.py` so the test pins the code that runs (T1).

    The background for an endpoint is pooled from the other endpoints' molecules,
    which is cheap and plentiful — but TDC's datasets overlap heavily, so without
    this a large share of an endpoint's "background" is its own training set, every
    member of which scores a nearest neighbour of 1.0 against itself. The threshold
    then rises until genuinely in-domain queries read as out-of-domain, and what it
    is measuring is dataset overlap rather than chemistry.

    `key_of` maps a SMILES to its **standardised identity**. Matching on the raw
    string is what matched 21 of lipophilicity's 420 molecules while reporting the
    comparison as clean; here it would leave every salt form and re-spelled
    tautomer of a training molecule sitting in its own background. A molecule with
    no entry in `key_of` is kept — it cannot be shown to be in the reference set,
    and dropping unknowns would shrink the background for the wrong reason.
    """
    excluded = {key_of[s] for s in reference if s in key_of}
    return [s for s in candidates if key_of.get(s) not in excluded]
