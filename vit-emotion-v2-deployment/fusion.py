"""
Late fusion of the face and text modalities over the shared 7-class space.

The method is a logarithmic opinion pool - a weighted geometric mean of
the two probability vectors, renormalised:

    log s[c] = sum_m  e_m * r_m[c] * (log p_m[c] - mean_c log p_m[c])
    p_fused  = softmax(log s / T)

Why a geometric mean rather than an arithmetic one
--------------------------------------------------
Averaging probabilities (a linear pool) produces a distribution at least
as spread out as its inputs - it can never be more confident than the
modalities it combines, and it keeps mass on any class either modality
merely tolerates. The geometric pool is a product of experts: it
multiplies, so a class both modalities find plausible is reinforced and
a class either one confidently rules out is suppressed. That is the
right behaviour when the modalities are close to conditionally
independent given the true emotion, which face pixels and typed words
largely are - two different channels reporting on one hidden state.

What the exponent e_m means
---------------------------
e_m is how many calibrated observations this modality is currently
worth, in [0, 1] per modality:

    e_face = base_face * quality
    e_text = base_text * recency * informativeness

    quality         is this frame's face worth looking at at all?
                    (detector confidence, crop size, blur, exposure,
                     head pose, temporal stability - see face_pipeline)
    recency         how long ago was this typed? Face evidence is
                    continuous; a message is one observation that goes
                    stale. Without this a single message would pin the
                    fused label for the rest of the session.
    informativeness is there enough text to carry an emotion at all?
                    "ok" is not evidence the way a sentence is.

Note what is deliberately *absent* from the exponent: the prediction's
own confidence. A log pool already handles that for free - a uniform
p_m has a flat log vector, which is a constant offset and cancels
exactly in the softmax. A modality that is merely guessing therefore
contributes nothing without being told to. Putting a certainty term in
the exponent as well would discount the same uncertainty twice and, at
e_m < 1, would flatten a single trustworthy modality below its own
honest confidence.

Certainty is still computed - it is what the UI shows to explain a low
influence, and it is what `influence` is scaled by - but it steers
attribution, not the arithmetic.

Why the log-probabilities are centred
-------------------------------------
Per-class reliability r_m[c] scales each modality by how well it
actually performs on that class: the face model scores 0.86 F1 on
'happy' and 0.52 on 'fear', so it should lead where it is strong. But
because r varies across c, an uncentred r[c]*log p[c] would stop a
uniform p from cancelling - log(1/7) is negative, so high-reliability
classes would be pushed *down* by a modality that said nothing at all.
Subtracting the mean log-probability first makes a uniform input
contribute exactly zero for any r, and changes nothing when r is flat.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from labels import NUM_SHARED_LABELS, SHARED_LABELS

# Guards log(0) in the pool. Small enough not to distort a real
# probability, large enough that a zeroed class does not produce -inf
# and poison the whole vector.
EPSILON = 1e-6

RELIABILITY_PATH = Path(__file__).resolve().parent / "reliability.json"

# log(7): entropy of a uniform distribution over the shared space, used
# to normalise certainty onto [0, 1].
MAX_ENTROPY = math.log(NUM_SHARED_LABELS)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class ModalityProfile:
    """Static, measured competence of one modality."""

    name: str
    base_weight: float
    reliability: np.ndarray   # length 7, normalised to mean 1.0

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> ModalityProfile:
        per_class = config.get("per_class_f1", {})

        values = np.array(
            [float(per_class.get(label, 1.0)) for label in SHARED_LABELS],
            dtype=np.float64,
        )

        values = np.clip(values, 1e-3, None)

        # Normalise to mean 1. Only the ratios between classes carry
        # meaning; the absolute level would otherwise silently act as a
        # second, hidden base weight.
        values = values / values.mean()

        return cls(
            name=name,
            base_weight=float(config.get("base_weight", 1.0)),
            reliability=values,
        )


@dataclass
class FusionConfig:
    """Everything tunable about the pool, loaded from reliability.json."""

    face: ModalityProfile
    text: ModalityProfile

    text_half_life_seconds: float = 30.0
    text_cutoff_seconds: float = 180.0
    min_informative_tokens: float = 8.0
    min_effective_weight: float = 0.02
    fusion_temperature: float = 1.0
    low_agreement_threshold: float = 0.45

    @classmethod
    def load(cls, path: Path | None = None) -> FusionConfig:
        path = RELIABILITY_PATH if path is None else path

        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        dynamics = raw.get("dynamics", {})

        return cls(
            face=ModalityProfile.from_config("face", raw.get("face", {})),
            text=ModalityProfile.from_config("text", raw.get("text", {})),
            text_half_life_seconds=float(
                dynamics.get("text_half_life_seconds", 30.0)
            ),
            text_cutoff_seconds=float(
                dynamics.get("text_cutoff_seconds", 180.0)
            ),
            min_informative_tokens=float(
                dynamics.get("min_informative_tokens", 8.0)
            ),
            min_effective_weight=float(
                dynamics.get("min_effective_weight", 0.02)
            ),
            fusion_temperature=float(dynamics.get("fusion_temperature", 1.0)),
            low_agreement_threshold=float(
                dynamics.get("low_agreement_threshold", 0.45)
            ),
        )


# ============================================================
# FACTORS
# ============================================================


def certainty(probabilities: np.ndarray) -> float:
    """Normalised negative entropy, in [0, 1].

    1.0 for a one-hot prediction, 0.0 for a uniform one. Reported to
    explain a modality's influence; see the module docstring for why it
    is deliberately not part of the pool exponent.
    """

    values = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0)
    values = values / values.sum()

    entropy = float(-np.sum(values * np.log(values)))

    return float(np.clip(1.0 - entropy / MAX_ENTROPY, 0.0, 1.0))


def recency(age_seconds: float, half_life: float, cutoff: float) -> float:
    """Exponential decay of text evidence, hard-zeroed past the cutoff.

    The hard cutoff exists so a stale session collapses cleanly to
    face-only, rather than trailing an ever-smaller but nonzero term
    that keeps the UI claiming text is still in play.
    """

    if age_seconds < 0:
        age_seconds = 0.0

    if age_seconds >= cutoff:
        return 0.0

    return float(0.5 ** (age_seconds / max(half_life, 1e-6)))


def informativeness(token_count: int, minimum: float) -> float:
    """How much emotional evidence a message of this length can carry.

    A two-word reply genuinely says less than a sentence does, and the
    classifier is no less confident about it - short inputs are exactly
    where its confidence is least earned.
    """

    if token_count <= 0:
        return 0.0

    return float(np.clip(token_count / max(minimum, 1e-6), 0.0, 1.0))


def agreement(first: np.ndarray, second: np.ndarray) -> float:
    """Bhattacharyya coefficient between two distributions, in [0, 1].

    1.0 when the modalities agree exactly, 0.0 only when their supports
    are disjoint. Reported rather than acted on: when the face says
    'happy' and the text says 'angry', the honest output is that the
    modalities disagree, not a confident blend of two contradictory
    claims.

    Note the *practical* range is much narrower than [0, 1]. Over seven
    classes both distributions keep mass on the five they do not favour,
    and those shared tails floor the coefficient well above zero.
    Measured over realistic pairs:

        happy .8  / happy .8    1.00   identical
        happy .5  / happy .8    0.95   same call, different confidence
        happy .5  / neutral .5  0.83   weak disagreement
        happy .6  / angry .6    0.73   clear disagreement
        happy .7  / angry .85   0.52   the "smiling while typing
                                       'i am furious'" demo case
        happy .99 / angry .99   0.09   only at implausible confidence

    So a conflict threshold belongs near 0.8, not near 0 - which is why
    it is paired with an argmax-differs test in `FusionEngine.fuse`
    rather than used alone.
    """

    a = np.clip(np.asarray(first, dtype=np.float64), 0.0, None)
    b = np.clip(np.asarray(second, dtype=np.float64), 0.0, None)

    if a.sum() <= 0 or b.sum() <= 0:
        return 0.0

    a = a / a.sum()
    b = b / b.sum()

    return float(np.clip(np.sum(np.sqrt(a * b)), 0.0, 1.0))


# ============================================================
# EVIDENCE
# ============================================================


@dataclass
class TextEvidence:
    """The most recent text message and when it arrived."""

    probabilities: np.ndarray
    token_count: int
    text: str
    timestamp: float
    label: str
    confidence: float

    def age(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return max(0.0, now - self.timestamp)


@dataclass
class ModalityContribution:
    """One modality's pool exponent, display influence, and the factors
    that produced them.

    Every factor is kept rather than just the product, because the whole
    point of a dynamic weight is that the user can see *why* a modality
    was trusted or ignored on this particular frame.
    """

    present: bool
    exponent: float = 0.0            # what the pool actually uses
    influence: float = 0.0           # normalised share, for the UI
    factors: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "exponent": round(self.exponent, 4),
            "share": round(self.influence, 4),
            "factors": {
                name: round(value, 4) for name, value in self.factors.items()
            },
        }


@dataclass
class FusionResult:
    """The fused decision plus a full account of how it was reached."""

    label: str | None
    confidence: float
    probabilities: np.ndarray | None
    face: ModalityContribution
    text: ModalityContribution
    agreement: float
    conflicted: bool
    mode: str   # "fused" | "face_only" | "text_only" | "none"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "mode": self.mode,
            "agreement": round(self.agreement, 4),
            "conflicted": self.conflicted,
            "weights": {
                "face": round(self.face.influence, 4),
                "text": round(self.text.influence, 4),
            },
            "face": self.face.as_dict(),
            "text": self.text.as_dict(),
            "probs": None,
        }

        if self.probabilities is not None:
            payload["probs"] = {
                label: round(float(self.probabilities[index]), 4)
                for index, label in enumerate(SHARED_LABELS)
            }

        return payload


# ============================================================
# THE POOL
# ============================================================


def _log_pool(
    contributions: Iterable[tuple[np.ndarray, float, np.ndarray]],
    temperature: float,
) -> np.ndarray:
    """Weighted geometric mean of distributions, renormalised.

    Each contribution is (probabilities, exponent, reliability).
    """

    accumulator = np.zeros(NUM_SHARED_LABELS, dtype=np.float64)

    for probabilities, exponent, reliability in contributions:
        if exponent <= 0:
            continue

        safe = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0)
        safe = safe / safe.sum()

        log_probabilities = np.log(safe)

        # Centre before applying per-class reliability, so an
        # uninformative modality contributes exactly zero whatever its
        # reliability profile says. See the module docstring.
        centred = log_probabilities - log_probabilities.mean()

        accumulator += exponent * reliability * centred

    accumulator /= max(temperature, 1e-6)

    accumulator -= accumulator.max()
    pooled = np.exp(accumulator)

    total = pooled.sum()

    if total <= 0 or not np.isfinite(total):
        # Unreachable given the epsilon floor, but a uniform fallback
        # beats emitting NaNs into the UI.
        return np.full(NUM_SHARED_LABELS, 1.0 / NUM_SHARED_LABELS)

    return pooled / total


class FusionEngine:
    """Combines the two modalities. Stateless; config is read once."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config if config is not None else FusionConfig.load()

    # --------------------------------------------------------

    def _face_contribution(
        self,
        probabilities: np.ndarray | None,
        quality: float,
    ) -> ModalityContribution:

        if probabilities is None:
            return ModalityContribution(present=False)

        quality = float(np.clip(quality, 0.0, 1.0))
        face_certainty = certainty(probabilities)

        exponent = self.config.face.base_weight * quality

        return ModalityContribution(
            present=True,
            exponent=exponent,
            factors={
                "base": self.config.face.base_weight,
                "quality": quality,
                "certainty": face_certainty,
            },
        )

    def _text_contribution(
        self,
        evidence: TextEvidence | None,
        now: float,
    ) -> ModalityContribution:

        if evidence is None:
            return ModalityContribution(present=False)

        age = evidence.age(now)

        text_recency = recency(
            age,
            self.config.text_half_life_seconds,
            self.config.text_cutoff_seconds,
        )

        text_informativeness = informativeness(
            evidence.token_count,
            self.config.min_informative_tokens,
        )

        text_certainty = certainty(evidence.probabilities)

        exponent = (
            self.config.text.base_weight * text_recency * text_informativeness
        )

        return ModalityContribution(
            present=True,
            exponent=exponent,
            factors={
                "base": self.config.text.base_weight,
                "recency": text_recency,
                "informativeness": text_informativeness,
                "certainty": text_certainty,
                "age_seconds": round(age, 2),
            },
        )

    # --------------------------------------------------------

    def fuse(
        self,
        face_probabilities: np.ndarray | None,
        face_quality: float,
        text_evidence: TextEvidence | None,
        now: float | None = None,
    ) -> FusionResult:
        """Combine whatever evidence is currently available."""

        now = time.monotonic() if now is None else now

        face = self._face_contribution(face_probabilities, face_quality)
        text = self._text_contribution(text_evidence, now)

        # An exponent below the floor is treated as absent rather than
        # as a tiny contribution: it cannot change any outcome, and
        # carrying it would keep the UI claiming a modality is in play.
        floor = self.config.min_effective_weight

        face_active = face.present and face.exponent >= floor
        text_active = text.present and text.exponent >= floor

        if not face_active and not text_active:
            return FusionResult(
                label=None,
                confidence=0.0,
                probabilities=None,
                face=face,
                text=text,
                agreement=0.0,
                conflicted=False,
                mode="none",
            )

        # Display influence: how much this modality actually moved the
        # result. Unlike the exponent this *does* include certainty,
        # because a present-but-uniform modality contributes nothing and
        # the bar should say so.
        face_influence = (
            face.exponent * face.factors.get("certainty", 0.0)
            if face_active
            else 0.0
        )
        text_influence = (
            text.exponent * text.factors.get("certainty", 0.0)
            if text_active
            else 0.0
        )

        influence_total = face_influence + text_influence

        if influence_total > 0:
            face.influence = face_influence / influence_total
            text.influence = text_influence / influence_total
        else:
            # Both present but both perfectly uniform. Split the bar
            # evenly among the active modalities rather than showing
            # zeros, which would read as "no modalities".
            active = int(face_active) + int(text_active)
            face.influence = (1.0 / active) if face_active else 0.0
            text.influence = (1.0 / active) if text_active else 0.0

        contributions: list[tuple[np.ndarray, float, np.ndarray]] = []

        if face_active:
            contributions.append(
                (
                    np.asarray(face_probabilities, dtype=np.float64),
                    face.exponent,
                    self.config.face.reliability,
                )
            )

        if text_active and text_evidence is not None:
            contributions.append(
                (
                    np.asarray(text_evidence.probabilities, dtype=np.float64),
                    text.exponent,
                    self.config.text.reliability,
                )
            )

        pooled = _log_pool(contributions, self.config.fusion_temperature)

        index = int(np.argmax(pooled))

        # Agreement only means something when there are two opinions to
        # compare. With one modality it is 1.0 - nothing disagrees -
        # rather than 0.0, which would read as total conflict.
        if face_active and text_active and text_evidence is not None:
            coefficient = agreement(
                face_probabilities, text_evidence.probabilities
            )
            mode = "fused"

            # Two conditions, because either alone misfires. A low
            # coefficient with the same argmax just means the two
            # modalities are differently confident about the same
            # emotion, which is not a conflict. Differing argmaxes with
            # a high coefficient means both are near-uniform and are
            # not really claiming anything, which is not a conflict
            # either - it is two shrugs.
            names_differ = int(
                np.argmax(face_probabilities)
            ) != int(np.argmax(text_evidence.probabilities))

            conflicted = (
                names_differ
                and coefficient < self.config.low_agreement_threshold
            )
        else:
            coefficient = 1.0
            conflicted = False
            mode = "face_only" if face_active else "text_only"

        return FusionResult(
            label=SHARED_LABELS[index],
            confidence=float(pooled[index]),
            probabilities=pooled,
            face=face,
            text=text,
            agreement=coefficient,
            conflicted=conflicted,
            mode=mode,
        )
