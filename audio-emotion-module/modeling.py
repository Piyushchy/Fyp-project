# ============================================================
# AUDIO EMOTION MODULE - MODELING
# ============================================================
#
# Two classifiers over the shared seven, both exposing the same
# two-method interface the multimodal server already expects of an
# audio encoder:
#
#   forward_encoder(x) -> [batch, EMBED_DIM]   fusion features
#   forward(x)         -> [batch, 7]           shared-space logits
#
# EMBED_DIM is 256 to match MultimodalTransformerFusion in
# ../main.py, which projects every modality to 256 before the
# attention stack. Keeping the width means the trained encoder
# drops into that module without a shim.
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODELS, NUM_LABELS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


EMBED_DIM = 256


# ============================================================
# WEIGHTED LAYER SUM
# ============================================================

class WeightedLayerSum(nn.Module):
    """Learned softmax mixture over an encoder's hidden states.

    A masked-prediction encoder's top layers drift towards the
    pretext task - predicting the masked acoustic unit, which is
    close to phonetic identity - while pitch, energy and voice
    quality peak in the middle of the stack. Reading only the last
    hidden state discards the layers that carry the emotion.

    Initialised uniform (all logits zero) rather than biased
    towards any depth, so the learned weights are evidence about
    where emotion lives in this encoder rather than an assumption
    about it. train_audio.py prints them.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Guard, not defensive padding. If the encoder hands back a
        # different number of layers than this mixture was built for,
        # the only safe move is to stop: weight i would be applied to
        # whatever layer happens to sit at index i, which is a silent
        # corruption of the learned mixture rather than an error.
        # LayerDrop causes exactly this, which is why it is disabled -
        # see LAYERDROP in config.py.
        if len(hidden_states) != self.weights.shape[0]:
            raise RuntimeError(
                f"encoder returned {len(hidden_states)} hidden states but "
                f"the layer mixture has {self.weights.shape[0]} weights. "
                f"LayerDrop must be 0 for a weighted layer sum."
            )

        stacked = torch.stack(hidden_states, dim=0)
        mixture = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)

        return (stacked * mixture).sum(dim=0)

    def distribution(self) -> list[float]:
        with torch.no_grad():
            return F.softmax(self.weights, dim=0).cpu().tolist()


# ============================================================
# ATTENTIVE STATISTICS POOLING
# ============================================================

class AttentiveStatsPooling(nn.Module):
    """Collapse [batch, time, dim] to [batch, 2 * dim].

    A clip carries one label across ~200 frames, so the frames have
    to be collapsed to a vector. Mean pooling gives the room tone
    at the head of a studio clip the same vote as the stressed
    vowel that carries the anger.

    The weighted standard deviation is concatenated to the weighted
    mean because how much the prosody MOVES across a clip is itself
    the signal: a flat read and an agitated one can share a mean
    pitch and energy and differ entirely in their spread.
    """

    def __init__(self, dim: int, bottleneck: int = 128) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(dim, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, dim, kernel_size=1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Conv1d wants [batch, dim, time].
        transposed = frames.transpose(1, 2)

        weights = F.softmax(self.attention(transposed), dim=2)

        mean = torch.sum(transposed * weights, dim=2)

        # E[x^2] - E[x]^2, clamped before the root. The two terms are
        # close for a near-constant channel and float error can put the
        # difference slightly below zero, which would produce NaN
        # gradients a few hundred steps into training.
        variance = torch.sum(transposed.pow(2) * weights, dim=2) - mean.pow(2)
        deviation = torch.sqrt(variance.clamp(min=1e-7))

        return torch.cat((mean, deviation), dim=1)


# ============================================================
# SSL CLASSIFIER
# ============================================================

class SslEmotionClassifier(nn.Module):
    """A pretrained speech encoder with an emotion head on top."""

    kind = "ssl"

    def __init__(
        self,
        hub_id: str,
        num_classes: int = NUM_LABELS,
        dropout: float = 0.2,
        mask_time_prob: float = 0.05,
        layerdrop: float = 0.05,
    ) -> None:
        super().__init__()

        from transformers import AutoConfig, AutoModel

        configuration = AutoConfig.from_pretrained(hub_id)
        configuration.output_hidden_states = True

        # SpecAugment-style masking inside the encoder. Cheaper and
        # better targeted than masking the waveform: it hides frames
        # the transformer has already contextualised, so the model has
        # to infer them from the rest of the utterance.
        configuration.apply_spec_augment = True
        configuration.mask_time_prob = mask_time_prob
        configuration.layerdrop = layerdrop

        self.backbone = AutoModel.from_pretrained(hub_id, config=configuration)

        # Frozen: the convolutional front end maps raw samples to 50 Hz
        # frames and is task-agnostic. Fine-tuning it through the whole
        # transformer destabilises training for no measured gain.
        self.backbone.freeze_feature_encoder()

        hidden_size = configuration.hidden_size
        num_layers = configuration.num_hidden_layers + 1  # + the embeddings

        self.layer_sum = WeightedLayerSum(num_layers)
        self.pooling = AttentiveStatsPooling(hidden_size)

        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_size, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(EMBED_DIM, num_classes)

    # -- interface the fusion module uses --

    def forward_encoder(self, waveform: torch.Tensor) -> torch.Tensor:
        """256-dim fusion feature for a batch of raw clips."""

        outputs = self.backbone(waveform)
        frames = self.layer_sum(outputs.hidden_states)

        return self.projection(self.pooling(frames))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Shared-space logits for a batch of raw clips."""

        return self.classifier(self.forward_encoder(waveform))

    # -- parameter groups --

    def backbone_parameters(self):
        return [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        ]

    def head_parameters(self):
        modules = (self.layer_sum, self.pooling, self.projection, self.classifier)

        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]


# ============================================================
# MFCC CNN BASELINE
# ============================================================

class MfccEmotionClassifier(nn.Module):
    """The multimodal branch's AudioCNN over the shared seven.

    Imported from ../audio_cnn.py rather than reimplemented, so the
    baseline is genuinely that branch's architecture and the
    comparison in results/benchmark.json is a comparison and not a
    redesign. The only change is num_classes: 5 -> 7, which the
    constructor already parameterised.
    """

    kind = "mfcc-cnn"

    def __init__(self, num_classes: int = NUM_LABELS) -> None:
        super().__init__()

        from audio_cnn import AudioCNN

        self.inner = AudioCNN(num_classes=num_classes)

    def forward_encoder(self, features: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.inner(features)

    def backbone_parameters(self):
        return []

    def head_parameters(self):
        return list(self.inner.parameters())


# ============================================================
# FACTORY
# ============================================================

def build(model_key: str) -> nn.Module:
    """Instantiate one entry of the MODELS registry."""

    if model_key not in MODELS:
        raise ValueError(
            f"unknown model {model_key!r}; known: {sorted(MODELS)}"
        )

    spec = MODELS[model_key]

    if spec["kind"] == "ssl":
        return SslEmotionClassifier(
            spec["hub_id"],
            mask_time_prob=spec.get("mask_time_prob", 0.05),
            layerdrop=spec.get("layerdrop", 0.05),
        )

    if spec["kind"] == "mfcc-cnn":
        return MfccEmotionClassifier()

    raise ValueError(f"unknown model kind {spec['kind']!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
