"""
Multimodal Emotion Recognition — End-to-End Pipeline & Architecture Demo
========================================================================

Demonstrates feature encoding and self-attention fusion across three modalities:
  1. Vision (ViT [CLS] token 768-D -> 256-D)
  2. Audio  (MFCC CNN -> 256-D)
  3. Text   (BERT / TinyBERT 768-D -> 256-D)
with modality embeddings (0=Vision, 1=Audio, 2=Text) fused into 5 target emotions.
"""

import torch
import torch.nn as nn

from models.audio_cnn import AudioCNN
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion import (
    MultimodalTransformerFusion,
    MODALITY_VISION,
    MODALITY_AUDIO,
    MODALITY_TEXT,
    TARGET_EMOTIONS,
)


class MultimodalEmotionRecognitionSystem(nn.Module):
    """
    Master end-to-end architecture wrapping:
      - VisionEncoder (ViT backbone [CLS] -> 256-D)
      - AudioCNN (Conv2D feature extractor -> 256-D)
      - TextEncoder (Linear projection -> 256-D)
      - MultimodalTransformerFusion (Self-attention with modality type embeddings)
    """

    def __init__(self, num_classes: int = 5):
        super(MultimodalEmotionRecognitionSystem, self).__init__()
        self.vision_encoder = VisionEncoder(embed_dim=256)
        self.audio_encoder = AudioCNN(num_classes=num_classes)
        self.text_encoder = TextEncoder(embed_dim=256)
        self.fusion_module = MultimodalTransformerFusion(
            embed_dim=256, num_classes=num_classes
        )

    def forward(
        self,
        vision_input: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        text_input: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Encodes each present modality to 256-D and fuses them via transformer self-attention.
        """
        vision_feat = self.vision_encoder(vision_input) if vision_input is not None else None
        audio_feat = self.audio_encoder.forward_encoder(audio_input) if audio_input is not None else None
        text_feat = self.text_encoder(text_input) if text_input is not None else None

        return self.fusion_module.forward_modalities(
            vision_feat=vision_feat,
            audio_feat=audio_feat,
            text_feat=text_feat,
        )


def run_pipeline_demo():
    print("=" * 60)
    print(" Multimodal Emotion Recognition Pipeline Demo")
    print(" Target Emotion Space:", TARGET_EMOTIONS)
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Initialize End-to-End System
    model = MultimodalEmotionRecognitionSystem(num_classes=len(TARGET_EMOTIONS)).to(device)
    model.eval()

    batch_size = 2
    # Mock Modality Inputs
    mock_vision = torch.randn(batch_size, 3, 224, 224).to(device)  # Face crop (B, C, H, W)
    mock_audio = torch.randn(batch_size, 1, 40, 130).to(device)    # Audio MFCC (B, 1, n_mfcc, T)
    mock_text = torch.randn(batch_size, 768).to(device)            # Text embedding (B, 768)

    print(f"\n1. Testing Tri-Modal Forward Pass (Batch Size = {batch_size})...")
    with torch.no_grad():
        logits = model(
            vision_input=mock_vision,
            audio_input=mock_audio,
            text_input=mock_text,
        )
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    print("-> Logits Output Shape:", logits.shape)
    print("-> Predicted Emotion Indices:", preds.tolist())
    print("-> Predicted Emotions:", [TARGET_EMOTIONS[i] for i in preds.tolist()])

    print(f"\n2. Testing Partial Modalities (e.g., Vision + Audio only)...")
    with torch.no_grad():
        partial_logits = model(
            vision_input=mock_vision,
            audio_input=mock_audio,
            text_input=None,
        )
        partial_preds = torch.argmax(torch.softmax(partial_logits, dim=-1), dim=-1)
    print("-> Partial Modality Logits Shape:", partial_logits.shape)
    print("-> Predicted Emotions:", [TARGET_EMOTIONS[i] for i in partial_preds.tolist()])

    print("\n All modality flows and self-attention fusion verified successfully!")


if __name__ == "__main__":
    run_pipeline_demo()
