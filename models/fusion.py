from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modality Type Identifiers
MODALITY_VISION = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2

MODALITY_ID_MAP: Dict[str, int] = {
    "vision": MODALITY_VISION,
    "audio": MODALITY_AUDIO,
    "text": MODALITY_TEXT,
}

# Unified 5-Class Target Emotion Space
TARGET_EMOTIONS: List[str] = ["happy", "sad", "angry", "fear", "surprise"]
NUM_CLASSES: int = len(TARGET_EMOTIONS)


class MultimodalTransformerFusion(nn.Module):
    """
    Multimodal Self-Attention Fusion Module.

    Takes 256-dimensional feature representations from up to 3 modalities (Vision, Audio, Text),
    adds modality-type embeddings (nn.Embedding(3, 256)) to distinguish modality identity,
    passes the sequence through a TransformerEncoder, and classifies into 5 target emotions:
    ['happy', 'sad', 'angry', 'fear', 'surprise'].
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 5,
        num_modalities: int = 3,
        dropout: float = 0.3,
    ):
        super(MultimodalTransformerFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Modality Identification Embedding Table (0=Vision, 1=Audio, 2=Text)
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head (256 -> 128 -> ReLU -> Dropout -> 5)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard batched forward pass with explicit modality tokens and IDs.

        Args:
            features: Tensor of shape (batch_size, seq_len, embed_dim)
            modality_ids: LongTensor of shape (batch_size, seq_len) or (seq_len,)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        if modality_ids.dim() == 1:
            modality_ids = modality_ids.unsqueeze(0).expand(features.size(0), -1)

        # Retrieve and inject modality embeddings
        mod_embs = self.modality_embed(modality_ids)  # (batch_size, seq_len, embed_dim)
        x = features + mod_embs

        # Transformer self-attention across modality tokens
        transformer_out = self.transformer(x)  # (batch_size, seq_len, embed_dim)

        # Mean pooling across available modality sequence tokens
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, embed_dim)

        # Classification head
        logits = self.classifier(pooled)
        return logits

    def forward_modalities(
        self,
        vision_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        text_feat: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Flexible convenience forward pass for 1, 2, or 3 modalities.
        Accepts individual 256-D feature tensors of shape (batch_size, 256) or (256,).

        Returns:
            Logits tensor of shape (batch_size, num_classes), or None if no modalities provided.
        """
        tokens: List[torch.Tensor] = []
        ids: List[int] = []
        device = next(self.parameters()).device

        def format_vector(v: torch.Tensor) -> torch.Tensor:
            if v.dim() == 1:
                return v.unsqueeze(0)  # (1, 256)
            return v

        if vision_feat is not None:
            tokens.append(format_vector(vision_feat).unsqueeze(1))  # (B, 1, 256)
            ids.append(MODALITY_VISION)

        if audio_feat is not None:
            tokens.append(format_vector(audio_feat).unsqueeze(1))  # (B, 1, 256)
            ids.append(MODALITY_AUDIO)

        if text_feat is not None:
            tokens.append(format_vector(text_feat).unsqueeze(1))  # (B, 1, 256)
            ids.append(MODALITY_TEXT)

        if not tokens:
            return None

        # Concatenate tokens along sequence dimension: (batch_size, num_modalities, 256)
        seq = torch.cat(tokens, dim=1).to(device)
        mod_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).expand(seq.size(0), -1)

        return self.forward(seq, mod_ids)
