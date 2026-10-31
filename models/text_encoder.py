import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Text Encoder that projects 768-dimensional token representations
    (from BERT or TinyBERT) into the shared 256-dimensional multimodal embedding space.
    """

    def __init__(self, in_features: int = 768, embed_dim: int = 256):
        super(TextEncoder, self).__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, 768) or (768,)
        Returns:
            Tensor of shape (batch_size, 256) or (256,)
        """
        return self.embedding(x)

