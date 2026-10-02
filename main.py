import torch
import torch.nn as nn
from audio_cnn import AudioCNN, predict_wav

# ==========================================
# 1. Other Modality Encoders (Placeholders)
# ==========================================
class TextEncoder(nn.Module):
    """
    Placeholder for a Text/Video encoder. 
    In a real scenario, this could be a pre-trained BERT mapping to 256 dims.
    """
    def __init__(self, embed_dim=256):
        super(TextEncoder, self).__init__()
        # Simulating a projection from BERT (768) to the shared 256-dim space
        self.embedding = nn.Linear(768, embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, 768)
        return self.embedding(x)

# ==========================================
# 2. Transformer Fusion Module
# ==========================================
class MultimodalTransformerFusion(nn.Module):
    """
    Fuses exactly 256-dim vectors from multiple modalities (Audio, Text, etc.)
    using Self-Attention, then classifies.
    """
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, num_classes=5):
        super(MultimodalTransformerFusion, self).__init__()
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio_features, text_features):
        """
        audio_features: (batch_size, embed_dim)
        text_features: (batch_size, embed_dim)
        """
        # Add sequence dimension and stack modalities: (batch_size, seq_len=2, embed_dim)
        audio_seq = audio_features.unsqueeze(1)
        text_seq = text_features.unsqueeze(1)
        
        # Shape: (batch_size, 2, 256)
        fused_seq = torch.cat((audio_seq, text_seq), dim=1)
        
        # Pass through Transformer to let modalities attend to each other
        transformer_out = self.transformer(fused_seq) # (batch_size, 2, 256)
        
        # Aggregate features (mean pooling across the sequence dimension)
        pooled_features = torch.mean(transformer_out, dim=1) # (batch_size, 256)
        
        # Final Classification
        logits = self.classifier(pooled_features)
        return logits

# ==========================================
# 3. End-to-End System Integration
# ==========================================
class MultimodalEmotionRecognitionSystem(nn.Module):
    """
    The master module that wraps the Audio Encoder, Text Encoder, and Fusion.
    """
    def __init__(self, num_classes=5):
        super(MultimodalEmotionRecognitionSystem, self).__init__()
        # Import the AudioCNN we just built
        self.audio_encoder = AudioCNN(num_classes=num_classes)
        
        # Init other encoders
        self.text_encoder = TextEncoder(embed_dim=256)
        
        # Init Fusion module
        self.fusion_module = MultimodalTransformerFusion(embed_dim=256, num_classes=num_classes)
        
    def forward(self, audio_input, text_input):
        # audio_input: (batch_size, 1, 40, 130)
        # text_input: (batch_size, 768)
        
        # 1. Independent Modality Encoding
        # CRITICAL: We bypass the audio_cnn's internal classifier and just grab the 256-dim feature!
        audio_features = self.audio_encoder.forward_encoder(audio_input) # (batch_size, 256)
        text_features = self.text_encoder(text_input)                    # (batch_size, 256)
        
        # 2. Transformer Fusion & Classification
        logits = self.fusion_module(audio_features, text_features)       # (batch_size, 5)
        return logits

def run_pipeline_demo():
    print("--- Initializing Multimodal End-to-End Pipeline ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Init end-to-end model
    model = MultimodalEmotionRecognitionSystem().to(device)
    model.eval()
    
    # Mock Batch Generation
    batch_size = 4
    # Mocking Audio: (batch, channels, mfcc, time)
    mock_audio = torch.randn(batch_size, 1, 40, 130).to(device) 
    # Mocking Text/Secondary Modality: (batch, BERT_dim)
    mock_text = torch.randn(batch_size, 768).to(device)         
    
    print(f"\n1. Executing Forward Pass (Batch Size = {batch_size})...")
    with torch.no_grad():
        logits = model(mock_audio, mock_text)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
    print("-> Pipeline Output Logits Shape:", logits.shape)
    print("-> Pipeline Output Probabilities:\n", probs)
    print("-> Predicted Emotion Classes:", predictions.tolist())
    print("\n--- Pipeline Successfully Connected! ---")

if __name__ == "__main__":
    run_pipeline_demo()
