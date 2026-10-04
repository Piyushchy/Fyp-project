import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import copy

# ==========================================
# 1. Dataset & DataLoaders
# ==========================================
class AudioDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path) # Expected shape: (N, 40, 130)
        self.y = np.load(y_path) # Expected shape: (N,)
        
        # Add channel dimension to make it (N, 1, 40, 130)
        if len(self.X.shape) == 3:
            self.X = np.expand_dims(self.X, axis=1)
            
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir='./processed', batch_size=32):
    train_dataset = AudioDataset(os.path.join(data_dir, 'X_train.npy'), os.path.join(data_dir, 'y_train.npy'))
    val_dataset = AudioDataset(os.path.join(data_dir, 'X_val.npy'), os.path.join(data_dir, 'y_val.npy'))
    test_dataset = AudioDataset(os.path.join(data_dir, 'X_test.npy'), os.path.join(data_dir, 'y_test.npy'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ==========================================
# 2. CNN Model (Encoder + Classifier)
# ==========================================
class AudioCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(AudioCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 32 x 20 x 65
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 64 x 10 x 32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 128 x 5 x 16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Output: 256 x 1 x 1
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
    def forward_encoder(self, x):
        """Returns the 256-dimensional feature vector."""
        features = self.encoder(x)
        features = features.view(features.size(0), -1) # Flatten to (batch_size, 256)
        return features
        
    def forward(self, x):
        """Returns the classification logits."""
        features = self.forward_encoder(x)
        logits = self.classifier(features)
        return logits

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10, save_path='best_audio_cnn.pth'):
    # Detect device (Supports CUDA, MPS on Mac, and CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_macro_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Macro-F1: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"--> Saved new best model with Val Macro-F1: {best_macro_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    print(f"Training complete. Best Val Macro-F1: {best_macro_f1:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 4. Evaluation Function
# ==========================================
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\n--- Test Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return acc, macro_f1, cm

# ==========================================
# 5. Inference Function
# ==========================================
EMOTION_CLASSES = {0: 'Happy', 1: 'Sad', 2: 'Angry', 3: 'Fear', 4: 'Surprise'}

def extract_mfcc(filepath, sr=22050, duration=3.0, n_mfcc=40, hop_length=512, max_len=130):
    # Load audio
    y, sr = librosa.load(filepath, sr=sr, duration=duration)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    
    # Pad or truncate to max_len
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
        
    # Normalize features per typical ML audio pipelines
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True) + 1e-8
    mfcc_normalized = (mfcc - mfcc_mean) / mfcc_std
    
    return mfcc_normalized

def predict_wav(model, filepath):
    """
    Replicates extraction, passes through model, and returns predicted emotion + raw 256-dim feature tensor.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Extract and preprocess features
    mfcc = extract_mfcc(filepath)
    
    # Convert to tensor: (1, 1, 40, 130) -> (Batch, Channels, Height, Width)
    input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get 256-dim feature vector
        features = model.forward_encoder(input_tensor)
        
        # Get classification prediction
        logits = model.classifier(features)
        _, pred = torch.max(logits, 1)
        
    emotion = EMOTION_CLASSES[pred.item()]
    
    return emotion, features.cpu().squeeze()

if __name__ == '__main__':
    print("Audio CNN initialized. Ready for multimodal emotion recognition!")
    
    # Example usage:
    # train_loader, val_loader, test_loader = get_dataloaders('./processed')
    # model = AudioCNN(num_classes=5)
    # trained_model = train_model(model, train_loader, val_loader)
    # evaluate_model(trained_model, test_loader)
    
    # emotion, feature_vector = predict_wav(trained_model, 'sample.wav')
    # print(f"Predicted: {emotion}, Vector Shape: {feature_vector.shape}")
