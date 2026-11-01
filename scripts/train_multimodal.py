# ==========================================================
# MULTIMODAL EMOTION RECOGNITION (OPTIMIZED FOR KAGGLE/GPU)
# ==========================================================

import os
import sys
import glob
import cv2
import torch
import librosa
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Add project root to sys.path so package-level imports work cleanly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from transformers import BertTokenizer, BertModel, ViTModel, ViTImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from models.fusion import (
    MultimodalTransformerFusion,
    MODALITY_VISION,
    MODALITY_AUDIO,
    MODALITY_TEXT,
    TARGET_EMOTIONS,
)

# ==========================================================
# CONFIG & DEVICE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4

VALID_EMOTIONS = TARGET_EMOTIONS
EMOTIONS = {e: i for i, e in enumerate(VALID_EMOTIONS)}
NUM_CLASSES = len(VALID_EMOTIONS)

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================================
# INTELLIGENT PATH RESOLUTION (Kaggle & Local)
# ==========================================================
def find_dataset_dir(target_name, default_local):
    """
    Auto-detects dataset paths whether running on Kaggle (/kaggle/input)
    or locally on Windows/Linux.
    """
    # 1. Environment variable override
    env_key = f"{target_name.upper()}_PATH"
    if env_key in os.environ and os.path.exists(os.environ[env_key]):
        return os.environ[env_key]

    # 2. Check local default first
    local_path = _PROJECT_ROOT / default_local
    if local_path.exists():
        return str(local_path)

    # 3. Check relative to current working dir
    if os.path.exists(default_local):
        return default_local

    # 4. Search in /kaggle/input if present
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Match by folder containing the emotions
        for root, dirs, _ in os.walk(str(kaggle_input)):
            dir_names = set(dirs)
            # Check if directory contains at least 3 valid emotion subfolders
            if len(dir_names.intersection(set(VALID_EMOTIONS))) >= 3:
                r_lower = root.lower()
                if target_name == "image" and any(k in r_lower for k in ["fer", "image", "face", "img"]):
                    return root
                if target_name == "audio" and any(k in r_lower for k in ["ravdess", "audio", "sound", "wav"]):
                    return root
                if target_name == "text" and any(k in r_lower for k in ["text", "transcript", "txt"]):
                    return root

        # Fallback search by name substring
        for root, dirs, _ in os.walk(str(kaggle_input)):
            for d in dirs:
                if target_name == "image" and ("fer2013" in d.lower() or "images" in d.lower()):
                    candidate = os.path.join(root, d)
                    if any(os.path.exists(os.path.join(candidate, e)) for e in VALID_EMOTIONS):
                        return candidate
                if target_name == "audio" and ("ravdess" in d.lower() or "audio" in d.lower()):
                    candidate = os.path.join(root, d)
                    if any(os.path.exists(os.path.join(candidate, e)) for e in VALID_EMOTIONS):
                        return candidate
                if target_name == "text" and ("text" in d.lower()):
                    candidate = os.path.join(root, d)
                    if any(os.path.exists(os.path.join(candidate, e)) for e in VALID_EMOTIONS):
                        return candidate

    return default_local


FER_PATH = find_dataset_dir("image", "data/fer2013/images")
RAVDESS_PATH = find_dataset_dir("audio", "data/ravdess/audio")
TEXT_PATH = find_dataset_dir("text", "data/text")

print("\n--- RESOLVED DATASET PATHS ---")
print(f"FER Image Path : {FER_PATH} (exists: {os.path.exists(FER_PATH)})")
print(f"Audio Path     : {RAVDESS_PATH} (exists: {os.path.exists(RAVDESS_PATH)})")
print(f"Text Path      : {TEXT_PATH} (exists: {os.path.exists(TEXT_PATH)})")

# Determine output & cache directories
IS_KAGGLE = os.path.exists("/kaggle")
OUTPUT_DIR = Path("/kaggle/working") if IS_KAGGLE else _PROJECT_ROOT
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# AUDIO EXTRACTION HELPER
# ==========================================================
def load_audio_mfcc(path, max_len=200):
    try:
        signal, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T
        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)))
        else:
            mfcc = mfcc[:max_len]
        return mfcc.astype(np.float32)
    except Exception as e:
        return np.zeros((max_len, 40), dtype=np.float32)


# ==========================================================
# LOAD MULTIMODAL DATA FRAME
# ==========================================================
def load_multimodal_data():
    texts, audios, images, labels = [], [], [], []

    for emo in VALID_EMOTIONS:
        text_dir = os.path.join(TEXT_PATH, emo)
        audio_dir = os.path.join(RAVDESS_PATH, emo)
        image_dir = os.path.join(FER_PATH, emo)

        text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))
        audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")) +
                             glob.glob(os.path.join(audio_dir, "*.mp3")))
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                             glob.glob(os.path.join(image_dir, "*.png")))

        if len(text_files) == 0 or len(audio_files) == 0 or len(image_files) == 0:
            print(f" Skipping {emo} (Missing files: T:{len(text_files)}, A:{len(audio_files)}, I:{len(image_files)})")
            continue

        max_len = max(len(text_files), len(audio_files), len(image_files))

        for i in range(max_len):
            txt_path = text_files[i % len(text_files)]
            aud_path = audio_files[i % len(audio_files)]
            img_path = image_files[i % len(image_files)]

            try:
                with open(txt_path, encoding="utf-8", errors="ignore") as f:
                    txt = f.read().strip()
            except Exception:
                txt = emo

            if not txt:
                txt = emo

            texts.append(txt)
            audios.append(aud_path)
            images.append(img_path)
            labels.append(emo)

    return pd.DataFrame({
        "text": texts,
        "audio": audios,
        "image": images,
        "emotion": labels
    })


data_df = load_multimodal_data()
if len(data_df) == 0:
    print("\nERROR: No data found. Please verify the dataset paths above.")
    sys.exit(1)

print("\nFULL DATASET DISTRIBUTION:")
print(data_df["emotion"].value_counts())

# Handle small classes
min_samples = 5
balanced_rows = []
for emo in VALID_EMOTIONS:
    subset = data_df[data_df["emotion"] == emo]
    if len(subset) == 0:
        continue
    if len(subset) < min_samples:
        extra = subset.sample(min_samples - len(subset), replace=True, random_state=42)
        subset = pd.concat([subset, extra])
    balanced_rows.append(subset)

data_df = pd.concat(balanced_rows).reset_index(drop=True)

# Stratified Split
train_df, temp_df = train_test_split(
    data_df, test_size=0.3, stratify=data_df["emotion"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=42
)

print(f"\nSPLITS: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


# ==========================================================
# OPTIMIZATION: BATCH FEATURE EXTRACTION & CACHING
# ==========================================================
class FastImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        # Direct PyTorch tensor normalized to [-1, 1] as expected by ViT
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (tensor - 0.5) / 0.5


def extract_features_for_split(split_name, df, cache_path):
    """
    Extracts Text (BERT), Audio (MFCC), and Image (ViT) in high-throughput
    GPU batches ONCE, then caches to disk.
    Subsequent runs take < 1 second!
    """
    if os.path.exists(cache_path):
        print(f"Loading precomputed features for {split_name} from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu")
        return cached["text"], cached["audio"], cached["image"], cached["labels"]

    print(f"\n==========================================================")
    print(f"EXTRACTING FEATURES FOR {split_name.upper()} ({len(df)} samples)")
    print(f"==========================================================")

    texts = df["text"].tolist()
    audio_paths = df["audio"].tolist()
    image_paths = df["image"].tolist()
    labels = torch.tensor([EMOTIONS[e] for e in df["emotion"]], dtype=torch.long)

    # 1. TEXT (BERT batch encoding on GPU)
    print("1/3 -> Batch encoding Text with BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval()

    # Encode unique texts to eliminate duplicate computation
    unique_texts = list(set(texts))
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    unique_feats = []
    text_batch_size = 128 if DEVICE == "cuda" else 32

    with torch.no_grad():
        for i in tqdm(range(0, len(unique_texts), text_batch_size), desc="BERT Batches"):
            batch = unique_texts[i : i + text_batch_size]
            tokens = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(DEVICE)
            out = bert(**tokens).last_hidden_state[:, 0, :].cpu()
            unique_feats.append(out)

    unique_feats = torch.cat(unique_feats, dim=0)
    full_text_indices = [text_to_idx[t] for t in texts]
    text_feats = unique_feats[full_text_indices]  # (N, 768)

    del bert, tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # 2. AUDIO (MFCC extraction with dictionary cache)
    print("2/3 -> Extracting Audio MFCCs...")
    unique_audio = list(set(audio_paths))
    audio_to_idx = {p: i for i, p in enumerate(unique_audio)}
    unique_mfccs = []

    for p in tqdm(unique_audio, desc="Audio MFCCs"):
        unique_mfccs.append(load_audio_mfcc(p))

    unique_mfccs = np.stack(unique_mfccs, axis=0)  # (U, 200, 40)
    full_audio_indices = [audio_to_idx[p] for p in audio_paths]
    audio_feats = torch.tensor(unique_mfccs[full_audio_indices], dtype=torch.float32)

    # 3. IMAGE (ViT batch encoding on GPU)
    print("3/3 -> Batch encoding Images with ViT...")
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
    vit.eval()

    img_ds = FastImageDataset(image_paths)
    img_loader = DataLoader(
        img_ds,
        batch_size=64 if DEVICE == "cuda" else 16,
        shuffle=False,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=(DEVICE == "cuda"),
    )

    img_feats = []
    with torch.no_grad():
        for batch_imgs in tqdm(img_loader, desc="ViT Batches"):
            batch_imgs = batch_imgs.to(DEVICE)
            out = vit(pixel_values=batch_imgs).last_hidden_state[:, 0, :].cpu()
            img_feats.append(out)

    img_feats = torch.cat(img_feats, dim=0)  # (N, 768)

    del vit
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Save cache
    print(f"Caching {split_name} features to {cache_path}...")
    torch.save(
        {
            "text": text_feats,
            "audio": audio_feats,
            "image": img_feats,
            "labels": labels,
        },
        cache_path,
    )

    return text_feats, audio_feats, img_feats, labels


# Extract or load features for all splits
t_train, a_train, v_train, y_train = extract_features_for_split(
    "train", train_df, CACHE_DIR / "features_train.pt"
)
t_val, a_val, v_val, y_val = extract_features_for_split(
    "val", val_df, CACHE_DIR / "features_val.pt"
)
t_test, a_test, v_test, y_test = extract_features_for_split(
    "test", test_df, CACHE_DIR / "features_test.pt"
)

train_ds = TensorDataset(t_train, a_train, v_train, y_train)
val_ds = TensorDataset(t_val, a_val, v_val, y_val)
test_ds = TensorDataset(t_test, a_test, v_test, y_test)

# Class weights & sampler
labels_list = y_train.tolist()
counts = Counter(labels_list)
total = sum(counts.values())
weights = [total / (NUM_CLASSES * (counts[i] + 1e-6)) for i in range(NUM_CLASSES)]
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

sample_weights = [weights[l] for l in labels_list]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================================
# MODEL DEFINITION
# ==========================================================
class MultimodalTrainingModel(nn.Module):
    """
    Multimodal network combining Text, Audio CNN, and Vision feature projectors
    with the standardized MultimodalTransformerFusion module.
    """
    def __init__(self):
        super().__init__()
        # TEXT
        self.t_fc = nn.Linear(768, 256)

        # AUDIO CNN
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.a_fc = nn.Linear(128, 256)

        # IMAGE (ViT [CLS] -> 256)
        self.v_fc = nn.Linear(768, 256)

        # STANDARDIZED TRANSFORMER FUSION WITH MODALITY EMBEDDINGS
        self.fusion = MultimodalTransformerFusion(embed_dim=256, num_classes=NUM_CLASSES)

    def forward(self, t, a, v):
        # TEXT: (B, 768) -> (B, 256)
        t = torch.relu(self.t_fc(t))

        # AUDIO CNN FORWARD: (B, 200, 40) -> (B, 1, 40, 200) -> (B, 256)
        a = a.permute(0, 2, 1).unsqueeze(1)
        a = self.audio_cnn(a)
        a = a.view(a.size(0), -1)
        a = torch.relu(self.a_fc(a))

        # IMAGE: (B, 768) -> (B, 256)
        v = torch.relu(self.v_fc(v))

        # FUSION WITH MODALITY EMBEDDINGS: [Vision=0, Audio=1, Text=2]
        tokens = torch.stack([v, a, t], dim=1)  # (B, 3, 256)
        mod_ids = torch.tensor(
            [MODALITY_VISION, MODALITY_AUDIO, MODALITY_TEXT],
            dtype=torch.long,
            device=tokens.device,
        )
        return self.fusion(tokens, mod_ids)


model = MultimodalTrainingModel().to(DEVICE)


# ==========================================================
# LOSS & OPTIMIZER
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


criterion = FocalLoss(alpha=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================================
# HIGH-SPEED TRAINING LOOP
# ==========================================================
print(f"\nStarting training for {EPOCHS} epochs (Early stopping patience: 5)...")
train_accs, val_accs = [], []
train_losses, val_losses = [], []
best_val = float("inf")
patience, counter = 5, 0

# Checkpoint paths
SAVE_FULL = OUTPUT_DIR / "data_Finalmodel.pt"
SAVE_FUSION = OUTPUT_DIR / "best_fusion.pth"

for epoch in range(EPOCHS):
    model.train()
    preds, labels = [], []
    train_loss = 0.0

    for t, a, v, y in train_loader:
        t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(t, a, v)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds += out.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()

    train_loss /= len(train_loader)
    train_acc = accuracy_score(labels, preds)

    model.eval()
    preds, labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for t, a, v, y in val_loader:
            t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
            out = model(t, a, v)
            loss = criterion(out, y)

            val_loss += loss.item()
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()

    val_loss /= len(val_loader)
    val_acc = accuracy_score(labels, preds)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        # 1. Save entire multimodal pipeline
        torch.save(model.state_dict(), str(SAVE_FULL))
        # 2. Save standalone fusion module for direct deployment in backend/app.py
        torch.save(model.fusion.state_dict(), str(SAVE_FUSION))
        print(f"  --> Saved new best checkpoint to {SAVE_FUSION.name}")
    else:
        counter += 1
        if counter >= patience:
            print("\nEarly stopping triggered!")
            break


# ==========================================================
# FINAL EVALUATION
# ==========================================================
def evaluate(loader, name):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for t, a, v, y in loader:
            t, a, v = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE)
            out = model(t, a, v)
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()

    acc = accuracy_score(labels, preds)
    print(f"\n==========================================================")
    print(f"{name.upper()} ACCURACY: {acc:.4f}")
    print(f"==========================================================")
    print(classification_report(labels, preds, target_names=VALID_EMOTIONS))


# Load the best saved weights
if SAVE_FULL.exists():
    model.load_state_dict(torch.load(str(SAVE_FULL), map_location=DEVICE))

evaluate(train_loader, "Train")
evaluate(val_loader, "Validation")
evaluate(test_loader, "Test")

print("\n==========================================================")
print(f"TRAINING COMPLETE!")
print(f"Deployed Checkpoint: {SAVE_FUSION.resolve()}")
print(f"Full Model Checkpoint: {SAVE_FULL.resolve()}")
print("==========================================================")
