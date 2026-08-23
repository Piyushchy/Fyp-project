# ==========================================================
# MULTIMODAL EMOTION RECOGNITION - CONFIGURABLE MODEL SELECTION
# ==========================================================
#
# Run examples (Member 4's three assigned combinations):
#   python main.py --text distilbert --audio cnn    --image swin
#   python main.py --text distilbert --audio hubert --image vit
#   python main.py --text bert       --audio hubert --image swin
#
# Baseline (Member 1 / original checkpoint, for sanity-checking the refactor):
#   python main.py --text bert --audio cnn --image vit
#
# Currently wired encoders:
#   --text  : bert | distilbert
#   --audio : cnn  | hubert
#   --image : vit  | swin
#
# Everything else from the experiment matrix (roberta, deberta, wav2vec2,
# ast, convnext, resnet50) is intentionally NOT implemented yet — the
# factory functions below raise NotImplementedError with a clear TODO
# marker so this script fails loudly instead of silently doing the
# wrong thing if someone requests an unsupported combination.
# ==========================================================

import argparse
import csv
import os
import glob
import cv2
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    BertTokenizer, BertModel,
    DistilBertTokenizer, DistilBertModel,
    AutoProcessor, HubertModel,
    ViTImageProcessor, ViTModel,
    AutoImageProcessor, SwinModel,
)

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# ARGPARSE
# ==========================================================
parser = argparse.ArgumentParser(description="Configurable multimodal emotion recognition")
parser.add_argument("--text", choices=["bert", "distilbert"], required=True)
parser.add_argument("--audio", choices=["cnn", "hubert"], required=True)
parser.add_argument("--image", choices=["vit", "swin"], required=True)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=None,
                     help="Defaults to 32 for the cnn/vit baseline, 8 for pretrained-heavy combos")
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()


# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = args.epochs
LR = args.lr

# Heavier pretrained combos (hubert / swin) need a smaller batch size than
# the lightweight cnn+vit baseline, purely for GPU memory reasons.
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
else:
    BATCH_SIZE = 8 if (args.audio == "hubert" or args.image == "swin") else 32

AUDIO_LENGTH = 64000  # 4 seconds @ 16kHz, only used by waveform-based audio models

# ----------------------------------------------------------
# DATA PATHS
# TODO: replace <DATASET-NAME> once the Kaggle dataset slug is confirmed.
# Kaggle mounts uploaded datasets read-only at /kaggle/input/<slug>/
# ----------------------------------------------------------
DATA_ROOT = "/kaggle/input/<DATASET-NAME>/data"
FER_PATH = os.path.join(DATA_ROOT, "fer2013/images")
RAVDESS_PATH = os.path.join(DATA_ROOT, "ravdess/audio")
TEXT_PATH = os.path.join(DATA_ROOT, "text")

# Writable output locations (Kaggle working dir)
OUTPUT_ROOT = "/kaggle/working"
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_NAME = f"{args.text}_{args.audio}_{args.image}"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}.pt")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

VALID_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise"]
EMOTIONS = {e: i for i, e in enumerate(VALID_EMOTIONS)}
NUM_CLASSES = len(VALID_EMOTIONS)

print("\n==========================================")
print("Text Model  :", args.text)
print("Audio Model :", args.audio)
print("Image Model :", args.image)
print("Classes     :", VALID_EMOTIONS)
print("Device      :", DEVICE)
print("Batch size  :", BATCH_SIZE)
print("Checkpoint  :", CHECKPOINT_PATH)
print("==========================================")


# ==========================================================
# TEXT ENCODER FACTORY
# All text models here are frozen (no gradients) and pooled via the
# first ([CLS]-equivalent) token of the last hidden state.
# ==========================================================
def get_text_model(name):
    if name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
        output_dim = 768
    elif name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
        output_dim = 768
    elif name in ("roberta", "deberta"):
        raise NotImplementedError(f"TODO: text encoder '{name}' not implemented yet")
    else:
        raise ValueError(f"Unknown text model: {name}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def encode_text(text):
        tokens = tokenizer(text, padding=True, truncation=True, max_length=128,
                            return_tensors="pt")
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
            features = out.last_hidden_state[:, 0, :]
        return features.squeeze(0)

    return encode_text, output_dim


# ==========================================================
# AUDIO ENCODER FACTORY
# "cnn"    -> raw MFCC features fed into a small trainable CNN2D
#             (kept INSIDE the Model class, since it's trainable, not
#             a fixed feature extractor).
# "hubert" -> frozen pretrained encoder on the raw waveform, mean-pooled
#             over time.
# Returns: kind ("cnn" | "pretrained"), a loader fn (path -> raw array),
#          an encode fn (raw array -> tensor, or None for "cnn" since
#          the raw MFCC IS the model input), and output_dim.
# ==========================================================
def get_audio_model(name):
    if name == "cnn":
        def load_audio(path, max_len=200):
            signal, sr = librosa.load(path, sr=16000)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T
            if len(mfcc) < max_len:
                mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)))
            else:
                mfcc = mfcc[:max_len]
            return mfcc.astype(np.float32)

        return "cnn", load_audio, None, 128  # 128 = CNN's own output dim before a_fc

    elif name == "hubert":
        processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        def load_audio(path, max_len=None):
            signal, sr = librosa.load(path, sr=16000, mono=True)
            if np.max(np.abs(signal)) > 0:
                signal = signal / np.max(np.abs(signal))
            if len(signal) < AUDIO_LENGTH:
                signal = np.pad(signal, (0, AUDIO_LENGTH - len(signal)))
            else:
                signal = signal[:AUDIO_LENGTH]
            return signal.astype(np.float32)

        def encode_audio(signal):
            inputs = processor(signal, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                features = out.last_hidden_state.mean(dim=1)
            return features.squeeze(0)

        return "pretrained", load_audio, encode_audio, 768

    elif name in ("wav2vec2", "ast"):
        raise NotImplementedError(f"TODO: audio encoder '{name}' not implemented yet")
    else:
        raise ValueError(f"Unknown audio model: {name}")


# ==========================================================
# IMAGE ENCODER FACTORY
# ViT is pooled via its CLS token; Swin has no CLS token so its patch
# outputs are mean-pooled instead. Frozen either way.
# ==========================================================
def get_image_model(name):
    if name == "vit":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
        output_dim = 768
        pool = "cls"
    elif name == "swin":
        processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224").to(DEVICE)
        output_dim = 1024
        pool = "mean"
    elif name in ("convnext", "resnet50"):
        raise NotImplementedError(f"TODO: image encoder '{name}' not implemented yet")
    else:
        raise ValueError(f"Unknown image model: {name}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def load_image(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Unable to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (224, 224))

    def encode_image(image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            if pool == "cls":
                features = out.last_hidden_state[:, 0, :]
            else:
                features = out.last_hidden_state.mean(dim=1)
        return features.squeeze(0)

    return load_image, encode_image, output_dim


# ==========================================================
# BUILD SELECTED ENCODERS
# ==========================================================
print("\nLoading text encoder:", args.text)
encode_text, text_dim = get_text_model(args.text)

print("Loading audio encoder:", args.audio)
audio_kind, load_audio, encode_audio, audio_dim = get_audio_model(args.audio)

print("Loading image encoder:", args.image)
load_image, encode_image, image_dim = get_image_model(args.image)


# ==========================================================
# LOAD MULTIMODAL DATA
# NOTE (methodology limitation, unchanged from the original script):
# samples are paired by emotion label + index position across three
# independently-collected datasets (FER2013 / RAVDESS / text). They are
# NOT synchronized recordings of the same person/session. Report this
# as a limitation — it affects how strongly "true multimodal fusion"
# claims can be made.
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
            print(f"Skipping {emo}: text={len(text_files)}, audio={len(audio_files)}, image={len(image_files)}")
            continue

        max_len = max(len(text_files), len(audio_files), len(image_files))
        for i in range(max_len):
            txt_path = text_files[i % len(text_files)]
            aud_path = audio_files[i % len(audio_files)]
            img_path = image_files[i % len(image_files)]

            with open(txt_path, encoding="utf-8") as f:
                txt = f.read().strip()
            if not txt:
                continue

            texts.append(txt)
            audios.append(aud_path)
            images.append(img_path)
            labels.append(emo)

    return pd.DataFrame({"text": texts, "audio": audios, "image": images, "emotion": labels})


data_df = load_multimodal_data()
print("\nFULL DATASET:")
print(data_df["emotion"].value_counts())

# ==========================================================
# HANDLE SMALL CLASSES
# ==========================================================
min_samples = 5
balanced_rows = []
for emo in VALID_EMOTIONS:
    subset = data_df[data_df["emotion"] == emo]
    if len(subset) == 0:
        print(f"WARNING: No samples for {emo}")
        continue
    if len(subset) < min_samples:
        extra = subset.sample(min_samples - len(subset), replace=True, random_state=42)
        subset = pd.concat([subset, extra])
    balanced_rows.append(subset)
data_df = pd.concat(balanced_rows).reset_index(drop=True)

# ==========================================================
# STRATIFIED SPLIT
# ==========================================================
train_df, temp_df = train_test_split(data_df, test_size=0.3, stratify=data_df["emotion"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=42)

print("\nTRAIN:\n", train_df["emotion"].value_counts())
print("\nVAL:\n", val_df["emotion"].value_counts())
print("\nTEST:\n", test_df["emotion"].value_counts())


# ==========================================================
# DATASET
# ==========================================================
class MultimodalDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.audio = df["audio"].tolist()
        self.image = df["image"].tolist()
        self.labels = [EMOTIONS[e] for e in df["emotion"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = encode_text(self.texts[idx])

        raw_audio = load_audio(self.audio[idx])
        if audio_kind == "cnn":
            a = torch.tensor(raw_audio, dtype=torch.float)  # (T, 40) MFCC, CNN handles it in Model
        else:
            a = encode_audio(raw_audio)  # already a pooled embedding vector

        v = encode_image(load_image(self.image[idx]))
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return t, a, v, y


train_ds = MultimodalDataset(train_df)
val_ds = MultimodalDataset(val_df)
test_ds = MultimodalDataset(test_df)

# ==========================================================
# CLASS WEIGHTS
# ==========================================================
labels_list = [EMOTIONS[e] for e in train_df["emotion"]]
counts = Counter(labels_list)
total = sum(counts.values())
weights = [total / (NUM_CLASSES * (counts[i] + 1e-6)) for i in range(NUM_CLASSES)]
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

sample_weights = [weights[l] for l in labels_list]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


# ==========================================================
# ATTENTION FUSION (unchanged from the original architecture)
# ==========================================================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        return torch.matmul(attn, V).mean(dim=1)


# ==========================================================
# MULTIMODAL MODEL
# Projects whichever encoder outputs were selected down to a shared
# 256-dim space, then fuses with attention exactly as in the original.
# The audio CNN is only instantiated when audio_kind == "cnn"; for
# pretrained audio encoders, audio arrives pre-pooled and only needs
# a linear projection like text/image.
# ==========================================================
class Model(nn.Module):
    def __init__(self, text_dim, audio_dim, image_dim, audio_kind):
        super().__init__()
        self.audio_kind = audio_kind

        self.t_fc = nn.Linear(text_dim, 256)
        self.v_fc = nn.Linear(image_dim, 256)

        if audio_kind == "cnn":
            self.audio_cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.a_fc = nn.Linear(128, 256)
        else:
            self.audio_cnn = None
            self.a_fc = nn.Linear(audio_dim, 256)

        self.attn = AttentionFusion(256)
        self.cls = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, t, a, v):
        t = torch.relu(self.t_fc(t))

        if self.audio_kind == "cnn":
            a = a.permute(0, 2, 1).unsqueeze(1)  # (B,1,40,T)
            a = self.audio_cnn(a)
            a = a.view(a.size(0), -1)
        a = torch.relu(self.a_fc(a))

        v = torch.relu(self.v_fc(v))

        x = torch.stack([t, a, v], dim=1)
        return self.cls(self.attn(x))


model = Model(text_dim, audio_dim, image_dim, audio_kind).to(DEVICE)


# ==========================================================
# LOSS
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
# TRAINING + EARLY STOPPING
# ==========================================================
train_accs, val_accs = [], []
train_losses, val_losses = [], []
best_val = float("inf")
patience, counter = 5, 0

for epoch in range(EPOCHS):
    model.train()
    preds, labels = [], []
    train_loss = 0

    for t, a, v, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(t, a, v)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds += out.argmax(1).detach().cpu().tolist()
        labels += y.cpu().tolist()

    train_loss /= len(train_loader)
    train_acc = accuracy_score(labels, preds)

    model.eval()
    preds, labels = [], []
    val_loss = 0
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

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")
    print(f"Val   Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f" Best model saved -> {CHECKPOINT_PATH}")
    else:
        counter += 1
        if counter >= patience:
            print(" Early stopping triggered!")
            break

# ==========================================================
# PLOTS
# ==========================================================
epochs_range = range(1, len(train_accs) + 1)

plt.figure()
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, val_accs, label="Validation Accuracy")
plt.legend(); plt.title(f"{RUN_NAME} - Accuracy vs Epoch"); plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, f"{RUN_NAME}_accuracy.png")); plt.show()

plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.legend(); plt.title(f"{RUN_NAME} - Loss vs Epoch"); plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, f"{RUN_NAME}_loss.png")); plt.show()


# ==========================================================
# EVALUATION
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=VALID_EMOTIONS, zero_division=0))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=VALID_EMOTIONS, yticklabels=VALID_EMOTIONS)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"{RUN_NAME} - {name} Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, f"{RUN_NAME}_{name.lower()}_confusion_matrix.png"))
    plt.show()

    return acc, precision, recall, f1


if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"\nLoaded best checkpoint: {CHECKPOINT_PATH}")

train_acc, _, _, _ = evaluate(train_loader, "Train")
val_acc, _, _, _ = evaluate(val_loader, "Validation")
test_acc, test_prec, test_rec, test_f1 = evaluate(test_loader, "Test")

# ==========================================================
# APPEND RESULTS TO SHARED CSV (for cross-experiment comparison)
# ==========================================================
row = {
    "experiment": RUN_NAME,
    "text_model": args.text,
    "audio_model": args.audio,
    "image_model": args.image,
    "train_accuracy": round(train_acc, 4),
    "val_accuracy": round(val_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "test_precision_macro": round(test_prec, 4),
    "test_recall_macro": round(test_rec, 4),
    "test_f1_macro": round(test_f1, 4),
}

file_exists = os.path.isfile(RESULTS_CSV)
with open(RESULTS_CSV, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"\nResults appended to {RESULTS_CSV}")
print(row)
