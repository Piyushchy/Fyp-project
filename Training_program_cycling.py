# ==========================================================
# MULTIMODAL EMOTION RECOGNITION
# ==========================================================

import os, glob, cv2, torch, librosa
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel, ViTModel, ViTImageProcessor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4

# FER_PATH = "addidata_balanced/fer2013/images"
# RAVDESS_PATH = "addidata_balanced/ravdess/audio"
# TEXT_PATH = "addidata_balanced/text"

# FER_PATH = "addidata/fer2013/images"
# RAVDESS_PATH = "addidata/ravdess/audio"
# TEXT_PATH = "addidata/text"


# FER_PATH = "data_balanced/fer2013/images"
# RAVDESS_PATH = "data_balanced/ravdess/audio"
# TEXT_PATH = "data_balanced/text"

FER_PATH = "data/fer2013/images"
RAVDESS_PATH = "data/ravdess/audio"
TEXT_PATH = "data/text"

VALID_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise"]
EMOTIONS = {e: i for i, e in enumerate(VALID_EMOTIONS)}
NUM_CLASSES = len(VALID_EMOTIONS)


# ==========================================================
# TEXT (BERT)
# ==========================================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert.eval()

def encode_text(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return bert(**tokens).last_hidden_state[:, 0, :].squeeze(0)

# ==========================================================
# AUDIO
# ==========================================================
def load_audio(path, max_len=200):
    signal, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T

    if len(mfcc) < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)))
    else:
        mfcc = mfcc[:max_len]

    return mfcc.astype(np.float32)

# ==========================================================
# IMAGE (ViT)
# ==========================================================
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
vit.eval()

for p in vit.parameters():
    p.requires_grad = False

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (224, 224))

def encode_image(image):
    inputs = vit_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return vit(**inputs).last_hidden_state[:, 0, :].squeeze(0)

# ==========================================================
# LOAD MULTIMODAL DATA
# ==========================================================
def load_multimodal_data():
    texts, audios, images, labels = [], [], [], []

    for emo in VALID_EMOTIONS:
        text_dir  = os.path.join(TEXT_PATH, emo)
        audio_dir = os.path.join(RAVDESS_PATH, emo)
        image_dir = os.path.join(FER_PATH, emo)

        text_files  = sorted(glob.glob(os.path.join(text_dir, "*.txt")))
        audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")) +
                             glob.glob(os.path.join(audio_dir, "*.mp3")))
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                             glob.glob(os.path.join(image_dir, "*.png")))

        if len(text_files)==0 or len(audio_files)==0 or len(image_files)==0:
            print(f" Skipping {emo}")
            continue

        max_len = max(len(text_files), len(audio_files), len(image_files))

        for i in range(max_len):
            txt_path  = text_files[i % len(text_files)]
            aud_path  = audio_files[i % len(audio_files)]
            img_path  = image_files[i % len(image_files)]

            with open(txt_path, encoding="utf-8") as f:
                txt = f.read().strip()

            if not txt:
                continue

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

# LOAD DATA
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

    if len(subset) < min_samples:
        extra = subset.sample(min_samples - len(subset), replace=True, random_state=42)
        subset = pd.concat([subset, extra])

    balanced_rows.append(subset)

data_df = pd.concat(balanced_rows).reset_index(drop=True)

# ==========================================================
# STRATIFIED SPLIT
# ==========================================================
train_df, temp_df = train_test_split(
    data_df, test_size=0.3, stratify=data_df["emotion"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=42
)

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

        # MFCC
        mfcc = load_audio(self.audio[idx])
        a = torch.tensor(mfcc, dtype=torch.float)

        v = encode_image(load_image(self.image[idx]))
        y = torch.tensor(self.labels[idx])

        return t, a, v, y

train_ds = MultimodalDataset(train_df)
val_ds   = MultimodalDataset(val_df)
test_ds  = MultimodalDataset(test_df)

# ==========================================================
# CLASS WEIGHTS (FIXED)
# ==========================================================
labels_list = [EMOTIONS[e] for e in train_df["emotion"]]
counts = Counter(labels_list)
total = sum(counts.values())

weights = [total / (NUM_CLASSES * (counts[i] + 1e-6)) for i in range(NUM_CLASSES)]
class_weights = torch.tensor(weights).to(DEVICE)

sample_weights = [weights[l] for l in labels_list]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ==========================================================
# MODEL
# ==========================================================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        Q,K,V = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1))/self.scale, dim=-1)
        return torch.matmul(attn, V).mean(dim=1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # TEXT
        self.t_fc = nn.Linear(768,256)

        # AUDIO CNN
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.a_fc = nn.Linear(128,256)

        # IMAGE
        self.v_fc = nn.Linear(768,256)

        # FUSION
        self.attn = AttentionFusion(256)

        self.cls = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,NUM_CLASSES)
        )

    def forward(self, t, a, v):
        # TEXT
        t = torch.relu(self.t_fc(t))

        # AUDIO CNN FORWARD
        a = a.permute(0,2,1).unsqueeze(1)   # (B,1,40,T)
        a = self.audio_cnn(a)
        a = a.view(a.size(0), -1)
        a = torch.relu(self.a_fc(a))

        # IMAGE
        v = torch.relu(self.v_fc(v))

        # FUSION
        x = torch.stack([t,a,v], dim=1)
        return self.cls(self.attn(x))

model = Model().to(DEVICE)

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
        return ((1-pt)**self.gamma)*ce

criterion = FocalLoss(alpha=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================================
# TRAINING + TRACKING + EARLY STOPPING
# ==========================================================
train_accs, val_accs = [], []
train_losses, val_losses = [], []

best_val = float("inf")
patience, counter = 5, 0

for epoch in range(EPOCHS):
    model.train()
    preds, labels = [], []
    train_loss = 0

    for t,a,v,y in tqdm(train_loader):
        t,a,v,y = t.to(DEVICE),a.to(DEVICE),v.to(DEVICE),y.to(DEVICE)

        optimizer.zero_grad()
        out = model(t,a,v)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds += out.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()

    train_loss /= len(train_loader)
    train_acc = accuracy_score(labels,preds)

    model.eval()
    preds, labels = [], []
    val_loss = 0

    with torch.no_grad():
        for t,a,v,y in val_loader:
            t,a,v,y = t.to(DEVICE),a.to(DEVICE),v.to(DEVICE),y.to(DEVICE)
            out = model(t,a,v)
            loss = criterion(out,y)

            val_loss += loss.item()
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()

    val_loss /= len(val_loader)
    val_acc = accuracy_score(labels,preds)

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
        #torch.save(model.state_dict(),"addidata_balanced/addidata_balanced_Finalmodel.pt")
        #torch.save(model.state_dict(),"addidata/addidata_Finalmodel.pt")
        #torch.save(model.state_dict(),"data_balanced/data_balanced_Finalmodel.pt")
        torch.save(model.state_dict(),"data/data_Finalmodel.pt")
        
        print(" Best model saved")
    else:
        counter += 1
        if counter >= patience:
            print(" Early stopping triggered!")
            break

# ==========================================================
# PLOTS
# ==========================================================
epochs = range(1, len(train_accs)+1)

plt.figure()
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Validation Accuracy")
plt.legend(); plt.title("Accuracy vs Epoch"); plt.grid(); plt.show()

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.legend(); plt.title("Loss vs Epoch"); plt.grid(); plt.show()

# ==========================================================
# EVALUATION
# ==========================================================
def evaluate(loader,name):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for t,a,v,y in loader:
            t,a,v = t.to(DEVICE),a.to(DEVICE),v.to(DEVICE)
            out = model(t,a,v)
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()

    print(f"\n{name} Accuracy:", accuracy_score(labels,preds))

    cm = confusion_matrix(labels,preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=VALID_EMOTIONS,
                yticklabels=VALID_EMOTIONS)
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    print(classification_report(labels,preds,target_names=VALID_EMOTIONS))

#model.load_state_dict(torch.load("addidata_balanced/addidata_balanced_Finalmodel.pt"))
#model.load_state_dict(torch.load("addidata/addidata_Finalmodel.pt"))
#model.load_state_dict(torch.load("data_balanced/Finalmodel.pt"))
model.load_state_dict(torch.load("data/data_Finalmodel.pt"))


evaluate(train_loader,"Train")
evaluate(val_loader,"Validation")
evaluate(test_loader,"Test")