# ============================================================
# TEXT EMOTION MODULE - DATA
# ============================================================
#
# Loads the MTEB EmotionClassification dataset and turns it
# into padded PyTorch batches.
#
# Splits (fixed by MTEB, not re-shuffled here):
#   train      15956
#   validation  1988
#   test        1986
# ============================================================

from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from config import DATASET_ID, ID2LABEL, NUM_LABELS, SEED


# ============================================================
# LOAD RAW SPLITS
# ============================================================

def load_splits():
    """Return the MTEB emotion DatasetDict (downloads on first call)."""

    dataset = load_dataset(DATASET_ID)

    # Guard against a silent label-order change upstream. The
    # integer ids are what the classifier head learns, so a
    # remap on the hub would invalidate saved checkpoints.
    observed = dict(
        zip(dataset["train"]["label"], dataset["train"]["label_text"])
    )

    assert observed == ID2LABEL, (
        f"Upstream label order is {observed}, expected {ID2LABEL}. "
        f"Update ID2LABEL in config.py."
    )

    return dataset


# ============================================================
# TOKENISATION
# ============================================================
#
# Sequences are tokenised without padding and padded per batch
# instead. The texts are short - mean 22 word-piece tokens,
# p99 57, longest 87 - so padding every row to max_length would
# spend roughly 5x more CPU on padding than on real tokens.
# ============================================================

class EncodedSplit(Dataset):
    """Variable-length tokenised rows, padded at collate time."""

    def __init__(self, split, tokenizer, max_length):

        self.input_ids = tokenizer(
            list(split["text"]),
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        self.labels = list(split["label"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]


def collate(batch, pad_token_id):
    """Pad a batch to its own longest sequence."""

    sequences, labels = zip(*batch)

    longest = max(len(sequence) for sequence in sequences)

    input_ids = torch.full((len(sequences), longest), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), longest), dtype=torch.long)

    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1

    return input_ids, attention_mask, torch.tensor(labels, dtype=torch.long)


def encode_split(split, tokenizer, max_length):
    """Tokenise one split into a padding-free dataset."""

    return EncodedSplit(split, tokenizer, max_length)


def build_dataloaders(tokenizer, batch_size, max_length, max_train_samples=None):
    """Build train / validation / test loaders for one tokenizer.

    max_train_samples truncates the training split only. It exists for
    smoke tests; the evaluation splits always stay complete so reported
    numbers remain comparable across runs.
    """

    dataset = load_splits()

    train_split = dataset["train"]

    if max_train_samples is not None and max_train_samples < len(train_split):
        train_split = train_split.shuffle(seed=SEED).select(range(max_train_samples))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    pad = partial(collate, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        encode_split(train_split, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=pad,
    )

    validation_loader = DataLoader(
        encode_split(dataset["validation"], tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=pad,
    )

    test_loader = DataLoader(
        encode_split(dataset["test"], tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=pad,
    )

    return train_loader, validation_loader, test_loader


# ============================================================
# CLASS WEIGHTS
# ============================================================

def class_weights(loader):
    """Inverse-frequency weights for the cross-entropy loss.

    MTEB emotion is heavily imbalanced - 'joy' has 5345 training
    rows while 'surprise' has 568. Without weighting, the rare
    classes are largely ignored and macro-F1 suffers.
    """

    counts = np.bincount(
        loader.dataset.labels,
        minlength=NUM_LABELS,
    ).astype(np.float64)

    # Guard against a subset that happens to miss a class.
    counts[counts == 0] = 1.0

    weights = counts.sum() / (NUM_LABELS * counts)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# MANUAL CHECK
# ============================================================

if __name__ == "__main__":

    dataset = load_splits()

    print("Splits:")

    for name, split in dataset.items():
        print(f"  {name:<11} {len(split):>6} rows")

    print("\nLabel distribution (train):")

    counts = np.bincount(dataset["train"]["label"], minlength=NUM_LABELS)

    for index, name in ID2LABEL.items():
        print(f"  {index} {name:<9} {counts[index]:>6}")

    print("\nExamples:")

    for row in dataset["test"].select(range(5)):
        print(f"  [{row['label_text']:<8}] {row['text'][:70]}")
