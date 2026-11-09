
#!/usr/bin/env python3

"""
Stage 5: Train the feature-enhanced CNN-BiLSTM.

Input:
    feature_dataset/features.h5

Output:
    models/cnn_bilstm_features_best.pt
    results/training_history.csv

Input channels:
    5  DNA one-hot
    2  GC content and GC skew
    16 dinucleotide frequencies
    64 trinucleotide frequencies

Total: 87 channels

Labels:
    E = 0
    I = 1
    X = 2 (ignored padding)

The test partition is never used during training or
validation-based checkpoint selection.
"""

import argparse
import csv
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


WINDOW_LENGTH = 2048
INPUT_CHANNELS = 87
NUM_CLASSES = 2

CLASS_NAMES = ["Exon", "Intron"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GenomicDataset(Dataset):

    def __init__(self, h5_path, split):

        self.h5_path = str(h5_path)
        self.split = split
        self.file = None

        with h5py.File(self.h5_path, "r") as h5:

            splits = h5["metadata/Split"][:]

            splits = np.array([
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in splits
            ])

            self.indices = np.flatnonzero(
                splits == split
            )

        if len(self.indices) == 0:
            raise ValueError(
                f"No windows found for split: {split}"
            )

        print(
            f"{split}: {len(self.indices):,} windows"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        row = int(self.indices[index])

        sequence = self.file["sequence"][row]
        composition = self.file["composition"][row]
        kmer = self.file["kmer"][row]
        labels = self.file["labels"][row]

        # Convert DNA integer encoding to one-hot.
        # A=0, C=1, G=2, T=3, N=4.
        sequence_tensor = torch.from_numpy(
            sequence.astype(np.int64)
        )

        one_hot = F.one_hot(
            sequence_tensor,
            num_classes=5
        ).float()

        # Shape: (L, 5)
        # Composition: uint8 [0,255].
        composition_tensor = torch.from_numpy(
            composition.astype(np.float32) / 255.0
        )

        # Decode GC skew from [0,255] back to [-1,1].
        composition_tensor[:, 1] = (
            composition_tensor[:, 1] * 2.0 - 1.0
        )

        # K-mer frequencies: uint8 [0,255].
        kmer_tensor = torch.from_numpy(
            kmer.astype(np.float32) / 255.0
        )

        # Concatenate:
        # (L,5) + (L,2) + (L,80) = (L,87)
        features = torch.cat(
            [
                one_hot,
                composition_tensor,
                kmer_tensor
            ],
            dim=1
        )

        # PyTorch Conv1d expects (channels, sequence_length).
        features = features.transpose(0, 1).contiguous()

        target = torch.from_numpy(
            labels.astype(np.int64)
        )

        return features, target


class CNNBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Conv1d(
            in_channels=INPUT_CHANNELS,
            out_channels=32,
            kernel_size=9,
            padding="same"
        )

        self.relu = nn.ReLU()

        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(
            in_features=64,
            out_features=NUM_CLASSES
        )

    def forward(self, x):

        # Input: (B, 87, 2048)
        x = self.cnn(x)

        # (B, 32, 2048)
        x = self.relu(x)

        # BiLSTM expects (B, L, features).
        x = x.transpose(1, 2)

        # (B, 2048, 64)
        x, _ = self.bilstm(x)

        # (B, 2048, 2)
        x = self.classifier(x)

        return x


def calculate_class_weights(h5_path):

    """
    Calculate class weights from training-set base occurrences.

    Exon and intron counts are read from the window metadata.
    Overlapping windows are counted as occurrences, consistent
    with the training objective.
    """

    with h5py.File(h5_path, "r") as h5:

        splits = h5["metadata/Split"][:]

        splits = np.array([
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in splits
        ])

        training_mask = splits == "Training"

        exon_counts = h5["metadata/ExonBases"][:]
        intron_counts = h5["metadata/IntronBases"][:]

        exon_total = int(
            np.sum(exon_counts[training_mask])
        )

        intron_total = int(
            np.sum(intron_counts[training_mask])
        )

    total = exon_total + intron_total

    if exon_total == 0 or intron_total == 0:
        raise ValueError("Invalid class counts.")

    weights = torch.tensor(
        [
            total / (2.0 * exon_total),
            total / (2.0 * intron_total)
        ],
        dtype=torch.float32
    )

    print("\nTraining class statistics:")
    print(f"Exon occurrences: {exon_total:,}")
    print(f"Intron occurrences: {intron_total:,}")
    print(f"Exon weight: {weights[0].item():.4f}")
    print(f"Intron weight: {weights[1].item():.4f}")

    return weights


def calculate_metrics(confusion):

    results = {}

    f1_values = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        tp = confusion[class_index, class_index]

        fp = (
            confusion[:, class_index].sum() - tp
        )

        fn = (
            confusion[class_index, :].sum() - tp
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        f1_values.append(f1)

    results["macro_f1"] = float(np.mean(f1_values))

    return results


def run_epoch(model, loader, criterion, optimizer, device):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_batches = 0

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    context = torch.enable_grad() if training else torch.no_grad()

    with context:

        for features, targets in loader:

            features = features.to(
                device, non_blocking=True
            )

            targets = targets.to(
                device, non_blocking=True
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)

            # Padding label 2 is ignored by CrossEntropyLoss.
            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                targets.reshape(-1)
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            predictions = logits.argmax(dim=-1)

            valid = targets != 2

            true_valid = targets[valid].cpu().numpy()
            pred_valid = predictions[valid].cpu().numpy()

            batch_confusion = np.bincount(
                NUM_CLASSES * true_valid + pred_valid,
                minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            confusion += batch_confusion

    metrics = calculate_metrics(confusion)

    average_loss = total_loss / max(total_batches, 1)

    return average_loss, metrics


def main():

    parser = argparse.ArgumentParser(
        description="Train feature-enhanced CNN-BiLSTM."
    )

    parser.add_argument(
        "--data",
        default="feature_dataset/features.h5"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output",
        default="models/cnn_bilstm_features_best.pt"
    )

    parser.add_argument(
        "--history",
        default="results/training_history.csv"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data)
    checkpoint_path = Path(args.output)
    history_path = Path(args.history)

    checkpoint_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    history_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Load train and validation partitions only.
    train_dataset = GenomicDataset(
        data_path, "Training"
    )

    val_dataset = GenomicDataset(
        data_path, "Validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    class_weights = calculate_class_weights(
        data_path
    ).to(device)

    model = CNNBiLSTM().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=2
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    best_macro_f1 = -1.0
    history = []

    print("\nStarting training...")

    for epoch in range(1, args.epochs + 1):

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device
        )

        print(f"Training loss: {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")

        for class_name in CLASS_NAMES:

            print(
                f"{class_name} validation: "
                f"Precision={val_metrics[class_name]['precision']:.4f}, "
                f"Recall={val_metrics[class_name]['recall']:.4f}, "
                f"F1={val_metrics[class_name]['f1']:.4f}"
            )

        print(
            f"Validation macro F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        history.append({
            "Epoch": epoch,
            "TrainingLoss": train_loss,
            "ValidationLoss": val_loss,
            "TrainingMacroF1": train_metrics["macro_f1"],
            "ValidationMacroF1": val_metrics["macro_f1"]
        })

        # Select checkpoint using validation macro F1 only.
        if val_metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = val_metrics["macro_f1"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_macro_f1": best_macro_f1,
                "class_weights": class_weights.cpu(),
                "model_config": {
                    "input_channels": INPUT_CHANNELS,
                    "cnn_filters": 32,
                    "cnn_kernel_size": 9,
                    "bilstm_hidden_size": 32,
                    "bidirectional": True,
                    "num_classes": NUM_CLASSES,
                    "window_length": WINDOW_LENGTH
                },
                "preprocessing": {
                    "sequence_encoding": "A=0,C=1,G=2,T=3,N=4",
                    "label_encoding": "E=0,I=1,X=2",
                    "composition_scaling": "GC / 255; skew decoded to [-1,1]",
                    "kmer_scaling": "uint8 / 255",
                    "feature_channels": 87
                }
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                f"Best checkpoint saved: {checkpoint_path}"
            )

    # Save training history.
    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Epoch",
                "TrainingLoss",
                "ValidationLoss",
                "TrainingMacroF1",
                "ValidationMacroF1"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 52)
    print("TRAINING COMPLETED")
    print("=" * 52)
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print("Test partition was not used.")
    print("=" * 52)


if __name__ == "__main__":
    main()
