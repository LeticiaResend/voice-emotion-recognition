"""
Train Model 1 (CNN from scratch) on one fold of EmoDB, evaluate on that
fold's held-out test speakers, and save curves + confusion matrix for QA.
"""

import json
import time
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, confusion_matrix

from features import fit_standardizer, extract_features
from dataset import EmoDBDataset, EMOTIONS
from model_scratch import ScratchCNN

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():  # Apple Silicon
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, uar, all_labels, all_preds


def train_one_fold(fold_id: int, max_epochs: int = 60, patience: int = 12,
                    batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
                    class_weight_power: float = 0.5):

    torch.manual_seed(seed)

    from paths import EMODB_METADATA_CSV, EMODB_FOLDS_JSON
    df = pd.read_csv(EMODB_METADATA_CSV, dtype={"speaker_id": str})
    folds = json.loads(open(EMODB_FOLDS_JSON).read())
    fold = folds[fold_id]
    print(f"Fold {fold_id}: train={fold['train_speakers']}, val={fold['val_speaker']}, test={fold['test_speakers']}")

    # Standardizer fit ONLY on this fold's training speakers.
    train_feats = [extract_features(fp) for fp in
                   df[df["speaker_id"].isin(fold["train_speakers"])]["filepath"]]
    mean, std = fit_standardizer(train_feats)

    audio_cache: dict = {}
    train_ds = EmoDBDataset(df, fold["train_speakers"], mean, std, augment=True, seed=seed, audio_cache=audio_cache)
    val_ds = EmoDBDataset(df, [fold["val_speaker"]], mean, std, augment=False, audio_cache=audio_cache)
    test_ds = EmoDBDataset(df, fold["test_speakers"], mean, std, augment=False, audio_cache=audio_cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    train_counts = df[df["speaker_id"].isin(fold["train_speakers"])]["emotion"].value_counts()
    class_weights = torch.tensor(
        [(1.0 / train_counts.get(e, 1)) ** class_weight_power for e in EMOTIONS], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * len(EMOTIONS)

    model = ScratchCNN(n_classes=len(EMOTIONS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_uar": []}
    best_val_uar = -1.0
    best_state = None
    epochs_without_improvement = 0

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_uar, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_uar"].append(val_uar)

        improved = val_uar > best_val_uar
        if improved:
            best_val_uar = val_uar
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(f"epoch {epoch:3d}  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  val_UAR={val_uar:.3f}{marker}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no val_UAR improvement in {patience} epochs.")
            break

    elapsed = time.time() - t0
    print(f"Training took {elapsed:.1f}s ({elapsed/epoch:.2f}s/epoch, {epoch} epochs)")

    model.load_state_dict(best_state)
    test_loss, test_uar, test_labels, test_preds = evaluate(model, test_loader, criterion)
    print(f"\nBest val_UAR: {best_val_uar:.3f}")
    print(f"Test (held-out speakers {fold['test_speakers']}): loss={test_loss:.3f}, UAR={test_uar:.3f}")

    per_class_recall = recall_score(test_labels, test_preds, average=None, labels=range(len(EMOTIONS)), zero_division=0)
    print("Per-class recall on test:")
    for e, r in zip(EMOTIONS, per_class_recall):
        print(f"  {e:10s}: {r:.2f}")

    cm = confusion_matrix(test_labels, test_preds, labels=range(len(EMOTIONS)))

    return {
        "fold": fold,
        "history": history,
        "best_val_uar": best_val_uar,
        "test_uar": test_uar,
        "test_loss": test_loss,
        "confusion_matrix": cm,
        "model_state": best_state,
        "mean": mean,
        "std": std,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    result = train_one_fold(args.fold, class_weight_power=args.class_weight_power)

    from paths import PROCESSED_DIR
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = PROCESSED_DIR / f"scratch_fold{args.fold}{suffix}_result.pt"
    torch.save(result, out_path)
    print(f"\nSaved result to {out_path}")
