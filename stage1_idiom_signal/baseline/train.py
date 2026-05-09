from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.idiom_signal import IdiomSignalModel

CACHE_PATH = REPO_ROOT / "stage1_idiom_signal/cache/embedding_cache_pooled.pt"
OUTPUT_DIR = REPO_ROOT / "stage1_idiom_signal/baseline/output"
MODEL_NAME = "xlm-roberta-base"
EPOCHS = 30  # Reduced for hyperparameter search speed
MAX_LENGTH = 128
VAL_RATIO = 0.15
SEED = 42

class CachedIdiomDataset(Dataset):
    def __init__(self, cache_path: Path) -> None:
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found at {cache_path}. Please build cache first.")
            
        cache = torch.load(cache_path, map_location="cpu")
        pooled = cache["pooled"].float()
        self.metadata = cache["metadata"]
        text_keys = cache["text_keys_order"]
        key_to_index = {key: index for index, key in enumerate(text_keys)}

        self.target_embs = pooled[:, key_to_index["target"]]
        self.labels = torch.tensor([float(m["label"]) for m in self.metadata], dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target_embs[index], self.labels[index]

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_train_val(labels: torch.Tensor, val_ratio: float, seed: int):
    labels_list = labels.tolist()
    indices = list(range(len(labels_list)))
    rng = random.Random(seed)
    
    by_label = {0.0: [], 1.0: []}
    for idx, lbl in enumerate(labels_list):
        by_label[lbl].append(idx)
        
    train_indices = []
    val_indices = []
    
    for lbl, group in by_label.items():
        rng.shuffle(group)
        val_count = max(1, round(len(group) * val_ratio))
        val_indices.extend(group[:val_count])
        train_indices.extend(group[val_count:])
        
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices

def compute_metrics(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float | None]:
    probs = probabilities.cpu().numpy()
    lbls = labels.cpu().numpy()
    preds = (probs >= 0.5).astype(float)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    roc_auc = None
    if len(set(lbls)) > 1:
        roc_auc = float(roc_auc_score(lbls, probs))

    return {
        "accuracy": float(accuracy_score(lbls, preds)),
        "precision": float(precision_score(lbls, preds, zero_division=0)),
        "recall": float(recall_score(lbls, preds, zero_division=0)),
        "f1": float(f1_score(lbls, preds, zero_division=0)),
        "roc_auc": roc_auc,
    }

def objective(trial: optuna.Trial, dataset, train_indices, val_indices, device) -> float:
    hidden_1 = trial.suggest_categorical("hidden_1", [256, 512, 1024])
    hidden_2 = trial.suggest_categorical("hidden_2", [64, 128, 256])
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.5, step=0.1)
    lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    
    # We build the classifier directly to avoid loading XLM-R into memory for every trial
    classifier = nn.Sequential(
        nn.Linear(768, hidden_1),
        nn.BatchNorm1d(hidden_1),
        nn.GELU(),
        nn.Dropout(dropout_1),
        nn.Linear(hidden_1, hidden_2),
        nn.BatchNorm1d(hidden_2),
        nn.GELU(),
        nn.Dropout(dropout_2),
        nn.Linear(hidden_2, 1)
    ).to(device)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(embs).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
        classifier.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for embs, labels in val_loader:
                embs, labels = embs.to(device), labels.to(device)
                logits = classifier(embs).squeeze(-1)
                all_logits.append(logits)
                all_labels.append(labels)
                
        metrics = compute_metrics(torch.sigmoid(torch.cat(all_logits)), torch.cat(all_labels))
        f1 = metrics["f1"]
        
        if f1 > best_f1:
            best_f1 = f1
            
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_f1

def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Cached Target Embeddings...")
    dataset = CachedIdiomDataset(CACHE_PATH)
    train_indices, val_indices = split_train_val(dataset.labels, VAL_RATIO, SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create Optuna study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    print("Starting Optuna Hyperparameter Search (30 Trials) for Baseline...")
    
    study.optimize(lambda trial: objective(trial, dataset, train_indices, val_indices, device), n_trials=30)
    
    print("\nSearch Complete!")
    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"Best Trial F1: {best_trial.value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Retrain the absolute best model
    print("\nRetraining Best Baseline Model for 50 epochs...")
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=best_params["batch_size"], shuffle=False)
    
    model = IdiomSignalModel(
        model_name=MODEL_NAME, 
        freeze_encoder=True,
        hidden_1=best_params["hidden_1"],
        hidden_2=best_params["hidden_2"],
        dropout_1=best_params["dropout_1"],
        dropout_2=best_params["dropout_2"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = -1.0
    best_epoch = 0
    best_metrics = None
    history = []
    
    for epoch in range(1, 51):
        model.classifier.train()
        total_loss = 0.0
        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.classifier(embs).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            
        train_loss = total_loss / len(train_indices)
        
        model.classifier.eval()
        all_logits = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for embs, labels in val_loader:
                embs, labels = embs.to(device), labels.to(device)
                logits = model.classifier(embs).squeeze(-1)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                all_logits.append(logits)
                all_labels.append(labels)
                
        metrics = compute_metrics(torch.sigmoid(torch.cat(all_logits)), torch.cat(all_labels))
        val_loss /= len(val_indices)
        metrics["loss"] = float(val_loss)
        
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            **metrics
        })
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_metrics = metrics
            
            torch.save({
                "epoch": epoch,
                "classifier_state_dict": model.classifier.state_dict(),
                "metrics": metrics,
            }, OUTPUT_DIR / "idiom_signal.pt")
            
    # Save config mapping so it can be loaded
    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "dropout": model.dropout,
        "encoder_frozen_during_training": True,
        "hyperparameters": best_params,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "seed": SEED,
    }
    save_json(OUTPUT_DIR / "config.json", config)
    model.tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
    
    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "history": history,
        "hyperparameters": best_params,
        "metrics": best_metrics
    }
    save_json(OUTPUT_DIR / "metrics.json", metrics_payload)
    
    print(f"\nFinal Best Val F1: {best_f1:.4f}. Model, config, and metrics saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
