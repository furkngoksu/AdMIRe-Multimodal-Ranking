import json
import random
from pathlib import Path
from typing import Any
import sys

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configuration
CACHE_PATH = REPO_ROOT / "stage1_idiom_signal/cache/embedding_cache_pooled.pt"
OUTPUT_DIR = REPO_ROOT / "stage1_idiom_signal/llm/output"
EPOCHS = 20  # Reduced for faster hyperparameter search
VAL_RATIO = 0.15
SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class NeuralMSignalDataset(Dataset):
    def __init__(self, cache_path: Path):
        cache = torch.load(cache_path, map_location="cpu")
        pooled = cache["pooled"].float()
        self.metadata = cache["metadata"]
        text_keys = cache["text_keys_order"]
        key_to_index = {key: index for index, key in enumerate(text_keys)}

        center = pooled.reshape(-1, pooled.size(-1)).mean(dim=0, keepdim=True)
        emb = pooled - center.view(1, 1, -1)

        target = emb[:, key_to_index["target"]]
        paraphrase = emb[:, key_to_index["paraphrase"]]
        lex_sub = emb[:, key_to_index["lex_sub"]]
        comp_drop = emb[:, key_to_index["comp_drop"]]
        literalize = emb[:, key_to_index["literalize"]]

        f_sub = torch.abs(target - lex_sub)
        f_comp = torch.abs(target - comp_drop)
        f_lit = torch.abs(target - literalize)
        f_para = torch.abs(target - paraphrase)

        self.features = torch.cat([f_sub, f_comp, f_lit, f_para], dim=-1)
        self.labels = torch.tensor([float(m["label"]) for m in self.metadata], dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

class NeuralMSignalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, dropout_1: float, dropout_2: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.GELU(),
            nn.Dropout(dropout_1),
            
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.GELU(),
            nn.Dropout(dropout_2),
            
            nn.Linear(hidden_2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

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

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = labels.cpu().numpy()
    preds = (probs >= 0.5).astype(float)
    
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs))
    }

def objective(trial: optuna.Trial, dataset, train_indices, val_indices, device) -> float:
    # 1. Hyperparameters
    hidden_1 = trial.suggest_categorical("hidden_1", [256, 512, 1024])
    hidden_2 = trial.suggest_categorical("hidden_2", [64, 128, 256])
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.5, step=0.1)
    lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    
    model = NeuralMSignalModel(
        input_dim=3072, 
        hidden_1=hidden_1, 
        hidden_2=hidden_2, 
        dropout_1=dropout_1, 
        dropout_2=dropout_2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = -1.0
    best_state = None
    best_metrics = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features).squeeze(-1)
                all_logits.append(logits)
                all_labels.append(labels)
                
        metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
        f1 = metrics["f1"]
        
        history.append({"epoch": epoch, "f1": f1})
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            best_metrics = metrics
            
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Save the best model for this trial temporarily to check later if it's the global best
    trial.set_user_attr("best_metrics", best_metrics)
    trial.set_user_attr("history", history)
    
    # Actually, we should just let optuna pick the best trial, then we save it at the end.
    # To do that, we can return the best validation F1.
    return best_f1

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CACHE_PATH.exists():
        print(f"Error: Cache not found at {CACHE_PATH}. Run the caching mechanism first.")
        sys.exit(1)

    print("Loading Dataset...")
    dataset = NeuralMSignalDataset(CACHE_PATH)
    train_indices, val_indices = split_train_val(dataset.labels, VAL_RATIO, SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create Optuna study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    print("Starting Optuna Hyperparameter Search (30 Trials)...")
    
    # We pass variables using lambda to the objective
    study.optimize(lambda trial: objective(trial, dataset, train_indices, val_indices, device), n_trials=30)
    
    print("\nSearch Complete!")
    best_trial = study.best_trial
    print(f"Best Trial F1: {best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Retrain the absolute best model completely (to save it properly)
    print("\nRetraining Best Model with optimal hyperparameters...")
    best_params = best_trial.params
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=best_params["batch_size"], shuffle=False)
    
    model = NeuralMSignalModel(
        input_dim=3072, 
        hidden_1=best_params["hidden_1"], 
        hidden_2=best_params["hidden_2"], 
        dropout_1=best_params["dropout_1"], 
        dropout_2=best_params["dropout_2"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = -1.0
    best_metrics = None
    history = []
    
    # We train for more epochs (50) for the final model
    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            
        train_loss = total_loss / len(train_indices)
        
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features).squeeze(-1)
                all_logits.append(logits)
                all_labels.append(labels)
                
        metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics
        })
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "hyperparameters": best_params
            }, OUTPUT_DIR / "neural_m_signal.pt")
            
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "best_val_f1": best_f1, 
            "history": history, 
            "hyperparameters": best_params,
            "metrics": best_metrics
        }, f, indent=2)
        
    print(f"\nFinal Best Val F1: {best_f1:.4f}. Model and metrics saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
