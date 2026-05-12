from __future__ import annotations
import gc, json, random
from pathlib import Path
from statistics import mean
from typing import Any
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

# --- CONFIG ---
REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHED_PATH = REPO_ROOT / "stage1_idiom_signal/data/matched_records.json"
MODEL_NAME = "xlm-roberta-base"
clean_model_name = MODEL_NAME.split("/")[-1].replace("-roberta-", "_").replace("-", "_")
OUTPUT_DIR = Path(__file__).resolve().parent / f"outputs_{clean_model_name}"
OPTUNA_SEEDS = (42, 100, 150)
FINAL_SEEDS = (42, 100, 150, 200, 250)
OPTUNA_EPOCHS = 30
FINAL_EPOCHS = 40
N_TRIALS = 30
MAX_LENGTH = 128
VAL_RATIO = 0.15
ENCODE_BATCH_SIZE = 16

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def cleanup_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    elif device.type == "mps": torch.mps.empty_cache()

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from stage1_idiom_signal.model_registry import split_train_val_by_mwe

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

def encode_texts(texts: list[str], model_name: str, max_length: int, device: torch.device) -> torch.Tensor:
    cache_dir = REPO_ROOT / "stage1_idiom_signal/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cache_{clean_model_name}_baseline.pt"
    if cache_path.exists():
        print(f"Loading cached embeddings from: {cache_path}")
        return torch.load(cache_path)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name).to(device); encoder.eval(); chunks = []
    for start in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch_texts = texts[start : start + ENCODE_BATCH_SIZE]
        if start == 0:
            print("\n--- DEBUG: BASELINE_MLP INPUT ---")
            for t in batch_texts[:3]: print(f"Sample: {t}")
            print("---------------------------------\n")
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device)).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"].to(device))
        chunks.append(pooled.cpu())
    del encoder; cleanup_memory(device)
    res = torch.cat(chunks, dim=0)
    print(f"Saving cached embeddings to: {cache_path}")
    torch.save(res, cache_path)
    return res

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: list[float], mwes: list[str]):
        self.embeddings = embeddings.float(); self.labels = torch.tensor(labels, dtype=torch.float); self.mwes = mwes
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.embeddings[idx], self.labels[idx]

def train_single_seed(dataset, seed, device, hidden_1, hidden_2, dropout_1, dropout_2, lr, weight_decay, batch_size, epochs, save_path=None):
    set_seed(seed); train_indices, val_indices = split_train_val_by_mwe(dataset.mwes, dataset.labels.tolist(), VAL_RATIO, seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    input_dim = dataset.embeddings.size(-1)
    classifier = nn.Sequential(
        nn.Linear(input_dim, hidden_1), nn.BatchNorm1d(hidden_1), nn.GELU(), nn.Dropout(dropout_1),
        nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2), nn.GELU(), nn.Dropout(dropout_2),
        nn.Linear(hidden_2, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay); criterion = nn.BCEWithLogitsLoss()
    best_f1, best_epoch, best_metrics, best_state = -1.0, 0, None, None
    patience, patience_counter = 5, 0
    history = []
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_train_loss, total_train_samples = 0.0, 0
        for embs, lbls in train_loader:
            embs, lbls = embs.to(device), lbls.to(device); optimizer.zero_grad(); logits = classifier(embs).squeeze(-1); loss = criterion(logits, lbls); loss.backward(); torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0); optimizer.step()
            total_train_loss += loss.item() * len(lbls); total_train_samples += len(lbls)
        classifier.eval(); all_logits, all_labels = [], []
        total_val_loss, total_val_samples = 0.0, 0
        with torch.no_grad():
            for embs, lbls in val_loader:
                embs, lbls = embs.to(device), lbls.to(device); logits = classifier(embs).squeeze(-1); loss = criterion(logits, lbls); total_val_loss += loss.item() * len(lbls); total_val_samples += len(lbls); all_logits.append(logits); all_labels.append(lbls)
        from sklearn.metrics import accuracy_score, f1_score
        probs = torch.sigmoid(torch.cat(all_logits)).cpu().numpy(); lbls = torch.cat(all_labels).cpu().numpy(); preds = (probs >= 0.5).astype(float); f1 = f1_score(lbls, preds, zero_division=0); acc = accuracy_score(lbls, preds)
        history.append({"epoch": epoch, "train_loss": total_train_loss / max(1, total_train_samples), "val_loss": total_val_loss / max(1, total_val_samples), "val_f1": float(f1), "val_acc": float(acc)})
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch; best_metrics = {"f1": f1, "accuracy": acc}
            if save_path: best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if save_path and best_state:
        torch.save({"classifier_state_dict": best_state, "input_dim": input_dim, "hyperparameters": {"hidden_1": hidden_1, "hidden_2": hidden_2, "dropout_1": dropout_1, "dropout_2": dropout_2}, "seed": seed, "best_epoch": best_epoch, "metrics": best_metrics}, save_path)
    del classifier, optimizer; cleanup_memory(device)
    return {"seed": seed, "best_epoch": best_epoch, "best_val_f1": best_f1, "best_metrics": best_metrics, "history": history}

def objective(trial, dataset, device):
    params = {"hidden_1": trial.suggest_categorical("hidden_1", [256, 512, 1024]), "hidden_2": trial.suggest_categorical("hidden_2", [64, 128, 256]), "dropout_1": trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1), "dropout_2": trial.suggest_float("dropout_2", 0.1, 0.5, step=0.1), "lr": trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3]), "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2]), "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64])}
    return mean([train_single_seed(dataset, seed, device, epochs=OPTUNA_EPOCHS, **params)["best_val_f1"] for seed in OPTUNA_SEEDS])

def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(payload, f, indent=2, ensure_ascii=False)

def main():
    set_seed(OPTUNA_SEEDS[0]); OUTPUT_DIR.mkdir(parents=True, exist_ok=True); device = get_device()
    with MATCHED_PATH.open("r", encoding="utf-8") as f: records = json.load(f)
    texts = [r["target"] for r in records]; labels = [float(r["label"]) for r in records]; mwes = [r["mwe"] for r in records]
    embeddings = encode_texts(texts, MODEL_NAME, MAX_LENGTH, device); dataset = EmbeddingDataset(embeddings, labels, mwes)
    study = optuna.create_study(direction="maximize"); study.optimize(lambda t: objective(t, dataset, device), n_trials=N_TRIALS)
    best_params = study.best_trial.params; per_seed = [train_single_seed(dataset, s, device, epochs=FINAL_EPOCHS, **best_params) for s in FINAL_SEEDS]
    best_run = max(per_seed, key=lambda r: r["best_val_f1"])
    final_res = train_single_seed(dataset, best_run["seed"], device, epochs=FINAL_EPOCHS, save_path=OUTPUT_DIR / "idiom_signal.pt", **best_params)
    save_json(OUTPUT_DIR / "history.json", final_res["history"])
    save_json(OUTPUT_DIR / "config.json", {"model_name": MODEL_NAME, "input_format": "Pure Raw Sentence (Baseline)", "hyperparameters": best_params, "final_seeds": list(FINAL_SEEDS), "optuna_epochs": OPTUNA_EPOCHS, "final_epochs": FINAL_EPOCHS})
    save_json(OUTPUT_DIR / "metrics.json", {"name": "baseline_mlp", "mean_val_f1": mean(r["best_val_f1"] for r in per_seed), "best_seed": best_run["seed"], "best_val_f1": best_run["best_val_f1"], "best_metrics": best_run["best_metrics"], "hyperparameters": best_params, "per_seed": per_seed})
    print(f"Done. Mean F1: {mean(r['best_val_f1'] for r in per_seed):.4f}")

if __name__ == "__main__": main()
