from __future__ import annotations

import csv
import gc
import json
import random
import re
from pathlib import Path
from statistics import mean
from typing import Any

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "stage1_idiom_signal/data/unique_mwe_target_label_language.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_NAME = "xlm-roberta-base"
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def mark_mwe_in_sentence(sentence: str, mwe: str) -> str:
    if not mwe:
        return sentence
    match = re.search(re.escape(mwe), sentence, flags=re.IGNORECASE)
    if match is None:
        return sentence
    s, e = match.span()
    return f"{sentence[:s]}<mwe>{sentence[s:e]}</mwe>{sentence[e:]}"


def build_input(row: dict[str, Any]) -> str:
    marked = mark_mwe_in_sentence(row["target"], row["mwe"])
    return f"[LANG={row['language']}] {marked} [MWE] {row['mwe']}"


def read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "language": (row.get("Language") or "").strip(),
                "mwe": (row.get("MWE") or "").strip(),
                "target": (row.get("Target") or "").strip(),
                "label": float((row.get("Label") or "").strip()),
            })
    return rows


def split_train_val(labels: list[float], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[float, list[int]] = {}
    for idx, lbl in enumerate(labels):
        by_label.setdefault(lbl, []).append(idx)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for group in by_label.values():
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
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    roc_auc = float(roc_auc_score(lbls, probs)) if len(set(lbls)) > 1 else None
    return {
        "accuracy": float(accuracy_score(lbls, preds)),
        "precision": float(precision_score(lbls, preds, zero_division=0)),
        "recall": float(recall_score(lbls, preds, zero_division=0)),
        "f1": float(f1_score(lbls, preds, zero_division=0)),
        "roc_auc": roc_auc,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_texts(texts: list[str], model_name: str, max_length: int, device: torch.device) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch_texts = texts[start : start + ENCODE_BATCH_SIZE]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device)).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"].to(device))
        chunks.append(pooled.cpu())
    del encoder
    cleanup_memory(device)
    return torch.cat(chunks, dim=0)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: list[float]) -> None:
        self.embeddings = embeddings.float()
        self.labels = torch.tensor(labels, dtype=torch.float)
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


def train_single_seed(
    dataset: EmbeddingDataset, seed: int, device: torch.device,
    lr: float, weight_decay: float, batch_size: int, dropout: float,
    epochs: int,
    save_path: Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    train_indices, val_indices = split_train_val(dataset.labels.tolist(), VAL_RATIO, seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    input_dim = dataset.embeddings.size(-1)
    classifier = nn.Sequential(
        nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(input_dim // 2, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_f1, best_epoch, best_metrics, best_state = -1.0, 0, None, None

    for epoch in range(1, epochs + 1):
        classifier.train()
        for embs, lbls in train_loader:
            embs, lbls = embs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(embs).squeeze(-1), lbls)
            loss.backward()
            optimizer.step()

        classifier.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for embs, lbls in val_loader:
                embs, lbls = embs.to(device), lbls.to(device)
                all_logits.append(classifier(embs).squeeze(-1))
                all_labels.append(lbls)

        metrics = compute_metrics(torch.sigmoid(torch.cat(all_logits)), torch.cat(all_labels))
        if metrics["f1"] > best_f1:
            best_f1, best_epoch, best_metrics = metrics["f1"], epoch, metrics
            if save_path is not None:
                best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if save_path is not None and best_state is not None:
        torch.save({"classifier_state_dict": best_state, "input_dim": input_dim,
            "hyperparameters": {"dropout": dropout}, "seed": seed, "best_epoch": best_epoch, "metrics": best_metrics}, save_path)

    classifier.to("cpu")
    del classifier, optimizer, criterion, train_loader, val_loader
    cleanup_memory(device)
    return {"seed": seed, "best_epoch": best_epoch, "best_val_f1": best_f1, "best_metrics": best_metrics}


def objective(trial: optuna.Trial, dataset: EmbeddingDataset, device: torch.device) -> float:
    params = {
        "lr": trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
    }
    seed_f1s = [
        train_single_seed(
            dataset=dataset,
            seed=seed,
            device=device,
            epochs=OPTUNA_EPOCHS,
            **params,
        )["best_val_f1"]
        for seed in OPTUNA_SEEDS
    ]
    mean_f1 = mean(seed_f1s)
    print(f"  trial={trial.number}  mean_f1={mean_f1:.4f}  per_seed={[f'{f:.4f}' for f in seed_f1s]}")
    return mean_f1


def main() -> None:
    set_seed(OPTUNA_SEEDS[0])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("Loading CSV...")
    rows = read_csv(CSV_PATH)
    texts = [build_input(row) for row in rows]
    labels = [row["label"] for row in rows]

    print("Encoding marked sentences with XLM-R...")
    embeddings = encode_texts(texts, MODEL_NAME, MAX_LENGTH, device)
    dataset = EmbeddingDataset(embeddings, labels)
    print(f"  {len(dataset)} samples, dim={dataset.embeddings.size(-1)}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEEDS[0]))
    print(f"Starting Optuna ({N_TRIALS} trials, seeds={OPTUNA_SEEDS}, epochs={OPTUNA_EPOCHS})...")
    study.optimize(lambda trial: objective(trial, dataset, device), n_trials=N_TRIALS)

    best_params = study.best_trial.params
    print(f"\nBest mean F1: {study.best_trial.value:.4f}")

    print("\nCollecting per-seed results...")
    per_seed = [
        train_single_seed(
            dataset=dataset,
            seed=s,
            device=device,
            epochs=FINAL_EPOCHS,
            **best_params,
        )
        for s in FINAL_SEEDS
    ]
    mean_f1 = mean(r["best_val_f1"] for r in per_seed)
    best_run = max(per_seed, key=lambda r: r["best_val_f1"])

    print(f"Saving best model (seed={best_run['seed']}, f1={best_run['best_val_f1']:.4f})...")
    train_single_seed(dataset=dataset, seed=best_run["seed"], device=device, epochs=FINAL_EPOCHS, save_path=OUTPUT_DIR / "idiom_signal.pt", **best_params)

    save_json(OUTPUT_DIR / "config.json", {
        "model_name": MODEL_NAME, "max_length": MAX_LENGTH,
        "input_format": "[LANG] + <mwe>tagged</mwe> sentence + [MWE]", "head_type": "simple_relu",
        "encoder_frozen": True, "hyperparameters": best_params,
        "optuna_seeds": list(OPTUNA_SEEDS), "final_seeds": list(FINAL_SEEDS),
        "optuna_epochs": OPTUNA_EPOCHS, "final_epochs": FINAL_EPOCHS,
    })
    save_json(OUTPUT_DIR / "metrics.json", {
        "name": "marked_simple", "seeds": list(FINAL_SEEDS), "hyperparameters": best_params,
        "mean_val_f1": mean_f1, "best_seed": best_run["seed"], "best_val_f1": best_run["best_val_f1"],
        "best_epoch": best_run["best_epoch"], "best_metrics": best_run["best_metrics"],
        "per_seed": [{"seed": r["seed"], "best_epoch": r["best_epoch"], "best_val_f1": r["best_val_f1"], "best_metrics": r["best_metrics"]} for r in per_seed],
    })
    print(f"\nDone. Mean val F1: {mean_f1:.4f}")
    for r in per_seed:
        print(f"  seed={r['seed']}  f1={r['best_val_f1']:.4f}  epoch={r['best_epoch']}")


if __name__ == "__main__":
    main()
