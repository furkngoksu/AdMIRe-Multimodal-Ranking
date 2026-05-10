from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = REPO_ROOT / "stage1_idiom_signal/data/unique_mwe_target_label_language.csv"
DEFAULT_LLM_RESULTS_DIR = REPO_ROOT / "llm_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REQUIRED_COLUMNS = {"Language", "MWE", "Target", "Label"}
TEXT_KEYS = ["target", "paraphrase", "lex_sub", "comp_drop", "literalize"]
LANG_CODE_MAP = {"English": "EN", "Portuguese": "PT"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_idiom_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            label_text = (row.get("Label") or "").strip()
            if label_text == "":
                raise ValueError(f"Missing Label at CSV line {line_number}.")

            label = float(label_text)
            if label not in {0.0, 1.0}:
                raise ValueError(f"Label must be 0 or 1 at CSV line {line_number}.")

            rows.append(
                {
                    "language": (row.get("Language") or "").strip(),
                    "mwe": (row.get("MWE") or "").strip(),
                    "target": (row.get("Target") or "").strip(),
                    "label": label,
                }
            )

    if not rows:
        raise ValueError(f"CSV file contains no rows: {path}")
    return rows


def apply_limit(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows

    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[:limit]


def stratified_split(labels: list[float], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    if val_ratio == 0.0 or len(labels) < 2:
        return list(range(len(labels))), []

    by_label: dict[float, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(float(label), []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for group in by_label.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_metrics(probabilities: list[float], labels: list[float]) -> dict[str, float | None]:
    predictions = [1.0 if value >= 0.5 else 0.0 for value in probabilities]
    total = max(1, len(labels))
    correct = sum(pred == label for pred, label in zip(predictions, labels))

    tp = sum(pred == 1.0 and label == 1.0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1.0 and label == 0.0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0.0 and label == 1.0 for pred, label in zip(predictions, labels))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    roc_auc: float | None = None
    if len(set(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(labels, probabilities))
        except (ImportError, ValueError):
            roc_auc = None

    return {
        "accuracy": float(correct / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def token_set(text: str) -> set[str]:
    return set(text.lower().split())


def read_llm_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def match_llm_records(
    *,
    csv_rows: list[dict[str, Any]],
    llm_results_dir: Path,
    max_matches: int | None = None,
) -> list[dict[str, Any]]:
    all_matched: list[dict[str, Any]] = []

    for jsonl_path in sorted(llm_results_dir.glob("llm_*.jsonl")):
        language_name = jsonl_path.stem.split("_", 1)[1]
        language_code = LANG_CODE_MAP.get(language_name, language_name)
        csv_for_language = [row for row in csv_rows if row["language"] == language_code]
        target_tokens = [token_set(row["target"]) for row in csv_for_language]

        if not csv_for_language:
            continue

        llm_records = read_llm_jsonl(jsonl_path)
        for item in tqdm(llm_records, desc=f"Matching {jsonl_path.name}", leave=False):
            value = item.get("value") or {}
            required = ["lex_sub", "comp_drop", "paraphrase", "literalize"]
            if not all(key in value and "text" in value[key] for key in required):
                continue

            lex_sub_text = str(value["lex_sub"]["text"])
            lex_tokens = token_set(lex_sub_text)
            if not lex_tokens:
                continue

            best_score = 0.0
            best_index = -1
            for index, candidate_tokens in enumerate(target_tokens):
                union = lex_tokens | candidate_tokens
                if not union:
                    continue
                score = len(lex_tokens & candidate_tokens) / len(union)
                if score > best_score:
                    best_score = score
                    best_index = index

            if best_index == -1 or best_score <= 0.4:
                continue

            row = csv_for_language[best_index]
            all_matched.append(
                {
                    "key": item.get("key"),
                    "language": row["language"],
                    "mwe": row["mwe"],
                    "label": row["label"],
                    "target": row["target"],
                    "paraphrase": str(value["paraphrase"]["text"]),
                    "lex_sub": lex_sub_text,
                    "comp_drop": str(value["comp_drop"]["text"]),
                    "literalize": str(value["literalize"]["text"]),
                    "match_score": best_score,
                }
            )
            if max_matches is not None and len(all_matched) >= max_matches:
                return all_matched

    return all_matched


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_neural_m_features(
    *,
    csv_path: Path,
    llm_results_dir: Path,
    model_name: str,
    max_length: int,
    encode_batch_size: int,
    limit: int,
    seed: int,
) -> dict[str, Any]:
    rows = read_idiom_csv(csv_path)
    matched = match_llm_records(
        csv_rows=rows,
        llm_results_dir=llm_results_dir,
        max_matches=limit if limit > 0 else None,
    )
    matched = apply_limit(matched, limit=limit, seed=seed)
    if not matched:
        raise ValueError("No LLM perturbation records matched the CSV.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    pooled_chunks: list[torch.Tensor] = []
    num_texts = len(TEXT_KEYS)

    for start in tqdm(range(0, len(matched), encode_batch_size), desc="Encoding Neural M features"):
        batch = matched[start : start + encode_batch_size]
        flat_texts = [record[key] for record in batch for key in TEXT_KEYS]
        encoded = tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            hidden_states = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = mean_pool(hidden_states, attention_mask)

        pooled_chunks.append(pooled.view(len(batch), num_texts, -1).cpu().to(torch.float16))

    features = {
        "pooled": torch.cat(pooled_chunks, dim=0),
        "metadata": [
            {
                "key": record["key"],
                "target": record["target"],
                "mwe": record["mwe"],
                "language": record["language"],
                "label": record["label"],
                "match_score": record["match_score"],
            }
            for record in matched
        ],
        "text_keys_order": TEXT_KEYS,
        "model_name": model_name,
        "max_length": max_length,
        "limit": limit,
        "seed": seed,
        "num_examples": len(matched),
    }

    encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features


class NeuralMDataset(Dataset):
    def __init__(self, features: dict[str, Any]) -> None:
        pooled = features["pooled"].float()
        text_keys = features["text_keys_order"]
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
        self.labels = torch.tensor(
            [float(item["label"]) for item in features["metadata"]],
            dtype=torch.float,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class NeuralMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, dropout_1: float, dropout_2: float) -> None:
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
            nn.Linear(hidden_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


def evaluate_neural_m(
    model: NeuralMModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_probs, all_labels)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def train_neural_m(
    *,
    features: dict[str, Any],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_ratio: float,
    seed: int,
    hidden_1: int,
    hidden_2: int,
    dropout_1: float,
    dropout_2: float,
) -> dict[str, Any]:
    set_seed(seed)
    dataset = NeuralMDataset(features)
    train_indices, val_indices = stratified_split(dataset.labels.tolist(), val_ratio=val_ratio, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if batch_size < 2:
        raise ValueError("Neural M uses BatchNorm, so batch_size must be at least 2.")

    drop_last_train_batch = len(train_indices) > batch_size and len(train_indices) % batch_size == 1
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train_batch,
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    model = NeuralMModel(
        input_dim=dataset.features.size(-1),
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, float | None] | None = None
    history: list[dict[str, float | int | None]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual

        train_loss = total_loss / max(1, total_examples)
        val_metrics = evaluate_neural_m(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

        print(
            f"neural_m epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={float(val_metrics['loss']):.4f} "
            f"val_acc={float(val_metrics['accuracy']):.4f} "
            f"val_f1={float(val_metrics['f1']):.4f} "
            f"val_roc_auc={val_metrics['roc_auc']}"
        )

        current_f1 = float(val_metrics["f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(
                {
                    "name": "neural_m",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": val_metrics,
                },
                output_dir / "model.pt",
            )

    payload = {
        "name": "neural_m",
        "input_format": "abs(target - perturbation) vectors: sub/comp/lit/paraphrase",
        "head_type": "new_mlp_3072",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_ratio": val_ratio,
        "seed": seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "num_examples": len(dataset),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_metrics": best_metrics,
        "history": history,
    }
    save_json(output_dir / "metrics.json", payload)

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lr": trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2]),
        "hidden_1": trial.suggest_categorical("hidden_1", [256, 512, 1024]),
        "hidden_2": trial.suggest_categorical("hidden_2", [64, 128, 256]),
        "dropout_1": trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1),
        "dropout_2": trial.suggest_float("dropout_2", 0.1, 0.5, step=0.1),
    }


def tune_hyperparameters(
    *,
    features: dict[str, Any],
    output_dir: Path,
    tune_epochs: int,
    val_ratio: float,
    seeds: list[int],
    n_trials: int,
) -> tuple[dict[str, Any], optuna.Study]:
    import optuna

    trial_seed = seeds[0]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        result = train_neural_m(
            features=features,
            output_dir=output_dir / "optuna_trials" / f"trial_{trial.number}",
            epochs=tune_epochs,
            val_ratio=val_ratio,
            seed=trial_seed,
            **params,
        )
        score = float(result["best_val_f1"])
        trial.report(score, tune_epochs)
        return score

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_trial.params), study


def run_seed_sweep(
    *,
    features: dict[str, Any],
    output_dir: Path,
    final_epochs: int,
    val_ratio: float,
    seeds: list[int],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        results.append(
            train_neural_m(
                features=features,
                output_dir=output_dir / f"seed_{seed}",
                epochs=final_epochs,
                val_ratio=val_ratio,
                seed=seed,
                **params,
            )
        )
    return results


def summarize_results(
    *,
    output_dir: Path,
    best_params: dict[str, Any],
    study: optuna.Study | None,
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    f1_scores = [float(result["best_val_f1"]) for result in seed_results]
    best_result = max(seed_results, key=lambda result: float(result["best_val_f1"]))
    mean_f1 = sum(f1_scores) / max(1, len(f1_scores))
    payload = {
        "name": "neural_m",
        "best_params": best_params,
        "best_seed": best_result["seed"],
        "best_val_f1": best_result["best_val_f1"],
        "mean_val_f1": mean_f1,
        "std_val_f1": (sum((score - mean_f1) ** 2 for score in f1_scores) / len(f1_scores)) ** 0.5
        if f1_scores
        else 0.0,
        "seed_results": seed_results,
    }
    if study is not None:
        payload["optuna"] = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_trial.value,
            "trials": [
                {"number": trial.number, "value": trial.value, "params": trial.params, "state": str(trial.state)}
                for trial in study.trials
            ],
        }
    save_json(output_dir / "metrics.json", payload)
    save_json(output_dir / "best_params.json", best_params)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM perturbation-vector Neural M baseline.")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--llm_results_dir", default=str(DEFAULT_LLM_RESULTS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--final_epochs", type=int, default=None)
    parser.add_argument("--tune_epochs", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--limit", type=int, default=512, help="Use 0 for all matched LLM records.")
    parser.add_argument("--hidden_1", type=int, default=512)
    parser.add_argument("--hidden_2", type=int, default=128)
    parser.add_argument("--dropout_1", type=float, default=0.2)
    parser.add_argument("--dropout_2", type=float, default=0.2)
    parser.add_argument("--skip_tuning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    final_epochs = args.final_epochs if args.final_epochs is not None else args.epochs
    features = build_neural_m_features(
        csv_path=Path(args.csv_path),
        llm_results_dir=Path(args.llm_results_dir),
        model_name=args.model_name,
        max_length=args.max_length,
        encode_batch_size=args.encode_batch_size,
        limit=args.limit,
        seed=args.seeds[0],
    )
    if args.skip_tuning:
        best_params = {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_1": args.hidden_1,
            "hidden_2": args.hidden_2,
            "dropout_1": args.dropout_1,
            "dropout_2": args.dropout_2,
        }
        study = None
    else:
        best_params, study = tune_hyperparameters(
            features=features,
            output_dir=output_dir,
            tune_epochs=args.tune_epochs,
            val_ratio=args.val_ratio,
            seeds=args.seeds,
            n_trials=args.n_trials,
        )

    seed_results = run_seed_sweep(
        features=features,
        output_dir=output_dir,
        final_epochs=final_epochs,
        val_ratio=args.val_ratio,
        seeds=args.seeds,
        params=best_params,
    )
    summarize_results(output_dir=output_dir, best_params=best_params, study=study, seed_results=seed_results)


if __name__ == "__main__":
    main()
