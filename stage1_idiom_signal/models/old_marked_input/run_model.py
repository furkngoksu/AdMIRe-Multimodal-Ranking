from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = REPO_ROOT / "stage1_idiom_signal/data/unique_mwe_target_label_language.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REQUIRED_COLUMNS = {"Language", "MWE", "Target", "Label"}


def mark_mwe_in_sentence(sentence: str, mwe: str) -> str:
    if not mwe:
        return sentence

    match = re.search(re.escape(mwe), sentence, flags=re.IGNORECASE)
    if match is None:
        return sentence

    start, end = match.span()
    return f"{sentence[:start]}<mwe>{sentence[start:end]}</mwe>{sentence[end:]}"


def build_old_marked_input(row: dict[str, Any]) -> str:
    marked = mark_mwe_in_sentence(row["target"], row["mwe"])
    return f"[LANG={row['language']}] {marked} [MWE] {row['mwe']}"


def build_new_target_only_input(row: dict[str, Any]) -> str:
    return row["target"]


def build_plain_sentence_mwe_input(row: dict[str, Any]) -> str:
    return f"[LANG={row['language']}] {row['target']} [MWE] {row['mwe']}"


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


class IdiomTextDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        text_builder: Callable[[dict[str, Any]], str],
    ) -> None:
        self.items = [
            {
                **row,
                "text": text_builder(row),
            }
            for row in rows
        ]
        self.labels = [float(item["label"]) for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_text_batch(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, Any]:
    texts = [item["text"] for item in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float)
    return encoded


class XlmrTextClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        head_type: str,
        dropout: float,
        freeze_encoder: bool,
        hidden_1: int,
        hidden_2: int,
        dropout_1: float,
        dropout_2: float,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_frozen = freeze_encoder

        hidden_dim = int(self.encoder.config.hidden_size)
        if head_type == "old_simple":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif head_type == "new_mlp":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_1),
                nn.BatchNorm1d(hidden_1),
                nn.GELU(),
                nn.Dropout(dropout_1),
                nn.Linear(hidden_1, hidden_2),
                nn.BatchNorm1d(hidden_2),
                nn.GELU(),
                nn.Dropout(dropout_2),
                nn.Linear(hidden_2, 1),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        for parameter in self.encoder.parameters():
            parameter.requires_grad = not freeze_encoder
        if freeze_encoder:
            self.encoder.eval()

    def train(self, mode: bool = True) -> "XlmrTextClassifier":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    @staticmethod
    def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder_frozen:
            with torch.no_grad():
                hidden_states = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        pooled = self.mean_pool(hidden_states, attention_mask)
        return self.classifier(pooled).squeeze(-1)


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    labels = batch["labels"].to(device)
    return inputs, labels


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


def evaluate(
    model: XlmrTextClassifier,
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
        for batch in loader:
            inputs, labels = move_batch(batch, device)
            logits = model(**inputs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_probs, all_labels)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_training(
    *,
    name: str,
    csv_path: str | Path,
    output_dir: str | Path,
    text_builder: Callable[[dict[str, Any]], str],
    input_format: str,
    head_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    val_ratio: float,
    seed: int,
    limit: int,
    model_name: str,
    unfreeze_encoder: bool,
    hidden_1: int = 512,
    hidden_2: int = 128,
    dropout: float = 0.1,
    dropout_1: float = 0.2,
    dropout_2: float = 0.2,
) -> dict[str, Any]:
    set_seed(seed)
    rows = apply_limit(read_idiom_csv(csv_path), limit=limit, seed=seed)
    dataset = IdiomTextDataset(rows, text_builder=text_builder)
    train_indices, val_indices = stratified_split(dataset.labels, val_ratio=val_ratio, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XlmrTextClassifier(
        model_name=model_name,
        head_type=head_type,
        dropout=dropout,
        freeze_encoder=not unfreeze_encoder,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
    ).to(device)

    if head_type == "new_mlp" and batch_size < 2:
        raise ValueError("new_mlp uses BatchNorm, so batch_size must be at least 2.")

    drop_last_train_batch = (
        head_type == "new_mlp"
        and len(train_indices) > batch_size
        and len(train_indices) % batch_size == 1
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train_batch,
        collate_fn=lambda batch: collate_text_batch(batch, model.tokenizer, max_length),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_text_batch(batch, model.tokenizer, max_length),
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, float | None] | None = None
    history: list[dict[str, float | int | None]] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0

        for batch in train_loader:
            inputs, labels = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_train_loss += float(loss.item()) * batch_size_actual
            total_train_examples += batch_size_actual

        train_loss = total_train_loss / max(1, total_train_examples)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_record)

        print(
            f"{name} epoch={epoch}/{epochs} "
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
                    "name": name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": val_metrics,
                },
                output_path / "model.pt",
            )

    payload = {
        "name": name,
        "input_format": input_format,
        "head_type": head_type,
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "val_ratio": val_ratio,
        "seed": seed,
        "limit": limit,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_metrics": best_metrics,
        "history": history,
    }
    save_json(output_path / "metrics.json", payload)

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def print_metric_table(results: list[dict[str, Any]]) -> None:
    print("")
    print("model\tbest_epoch\tval_f1\tval_acc\tval_auc\tinput")
    for result in results:
        metrics = result.get("best_metrics") or {}
        auc = metrics.get("roc_auc")
        auc_text = "None" if auc is None else f"{float(auc):.4f}"
        print(
            f"{result['name']}\t"
            f"{result['best_epoch']}\t"
            f"{float(result['best_val_f1']):.4f}\t"
            f"{float(metrics.get('accuracy', 0.0)):.4f}\t"
            f"{auc_text}\t"
            f"{result['input_format']}"
        )


def train_old_marked_input(
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_name: str = "xlm-roberta-base",
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_length: int = 128,
    val_ratio: float = 0.15,
    seed: int = 42,
    limit: int = 512,
    unfreeze_encoder: bool = False,
) -> dict[str, Any]:
    return run_training(
        name="old_marked_input",
        csv_path=csv_path,
        output_dir=output_dir,
        text_builder=build_old_marked_input,
        input_format="[LANG] + sentence with <mwe> tags + [MWE]",
        head_type="old_simple",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        val_ratio=val_ratio,
        seed=seed,
        limit=limit,
        model_name=model_name,
        unfreeze_encoder=unfreeze_encoder,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the old marked-input Stage 1 baseline.")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=512, help="Use 0 for the full CSV.")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_old_marked_input(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit=args.limit,
        unfreeze_encoder=args.unfreeze_encoder,
    )


if __name__ == "__main__":
    main()
