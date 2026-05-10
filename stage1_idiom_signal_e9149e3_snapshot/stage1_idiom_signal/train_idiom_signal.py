from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.idiom_signal import IdiomSignalModel, build_idiom_input


REQUIRED_COLUMNS = {"Language", "MWE", "Target", "Label"}


class IdiomCsvDataset(Dataset):
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.rows = self._read_rows(self.csv_path)
        self.labels = [row["label"] for row in self.rows]

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, Any]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError("CSV file has no header row.")

            missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"CSV is missing required columns: {missing}")

            for line_number, row in enumerate(reader, start=2):
                language = (row.get("Language") or "").strip()
                mwe = (row.get("MWE") or "").strip()
                target = (row.get("Target") or "").strip()
                label_text = (row.get("Label") or "").strip()

                if label_text == "":
                    raise ValueError(f"Missing Label at CSV line {line_number}.")

                try:
                    label = float(label_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Label must be numeric 0 or 1 at CSV line {line_number}: {label_text}"
                    ) from exc

                if label not in {0.0, 1.0}:
                    raise ValueError(
                        f"Label must be binary 0 or 1 at CSV line {line_number}: {label_text}"
                    )

                rows.append(
                    {
                        "language": language,
                        "mwe": mwe,
                        "target": target,
                        "label": label,
                        "text": build_idiom_input(
                            sentence=target,
                            mwe=mwe,
                            language=language,
                        ),
                    }
                )

        if not rows:
            raise ValueError(f"CSV file contains no data rows: {csv_path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val(
    labels: list[float],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val_ratio must be in [0, 1).")

    indices = list(range(len(labels)))
    if val_ratio == 0.0 or len(indices) < 2:
        return indices, []

    rng = random.Random(seed)
    by_label: dict[float, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)

    can_stratify = len(by_label) > 1 and all(len(group) >= 2 for group in by_label.values())
    if can_stratify:
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

    shuffled = indices[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def collate_idiom_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
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
    encoded["labels"] = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.float,
    )
    encoded["texts"] = texts
    return encoded


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = batch["labels"].to(device)
    model_inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    return model_inputs, labels


def compute_metrics(
    probabilities: list[float],
    labels: list[float],
) -> dict[str, float | None]:
    predictions = [1.0 if probability >= 0.5 else 0.0 for probability in probabilities]

    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    accuracy = correct / max(1, len(labels))

    tp = sum(
        prediction == 1.0 and label == 1.0
        for prediction, label in zip(predictions, labels)
    )
    fp = sum(
        prediction == 1.0 and label == 0.0
        for prediction, label in zip(predictions, labels)
    )
    fn = sum(
        prediction == 0.0 and label == 1.0
        for prediction, label in zip(predictions, labels)
    )

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    roc_auc: float | None = None
    if len(set(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(labels, probabilities))
        except (ImportError, ValueError):
            roc_auc = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }


def evaluate(
    model: IdiomSignalModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_probabilities: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            model_inputs, labels = move_batch_to_device(batch=batch, device=device)
            logits = model(**model_inputs).squeeze(-1)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            probabilities = torch.sigmoid(logits).detach().cpu().tolist()
            all_probabilities.extend(float(value) for value in probabilities)
            all_labels.extend(float(value) for value in labels.detach().cpu().tolist())

    metrics = compute_metrics(probabilities=all_probabilities, labels=all_labels)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_checkpoint(
    model: IdiomSignalModel,
    output_dir: Path,
    epoch: int,
    metrics: dict[str, float | None],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        output_dir / "idiom_signal.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 idiomaticity module M.")
    parser.add_argument("--csv_path", required=True, help="Path to idiom CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory for saved model files.")
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = IdiomCsvDataset(args.csv_path)
    train_indices, val_indices = split_train_val(
        labels=dataset.labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdiomSignalModel(
        model_name=args.model_name,
        freeze_encoder=not args.unfreeze_encoder,
    ).to(device)

    collate_fn = lambda batch: collate_idiom_batch(
        batch=batch,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        if val_indices
        else None
    )

    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "dropout": model.dropout,
        "encoder_frozen_during_training": not args.unfreeze_encoder,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "seed": args.seed,
    }
    save_json(output_dir / "config.json", config)
    model.tokenizer.save_pretrained(output_dir / "tokenizer")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0

    print(
        f"Loaded {len(dataset)} rows; train={len(train_indices)} val={len(val_indices)} "
        f"device={device} encoder_frozen={not args.unfreeze_encoder}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0

        for batch in train_loader:
            model_inputs, labels = move_batch_to_device(batch=batch, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(**model_inputs).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_train_loss += float(loss.item()) * batch_size
            total_train_examples += batch_size

        train_loss = total_train_loss / max(1, total_train_examples)
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
        }

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            epoch_metrics["val"] = val_metrics
            val_f1 = float(val_metrics["f1"] or 0.0)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                save_checkpoint(
                    model=model,
                    output_dir=output_dir,
                    epoch=epoch,
                    metrics=val_metrics,
                )

            roc_auc = val_metrics["roc_auc"]
            roc_auc_text = "NA" if roc_auc is None else f"{roc_auc:.4f}"
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={float(val_metrics['loss']):.4f} "
                f"val_acc={float(val_metrics['accuracy']):.4f} "
                f"val_f1={float(val_metrics['f1']):.4f} "
                f"val_roc_auc={roc_auc_text}"
            )
        else:
            save_checkpoint(
                model=model,
                output_dir=output_dir,
                epoch=epoch,
                metrics={"loss": None, "accuracy": None, "f1": None, "roc_auc": None},
            )
            best_epoch = epoch
            print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f}")

        history.append(epoch_metrics)

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_f1": None if best_f1 < 0 else best_f1,
        "history": history,
    }
    save_json(output_dir / "metrics.json", metrics_payload)

    print(f"Saved best checkpoint to {output_dir / 'idiom_signal.pt'}")
    print(f"Saved tokenizer/config/metrics under {output_dir}")


if __name__ == "__main__":
    main()
