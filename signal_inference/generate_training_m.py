from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal_models.model_registry import MODEL_REGISTRY
from src.config import STAGE1_SIGNAL_CONFIG, TRAINING_DATA_CONFIG, get_device


REWRITE_KEYS = ("lex_sub", "comp_drop", "paraphrase", "literalize")


class LLMLoader:
    """Loads LLM-generated rewrites used by the Stage-1 idiom signal model."""

    def __init__(self, data_dir: Path) -> None:
        self.records = self._load_records(data_dir)

    def get_rewrites(self, sentence: str) -> dict[str, str]:
        sentence = sentence.strip()
        record = self.records.get(sentence)

        if record is None:
            return {key: sentence for key in REWRITE_KEYS}

        return {
            key: record.get(key, {}).get("text", sentence)
            for key in REWRITE_KEYS
        }

    @staticmethod
    def _load_records(data_dir: Path) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}

        for path in sorted(data_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    record = json.loads(line)
                    sentence = record.get("sentence", "").strip()

                    if sentence:
                        records[sentence] = record

        return records


def read_training_rows() -> list[dict[str, str]]:
    """Reads EN/PT train, dev, test, and xeval rows using the central config."""
    rows: list[dict[str, str]] = []

    for split_name, files in TRAINING_DATA_CONFIG.split_files.items():
        for language, split, tsv_path, _image_root in files:
            with tsv_path.open("r", encoding="utf-8") as file:
                reader = csv.DictReader(file, delimiter="\t")

                for index, row in enumerate(reader):
                    rows.append(
                        {
                            "sample_id": f"{language}_{split}_{index}",
                            "language": language,
                            "split": split_name,
                            "sentence": row["sentence"].strip(),
                            "compound": row["compound"].strip(),
                        }
                    )

            print(f"[READ] {language}/{split}: {tsv_path.name}")

    return rows


def mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pools transformer hidden states over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def load_transformer(device: torch.device) -> tuple[Any, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(STAGE1_SIGNAL_CONFIG.transformer_name)

    encoder = AutoModel.from_pretrained(
        STAGE1_SIGNAL_CONFIG.transformer_name,
    ).to(device)
    encoder.eval()

    return tokenizer, encoder


def load_signal_model(device: torch.device) -> tuple[Any, torch.nn.Module]:
    engine = MODEL_REGISTRY[STAGE1_SIGNAL_CONFIG.model_name]

    payload = torch.load(
        STAGE1_SIGNAL_CONFIG.weights_path,
        map_location=device,
        weights_only=False,
    )

    classifier = engine.build_classifier(
        payload["input_dim"],
        payload["hyperparameters"],
    ).to(device)

    classifier.load_state_dict(payload["classifier_state_dict"])
    classifier.eval()

    return engine, classifier


def build_batch_texts(
    batch: list[dict[str, str]],
    engine: Any,
    llm_loader: LLMLoader,
) -> list[str]:
    texts: list[str] = []

    for row in batch:
        texts.extend(
            engine.prepare_texts(
                row["language"],
                row["sentence"],
                row["compound"],
                llm_loader,
            )
        )

    return texts


def predict_batch(
    batch: list[dict[str, str]],
    engine: Any,
    classifier: torch.nn.Module,
    tokenizer: Any,
    encoder: torch.nn.Module,
    llm_loader: LLMLoader,
    device: torch.device,
) -> list[float]:
    batch_texts = build_batch_texts(batch, engine, llm_loader)
    num_inputs = len(batch_texts) // len(batch)

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=STAGE1_SIGNAL_CONFIG.max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hidden_states = encoder(**encoded).last_hidden_state
        pooled = mean_pool(hidden_states, encoded["attention_mask"])
        pooled = pooled.view(len(batch), num_inputs, -1)

        features = engine.get_features(pooled)
        logits = classifier(features).squeeze(-1)
        probabilities = torch.sigmoid(logits)

    return probabilities.detach().cpu().tolist()


def write_scores(
    rows: list[dict[str, str]],
    probabilities: list[float],
) -> Path:
    output_path = STAGE1_SIGNAL_CONFIG.train_scores_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "m_prob"],
            delimiter="\t",
        )
        writer.writeheader()

        for row, probability in zip(rows, probabilities):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "m_prob": f"{probability:.6f}",
                }
            )

    return output_path


def generate_training_m_scores(device: torch.device) -> Path:
    print("\n" + "=" * 70)
    print("[STAGE 1] Training idiom signal inference")
    print("=" * 70)
    print(f"Model       : {STAGE1_SIGNAL_CONFIG.model_name}")
    print(f"Transformer : {STAGE1_SIGNAL_CONFIG.transformer_name}")
    print(f"Weights     : {STAGE1_SIGNAL_CONFIG.weights_path}")
    print(f"LLM inputs  : {STAGE1_SIGNAL_CONFIG.train_llm_dir}")
    print(f"Output file : {STAGE1_SIGNAL_CONFIG.train_scores_path}")
    print("=" * 70)

    rows = read_training_rows()
    print(f"Loaded training rows: {len(rows)}")

    llm_loader = LLMLoader(STAGE1_SIGNAL_CONFIG.train_llm_dir)
    tokenizer, encoder = load_transformer(device)
    engine, classifier = load_signal_model(device)

    probabilities: list[float] = []

    for start in tqdm(
        range(0, len(rows), STAGE1_SIGNAL_CONFIG.batch_size),
        desc=STAGE1_SIGNAL_CONFIG.model_name,
        dynamic_ncols=True,
    ):
        batch = rows[start : start + STAGE1_SIGNAL_CONFIG.batch_size]
        probabilities.extend(
            predict_batch(
                batch=batch,
                engine=engine,
                classifier=classifier,
                tokenizer=tokenizer,
                encoder=encoder,
                llm_loader=llm_loader,
                device=device,
            )
        )

    output_path = write_scores(rows, probabilities)
    print(f"[OK] Saved training M scores: {output_path}")

    return output_path


def main() -> None:
    device = get_device()
    generate_training_m_scores(device)


if __name__ == "__main__":
    main()