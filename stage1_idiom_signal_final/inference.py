from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_MODEL_NAME = "fusion_mlp"
MODEL_DIR = REPO_ROOT / "stage1_idiom_signal" / STAGE1_MODEL_NAME / "outputs"
MODEL_PATH = MODEL_DIR / "idiom_signal.pt"
INPUT_PATH = REPO_ROOT / "stage1_idiom_signal_final/input/submission_m_inputs.json"
OUTPUT_PATH = REPO_ROOT / "stage1_idiom_signal_final/output/submission_m_scores.json"

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
ENCODE_BATCH_SIZE = 16
TEXT_KEYS = ["hybrid_input", "target", "paraphrase", "lex_sub", "comp_drop", "literalize"]
REQUIRED_INPUT_KEYS = [
    "sample_id",
    "language",
    "row_index",
    "compound",
    "sentence",
    "paraphrase",
    "lex_sub",
    "comp_drop",
    "literalize",
]


def build_classifier(input_dim: int, hidden_1: int, hidden_2: int, dropout_1: float, dropout_2: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_1), nn.BatchNorm1d(hidden_1), nn.GELU(), nn.Dropout(dropout_1),
        nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2), nn.GELU(), nn.Dropout(dropout_2),
        nn.Linear(hidden_2, 1),
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def mark_compound(sentence: str, compound: str) -> str:
    pattern = re.compile(re.escape(compound), flags=re.IGNORECASE)
    match = pattern.search(sentence)
    if match is None:
        return sentence

    start, end = match.span()
    return f"{sentence[:start]}<mwe>{sentence[start:end]}</mwe>{sentence[end:]}"


def build_hybrid_input(record: dict[str, Any]) -> str:
    marked_sentence = mark_compound(
        sentence=str(record["sentence"]),
        compound=str(record["compound"]),
    )
    return f"[LANG={record['language']}] {marked_sentence} [MWE] {record['compound']}"


def read_input_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    for index, record in enumerate(records):
        missing = [key for key in REQUIRED_INPUT_KEYS if key not in record]
        if missing:
            raise KeyError(f"Missing keys in input record {index}: {missing}")

    return records


def build_model_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    model_records: list[dict[str, str]] = []

    for record in records:
        model_records.append(
            {
                "sample_id": str(record["sample_id"]),
                "language": str(record["language"]),
                "row_index": str(record["row_index"]),
                "compound": str(record["compound"]),
                "sentence": str(record["sentence"]),
                "hybrid_input": build_hybrid_input(record),
                "target": str(record["sentence"]),
                "paraphrase": str(record["paraphrase"]),
                "lex_sub": str(record["lex_sub"]),
                "comp_drop": str(record["comp_drop"]),
                "literalize": str(record["literalize"]),
            }
        )

    return model_records


def load_model(device: torch.device) -> nn.Sequential:
    payload = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    params = payload["hyperparameters"]
    model = build_classifier(
        input_dim=payload["input_dim"],
        hidden_1=params["hidden_1"], hidden_2=params["hidden_2"],
        dropout_1=params["dropout_1"], dropout_2=params["dropout_2"],
    ).to(device)
    model.load_state_dict(payload["classifier_state_dict"])
    model.eval()
    return model


def encode_records(records: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    encoder.eval()

    pooled_chunks: list[torch.Tensor] = []
    num_texts = len(TEXT_KEYS)

    for start in tqdm(range(0, len(records), ENCODE_BATCH_SIZE), desc="encoding M inputs"):
        batch = records[start : start + ENCODE_BATCH_SIZE]
        flat_texts = [record[key] for record in batch for key in TEXT_KEYS]

        encoded = tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            hidden_states = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            pooled = mean_pool(hidden_states, attention_mask)

        pooled_chunks.append(pooled.view(len(batch), num_texts, -1).cpu())

    encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(pooled_chunks, dim=0)


def build_features(pooled: torch.Tensor) -> torch.Tensor:
    key_to_index = {key: index for index, key in enumerate(TEXT_KEYS)}

    hybrid_input = pooled[:, key_to_index["hybrid_input"]]
    target = pooled[:, key_to_index["target"]]
    paraphrase = pooled[:, key_to_index["paraphrase"]]
    lex_sub = pooled[:, key_to_index["lex_sub"]]
    comp_drop = pooled[:, key_to_index["comp_drop"]]
    literalize = pooled[:, key_to_index["literalize"]]

    f_sub = torch.abs(target - lex_sub)
    f_comp = torch.abs(target - comp_drop)
    f_lit = torch.abs(target - literalize)
    f_para = torch.abs(target - paraphrase)

    return torch.cat([hybrid_input, f_sub, f_comp, f_lit, f_para], dim=-1)


def predict_records(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    device = get_device()
    model = load_model(device)
    pooled = encode_records(records, device)
    features = build_features(pooled).to(device)

    with torch.no_grad():
        logits = model(features).squeeze(-1)
        probabilities = torch.sigmoid(logits)

    outputs: list[dict[str, Any]] = []
    for record, logit, probability in zip(records, logits.cpu().tolist(), probabilities.cpu().tolist()):
        outputs.append(
            {
                "sample_id": record["sample_id"],
                "language": record["language"],
                "row_index": int(record["row_index"]),
                "compound": record["compound"],
                "sentence": record["sentence"],
                "m_logit": float(logit),
                "m_prob": float(probability),
            }
        )
    return outputs


def main() -> None:
    input_records = read_input_records(INPUT_PATH)
    model_records = build_model_records(input_records)
    predictions = predict_records(model_records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(predictions)} M scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
