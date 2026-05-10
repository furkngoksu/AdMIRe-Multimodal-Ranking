from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "stage1_idiom_signal/fusion_mlp/outputs"
MODEL_PATH = MODEL_DIR / "idiom_signal.pt"
LLM_RESULTS_DIR = REPO_ROOT / "llm_results_training"
OUTPUT_PATH = REPO_ROOT / "stage1_idiom_signal_final/output/admire_train_scores.tsv"

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
ENCODE_BATCH_SIZE = 16
TEXT_KEYS = ["hybrid_input", "target", "paraphrase", "lex_sub", "comp_drop", "literalize"]

SPLIT_FILES = {
    "EN": {
        "train": REPO_ROOT / "admire2_training/EN/train/subtask_a_train.tsv",
        "dev": REPO_ROOT / "admire2_training/EN/dev/subtask_a_dev.tsv",
        "test": REPO_ROOT / "admire2_training/EN/test/subtask_a_test.tsv",
        "xeval": REPO_ROOT / "admire2_training/EN/xeval/subtask_a_xe.tsv",
    },
    "PT": {
        "train": REPO_ROOT / "admire2_training/PT/train/subtask_a_train.tsv",
        "dev": REPO_ROOT / "admire2_training/PT/dev/subtask_a_dev.tsv",
        "test": REPO_ROOT / "admire2_training/PT/test/subtask_a_test.tsv",
        "xeval": REPO_ROOT / "admire2_training/PT/xeval/subtask_a_xp.tsv",
    },
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def mark_compound(sentence: str, compound: str) -> str:
    match = re.search(re.escape(compound), sentence, flags=re.IGNORECASE)
    if match is None:
        return sentence
    s, e = match.span()
    return f"{sentence[:s]}<mwe>{sentence[s:e]}</mwe>{sentence[e:]}"


def load_llm_results(language: str, split: str) -> dict[str, dict[str, Any]]:
    path = LLM_RESULTS_DIR / f"llm_{language}_{split}.jsonl"
    if not path.exists():
        print(f"  WARNING: LLM results not found at {path}")
        return {}

    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            value = record.get("value") or {}
            if isinstance(value, str):
                value = json.loads(value)
            
            orig_sent = value.get("sentence", "").strip()
            if orig_sent:
                results[orig_sent] = value
    
    print(f"  Loaded {len(results)} LLM records from {path.name}")
    return results


def read_training_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_inference_records(
    language: str, split: str, tsv_path: Path, llm_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_training_tsv(tsv_path)
    records: list[dict[str, Any]] = []
    missing_count = 0

    for row_index, row in enumerate(rows):
        sample_id = f"{language}_{split}_{row_index}"
        sentence = row["sentence"].strip()
        compound = row["compound"]

        llm = llm_results.get(sentence)
        if llm is None:
            missing_count += 1
            lex_sub = comp_drop = paraphrase = literalize = sentence
        else:
            lex_sub = str(llm.get("lex_sub", {}).get("text", sentence))
            comp_drop = str(llm.get("comp_drop", {}).get("text", sentence))
            paraphrase = str(llm.get("paraphrase", {}).get("text", sentence))
            literalize = str(llm.get("literalize", {}).get("text", sentence))

        marked = mark_compound(sentence, compound)
        hybrid_input = f"[LANG={language}] {marked} [MWE] {compound}"

        records.append({
            "sample_id": sample_id,
            "language": language,
            "compound": compound,
            "sentence": sentence,
            "hybrid_input": hybrid_input,
            "target": sentence,
            "paraphrase": paraphrase,
            "lex_sub": lex_sub,
            "comp_drop": comp_drop,
            "literalize": literalize,
        })

    if missing_count > 0:
        print(f"  WARNING: {missing_count}/{len(rows)} rows missing LLM results in {split}")
    return records


def build_classifier(input_dim: int, hidden_1: int, hidden_2: int, dropout_1: float, dropout_2: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_1), nn.BatchNorm1d(hidden_1), nn.GELU(), nn.Dropout(dropout_1),
        nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2), nn.GELU(), nn.Dropout(dropout_2),
        nn.Linear(hidden_2, 1),
    )


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


def encode_and_predict(records: list[dict[str, Any]], model: FusionMLPModel, device: torch.device) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    encoder.eval()

    all_probs: list[float] = []
    for start in tqdm(range(0, len(records), ENCODE_BATCH_SIZE), desc="inference"):
        batch = records[start : start + ENCODE_BATCH_SIZE]
        flat_texts = [rec[key] for rec in batch for key in TEXT_KEYS]

        encoded = tokenizer(flat_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            ).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"].to(device))
            pooled = pooled.view(len(batch), len(TEXT_KEYS), -1)

            # Build features: hybrid + abs diffs
            key_idx = {k: i for i, k in enumerate(TEXT_KEYS)}
            hybrid = pooled[:, key_idx["hybrid_input"]]
            target = pooled[:, key_idx["target"]]
            features = torch.cat([
                hybrid,
                torch.abs(target - pooled[:, key_idx["lex_sub"]]),
                torch.abs(target - pooled[:, key_idx["comp_drop"]]),
                torch.abs(target - pooled[:, key_idx["literalize"]]),
                torch.abs(target - pooled[:, key_idx["paraphrase"]]),
            ], dim=-1)

            logits = model(features).squeeze(-1)
            probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().tolist())

    del encoder
    return all_probs


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    print("Loading fusion_mlp model...")
    model = load_model(device)

    all_records: list[dict[str, Any]] = []
    for language, splits in SPLIT_FILES.items():
        print(f"\nProcessing {language}...")
        for split, tsv_path in splits.items():
            if not tsv_path.exists():
                print(f"  Skipping {split}: {tsv_path} not found")
                continue
            
            llm_results = load_llm_results(language, split)
            records = build_inference_records(language, split, tsv_path, llm_results)
            all_records.extend(records)
            print(f"  {split}: {len(records)} samples")

    print(f"\nTotal: {len(all_records)} samples")
    print("Running inference...")
    probs = encode_and_predict(all_records, model, device)

    # Write TSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "m_prob"], delimiter="\t")
        writer.writeheader()
        for record, prob in zip(all_records, probs):
            writer.writerow({"sample_id": record["sample_id"], "m_prob": f"{prob:.6f}"})

    print(f"\nWrote {len(probs)} M scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
