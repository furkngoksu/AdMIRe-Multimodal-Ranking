"""One-time script to match CSV idiom records with LLM perturbation outputs.

Writes data/matched_records.json so that perturbation_mlp and fusion_mlp
can load pre-matched data without repeating the matching step every run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "stage1_idiom_signal/data/unique_mwe_target_label_language.csv"
LLM_RESULTS_DIR = REPO_ROOT / "llm_results"
OUTPUT_PATH = REPO_ROOT / "stage1_idiom_signal/data/matched_records.json"
REQUIRED_COLUMNS = {"Language", "MWE", "Target", "Label"}
LANG_CODE_MAP = {"English": "EN", "Portuguese": "PT"}


def token_set(text: str) -> set[str]:
    return set(text.lower().split())


def read_idiom_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            label = float((row.get("Label") or "").strip())
            rows.append({
                "language": (row.get("Language") or "").strip(),
                "mwe": (row.get("MWE") or "").strip(),
                "target": (row.get("Target") or "").strip(),
                "label": label,
            })
    return rows


def read_llm_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def match_records(
    csv_rows: list[dict[str, Any]],
    llm_results_dir: Path,
) -> list[dict[str, Any]]:
    all_matched: list[dict[str, Any]] = []

    for jsonl_path in sorted(llm_results_dir.glob("llm_*.jsonl")):
        lang_name = jsonl_path.stem.split("_", 1)[1]
        lang_code = LANG_CODE_MAP.get(lang_name, lang_name)
        csv_for_lang = [r for r in csv_rows if r["language"] == lang_code]
        target_tokens = [token_set(r["target"]) for r in csv_for_lang]

        if not csv_for_lang:
            continue

        llm_records = read_llm_jsonl(jsonl_path)
        for item in tqdm(llm_records, desc=f"Matching {jsonl_path.name}", leave=False):
            value = item.get("value") or {}
            required = ["lex_sub", "comp_drop", "paraphrase", "literalize"]
            if not all(k in value and "text" in value[k] for k in required):
                continue

            lex_text = str(value["lex_sub"]["text"])
            lex_tokens = token_set(lex_text)
            if not lex_tokens:
                continue

            best_score, best_idx = 0.0, -1
            for idx, cand_tokens in enumerate(target_tokens):
                union = lex_tokens | cand_tokens
                if not union:
                    continue
                score = len(lex_tokens & cand_tokens) / len(union)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1 or best_score <= 0.4:
                continue

            row = csv_for_lang[best_idx]
            all_matched.append({
                "key": item.get("key"),
                "language": row["language"],
                "mwe": row["mwe"],
                "label": row["label"],
                "target": row["target"],
                "paraphrase": str(value["paraphrase"]["text"]),
                "lex_sub": lex_text,
                "comp_drop": str(value["comp_drop"]["text"]),
                "literalize": str(value["literalize"]["text"]),
                "match_score": best_score,
            })

    return all_matched


def main() -> None:
    print(f"Reading CSV: {CSV_PATH}")
    csv_rows = read_idiom_csv(CSV_PATH)
    print(f"  {len(csv_rows)} rows loaded")

    print(f"Matching with LLM results in: {LLM_RESULTS_DIR}")
    matched = match_records(csv_rows, LLM_RESULTS_DIR)
    print(f"  {len(matched)} records matched")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(matched, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
