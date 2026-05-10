from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def percent(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def counter_payload(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "count": count,
            "percent": percent(count, total),
        }
        for key, count in counter.most_common()
    }


def analyze_csv(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    total = len(rows)
    if total == 0:
        raise ValueError(f"CSV file contains no rows: {path}")

    labels = Counter((row.get("Label") or "").strip() for row in rows)
    languages = Counter((row.get("Language") or "").strip() for row in rows)
    mwes = Counter((row.get("MWE") or "").strip() for row in rows)
    targets = [(row.get("Target") or "").strip() for row in rows]

    language_label: dict[str, Counter[str]] = defaultdict(Counter)
    mwe_label: dict[str, Counter[str]] = defaultdict(Counter)
    mwe_found_in_target = 0

    for row in rows:
        language = (row.get("Language") or "").strip()
        label = (row.get("Label") or "").strip()
        mwe = (row.get("MWE") or "").strip()
        target = (row.get("Target") or "").strip()

        language_label[language][label] += 1
        mwe_label[mwe][label] += 1

        if mwe and mwe.lower() in target.lower():
            mwe_found_in_target += 1

    non_empty_targets = [target for target in targets if target]
    word_counts = [len(target.split()) for target in non_empty_targets]
    char_counts = [len(target) for target in non_empty_targets]

    mixed_label_mwes = sum(
        1
        for counts in mwe_label.values()
        if len([label for label, count in counts.items() if label and count > 0]) > 1
    )

    language_label_payload: dict[str, Any] = {}
    for language, counts in sorted(language_label.items()):
        language_total = sum(counts.values())
        language_label_payload[language] = counter_payload(counts, language_total)
        language_label_payload[language]["total"] = language_total

    return {
        "total_rows": total,
        "labels": counter_payload(labels, total),
        "languages": counter_payload(languages, total),
        "language_label": language_label_payload,
        "mwe": {
            "unique_count": len(mwes),
            "empty_count": sum(1 for key in mwes if key == ""),
            "mixed_label_mwe_count": mixed_label_mwes,
            "top_20": counter_payload(Counter(dict(mwes.most_common(20))), total),
        },
        "target": {
            "unique_count": len(set(targets)),
            "unique_percent": percent(len(set(targets)), total),
            "empty_count": sum(1 for target in targets if not target),
            "mwe_found_count": mwe_found_in_target,
            "mwe_found_percent": percent(mwe_found_in_target, total),
            "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0.0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "avg_chars": sum(char_counts) / len(char_counts) if char_counts else 0.0,
        },
        "missing": {
            "language": sum(1 for row in rows if not (row.get("Language") or "").strip()),
            "mwe": sum(1 for row in rows if not (row.get("MWE") or "").strip()),
            "target": sum(1 for row in rows if not (row.get("Target") or "").strip()),
            "label": sum(1 for row in rows if not (row.get("Label") or "").strip()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage 1 idiom CSV distribution.")
    parser.add_argument(
        "--csv_path",
        default="stage1_idiom_signal/data/unique_mwe_target_label_language.csv",
    )
    parser.add_argument(
        "--output_json",
        default="stage1_idiom_signal/reports/dataset_distribution.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_csv(args.csv_path)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
