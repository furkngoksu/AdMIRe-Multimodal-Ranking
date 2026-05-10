import csv
import sys
from pathlib import Path
from tqdm.auto import tqdm

# Repo root setup
REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.query_encoder import QueryEncoder

def main():
    # Setup paths
    SUBMISSION_TSV_DIR = REPO_ROOT / "admire2_submission/admire2_sentences"
    OUTPUT_CSV = REPO_ROOT / "span_verification_log.csv"
    
    results = []
    failed_samples = []
    tsv_files = sorted(SUBMISSION_TSV_DIR.glob("*.tsv"))
    
    print(f"Checking {len(tsv_files)} languages...")

    for tsv_path in tsv_files:
        lang = tsv_path.stem.replace("submission_", "")
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in list(reader):
                compound = row["compound"].strip()
                sentence = row["sentence"].strip()
                
                # Check Exact Match
                exact_start = sentence.lower().find(compound.lower())
                is_exact = exact_start != -1
                
                try:
                    start, end = QueryEncoder.find_compound_span(sentence, compound)
                    matched_text = sentence[start:end]
                    
                    # For score calculation
                    from difflib import SequenceMatcher
                    norm_s = QueryEncoder.normalize_for_match(matched_text)
                    norm_c = QueryEncoder.normalize_for_match(compound)
                    score = SequenceMatcher(None, norm_c, norm_s).ratio()
                    
                    results.append({
                        "lang": lang,
                        "match_type": "EXACT" if is_exact else "FUZZY",
                        "score": round(score, 3),
                        "compound": compound,
                        "matched_text": matched_text,
                        "sentence": sentence
                    })
                except Exception as e:
                    fail_info = {
                        "lang": lang,
                        "match_type": "FAILED",
                        "score": 0.0,
                        "compound": compound,
                        "matched_text": "NOT_FOUND",
                        "sentence": sentence
                    }
                    results.append(fail_info)
                    failed_samples.append(fail_info)

    # Save to CSV
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lang", "match_type", "score", "compound", "matched_text", "sentence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nVerification complete. Results saved to: {OUTPUT_CSV}")
    
    if failed_samples:
        print("\n" + "="*50)
        print("CRITICAL: FAILED SAMPLES (THRESHOLD TOO HIGH OR MISSING)")
        print("="*50)
        for i, f in enumerate(failed_samples, 1):
            print(f"{i}. [{f['lang']}] Compound: {f['compound']}")
            print(f"   Sentence: {f['sentence']}")
            print("-" * 30)
    else:
        print("\nNo failures found!")

    fuzzy_count = sum(1 for r in results if r["match_type"] == "FUZZY")
    fail_count = len(failed_samples)
    print(f"\nSummary:")
    print(f"  Exact: {len(results) - fuzzy_count - fail_count}")
    print(f"  Fuzzy: {fuzzy_count}")
    print(f"  Failed: {fail_count}")

if __name__ == "__main__":
    main()
