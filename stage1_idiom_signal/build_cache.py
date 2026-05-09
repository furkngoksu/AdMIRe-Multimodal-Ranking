import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "stage1_idiom_signal/data/unique_mwe_target_label_language.csv"
LLM_RESULTS_DIR = REPO_ROOT / "llm_results"
CACHE_DIR = REPO_ROOT / "stage1_idiom_signal/cache"
POOLED_CACHE_PATH = CACHE_DIR / "embedding_cache_pooled.pt"
HIDDEN_SHARD_DIR = CACHE_DIR / "hidden_state_shards"
HIDDEN_INDEX_PATH = CACHE_DIR / "embedding_hidden_state_index.json"
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
SAVE_HIDDEN_STATE_SHARDS = True


def remove_mwe_from_sentence(sentence: str, mwe: str) -> str:
    context = re.sub(re.escape(mwe), "", sentence, count=1, flags=re.IGNORECASE)
    return " ".join(context.split())


def find_phrase_span(text: str, phrase: str) -> tuple[int, int] | None:
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def span_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    offset_mapping: torch.Tensor,
    spans: list[tuple[int, int] | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = []
    span_mask = []

    for index, span in enumerate(spans):
        if span is None:
            pooled.append(hidden_states.new_zeros(hidden_states.size(-1)))
            span_mask.append(False)
            continue

        span_start, span_end = span
        token_starts = offset_mapping[index, :, 0]
        token_ends = offset_mapping[index, :, 1]

        token_mask = attention_mask[index].bool()
        token_mask = token_mask & (token_ends > token_starts)
        token_mask = token_mask & (token_ends > span_start)
        token_mask = token_mask & (token_starts < span_end)

        if not token_mask.any():
            pooled.append(hidden_states.new_zeros(hidden_states.size(-1)))
            span_mask.append(False)
            continue

        pooled.append(hidden_states[index, token_mask].mean(dim=0))
        span_mask.append(True)

    return torch.stack(pooled, dim=0), torch.tensor(span_mask, dtype=torch.bool)


def match_llm_to_csv(llm_data: list[dict[str, Any]], csv_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_strings = [c['target'] for c in csv_targets]
    
    # Pre-tokenize targets for O(1) comparison
    target_tokens = [set(t.lower().split()) for t in target_strings]
    
    matched_records = []
    
    for item in tqdm(llm_data, desc="Matching LLM to CSV", leave=False):
        v = item['value']
        
        # Sometimes one of the tasks fails, we check if they exist
        if not all(k in v for k in ['lex_sub', 'comp_drop', 'paraphrase', 'literal_desc', 'literalize']):
            continue
            
        lex_sub = v['lex_sub']['text']
        lex_tokens = set(lex_sub.lower().split())
        
        best_score = 0.0
        best_idx = -1
        
        for i, t_tokens in enumerate(target_tokens):
            if len(lex_tokens) == 0 and len(t_tokens) == 0:
                continue
            score = len(lex_tokens & t_tokens) / len(lex_tokens | t_tokens)
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx != -1 and best_score > 0.4:
            csv_record = csv_targets[best_idx]
            
            matched_records.append({
                'key': item['key'],
                'target': csv_record['target'],
                'context': remove_mwe_from_sentence(
                    sentence=csv_record['target'],
                    mwe=csv_record['mwe'],
                ),
                'label': csv_record['label'],
                'mwe': csv_record['mwe'],
                'language': csv_record['language'],
                'paraphrase': v['paraphrase']['text'],
                'lex_sub': lex_sub,
                'comp_drop': v['comp_drop']['text'],
                'literal_desc': v['literal_desc']['text'],
                'literalize': v['literalize']['text']
            })
    return matched_records

def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_HIDDEN_STATE_SHARDS:
        HIDDEN_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 1. Read CSV
    print(f"Reading CSV from {CSV_PATH}")
    csv_data = []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data.append({
                'target': row['Target'].strip(),
                'label': float(row['Label']),
                'mwe': row['MWE'].strip(),
                'language': row['Language'].strip()
            })
            
    # 2. Read JSONL for all languages
    llm_files = list(LLM_RESULTS_DIR.glob("llm_*.jsonl"))
    all_matched = []
    
    for jsonl_path in llm_files:
        lang_code = jsonl_path.stem.split('_')[1] # e.g. 'English'
        # Map back to simple code
        lang_code_map = {'English': 'EN', 'Portuguese': 'PT'}
        code = lang_code_map.get(lang_code, lang_code)
        
        print(f"Processing {jsonl_path.name} ({code})...")
        llm_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    llm_data.append(json.loads(line))
        
        # Filter csv_data for this language
        lang_csv = [c for c in csv_data if c['language'] == code]
        matched = match_llm_to_csv(llm_data, lang_csv)
        all_matched.extend(matched)
        
    print(f"Total matched records across all languages: {len(all_matched)}")
    
    # 3. Encode and Cache
    TEXT_KEYS = [
        'target',
        'context',
        'paraphrase',
        'lex_sub',
        'comp_drop',
        'literalize',
        'literal_desc',
    ]
    num_texts = len(TEXT_KEYS)
    
    all_pooled = []
    all_target_span_pooled = []
    all_target_span_mask = []
    metadata = []
    
    hidden_shards = []
    
    for i in tqdm(range(0, len(all_matched), BATCH_SIZE), desc="Encoding to Cache"):
        batch_records = all_matched[i:i+BATCH_SIZE]
        batch_id = i // BATCH_SIZE
        
        flat_texts = []
        flat_spans = []
        for r in batch_records:
            for k in TEXT_KEYS:
                text = r[k]
                flat_texts.append(text)
                flat_spans.append(find_phrase_span(text, r["mwe"]))
                
        encoded = tokenizer(
            flat_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        offset_mapping = encoded["offset_mapping"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            pooled = mean_pool(hidden_states, attention_mask)
            target_span_pooled, target_span_mask = span_mean_pool(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                offset_mapping=offset_mapping,
                spans=flat_spans,
            )
            
        B = len(batch_records)
        pooled = pooled.view(B, num_texts, -1).cpu().to(torch.float16)
        target_span_pooled = target_span_pooled.view(B, num_texts, -1).cpu().to(torch.float16)
        target_span_mask = target_span_mask.view(B, num_texts).cpu()
        input_ids = input_ids.view(B, num_texts, MAX_LENGTH).cpu()
        attention_mask = attention_mask.view(B, num_texts, MAX_LENGTH).cpu()
        offset_mapping = offset_mapping.view(B, num_texts, MAX_LENGTH, 2).cpu()
        
        all_pooled.append(pooled)
        all_target_span_pooled.append(target_span_pooled)
        all_target_span_mask.append(target_span_mask)

        if SAVE_HIDDEN_STATE_SHARDS:
            hidden_states = hidden_states.view(
                B,
                num_texts,
                MAX_LENGTH,
                -1,
            ).cpu().to(torch.float16)
            shard_path = HIDDEN_SHARD_DIR / f"hidden_states_{batch_id:04d}.pt"
            start_index = len(metadata)
            end_index = start_index + B

            torch.save(
                {
                    "hidden_states": hidden_states,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "offset_mapping": offset_mapping,
                    "start_index": start_index,
                    "end_index": end_index,
                },
                shard_path,
            )
            hidden_shards.append(
                {
                    "file": str(shard_path.relative_to(SCRIPT_DIR)),
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )
        
        for r in batch_records:
            metadata.append({
                'key': r['key'],
                'target': r['target'],
                'context': r['context'],
                'label': r['label'],
                'mwe': r['mwe'],
                'language': r['language']
            })
            
    # Concatenate all
    print("Concatenating tensors...")
    cache = {
        'pooled': torch.cat(all_pooled, dim=0),
        'target_span_pooled': torch.cat(all_target_span_pooled, dim=0),
        'target_span_mask': torch.cat(all_target_span_mask, dim=0),
        'metadata': metadata,
        'text_keys_order': TEXT_KEYS
    }
    
    print(f"Saving pooled cache to {POOLED_CACHE_PATH}...")
    print(f"Pooled shape: {cache['pooled'].shape} ({cache['pooled'].dtype})")
    print(f"Target span pooled shape: {cache['target_span_pooled'].shape} ({cache['target_span_pooled'].dtype})")
    print(f"Target span coverage: {cache['target_span_mask'].float().mean().item():.4f}")
    
    torch.save(cache, POOLED_CACHE_PATH)

    if SAVE_HIDDEN_STATE_SHARDS:
        hidden_index = {
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "text_keys_order": TEXT_KEYS,
            "num_records": len(metadata),
            "shards": hidden_shards,
        }
        HIDDEN_INDEX_PATH.write_text(
            json.dumps(hidden_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Hidden state shards: {len(hidden_shards)}")
        print(f"Hidden state index: {HIDDEN_INDEX_PATH}")

    print("Done! Cache is ready for super-fast training and analysis.")

if __name__ == "__main__":
    main()
