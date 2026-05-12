from __future__ import annotations
import gc, json, random, re, unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

# --- CONFIG ---
REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHED_PATH = REPO_ROOT / "stage1_idiom_signal/data/matched_records.json"
MODEL_NAME = "xlm-roberta-base"
clean_model_name = MODEL_NAME.split("/")[-1].replace("-roberta-", "_").replace("-", "_")
OUTPUT_DIR = Path(__file__).resolve().parent / f"outputs_{clean_model_name}"
OPTUNA_SEEDS = (42, 100, 150)
FINAL_SEEDS = (42, 100, 150, 200, 250)
OPTUNA_EPOCHS = 30
FINAL_EPOCHS = 40
N_TRIALS = 30
MAX_LENGTH = 128
VAL_RATIO = 0.15
ENCODE_BATCH_SIZE = 16
TEXT_KEYS = ["lex_sub", "comp_drop", "literalize", "paraphrase"]

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def cleanup_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    elif device.type == "mps": torch.mps.empty_cache()

# --- SPAN POOLING UTILS ---
CYRILLIC_TO_LATIN = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'}

def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.casefold()
    for cyr, lat in CYRILLIC_TO_LATIN.items(): text = text.replace(cyr, lat)
    text = text.replace('ʻ', "'").replace('’', "'").replace('`', "'")
    return "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))

def get_universal_tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\u4e00-\u9FFF]|[^\W_]+", text, flags=re.UNICODE)]

def find_compound_spans(sentence: str, compound: str) -> list[tuple[int, int]]:
    compound = re.sub(r'\s+\d+$', '', compound)
    exact_match = re.search(re.escape(compound), sentence, flags=re.IGNORECASE)
    if exact_match: return [exact_match.span()]

    s_tokens, c_tokens = get_universal_tokens(sentence), get_universal_tokens(compound)
    if not s_tokens or not c_tokens: return []

    matched_spans, matched_c_indices = [], set()
    for c_idx, (c_word, _, _) in enumerate(c_tokens):
        norm_c = normalize_text(c_word)
        best_score, best_span = 0.0, None
        for s_word, s_start, s_end in s_tokens:
            norm_s = normalize_text(s_word)
            score = 1.0 if (norm_c in norm_s and len(norm_c) >= 3) else SequenceMatcher(None, norm_c, norm_s).ratio()
            if score > best_score: best_score, best_span = score, (s_start, s_end)
        if best_score >= 0.60 and best_span:
            matched_spans.append(best_span)
            matched_c_indices.add(c_idx)

    if len(matched_c_indices) / len(c_tokens) < 0.35 or not matched_spans: return []
    return sorted(list(set(matched_spans)), key=lambda x: x[0])

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

def target_span_pool(hidden_states: torch.Tensor, offset_mapping: torch.Tensor, spans_list: list[list[tuple[int, int]]], q_sentences: torch.Tensor) -> torch.Tensor:
    batch_vectors = []
    for batch_idx, spans in enumerate(spans_list):
        if not spans:
            batch_vectors.append(q_sentences[batch_idx])
            continue
        token_vectors = []
        for token_idx in range(offset_mapping.size(1)):
            tok_s, tok_e = int(offset_mapping[batch_idx, token_idx, 0]), int(offset_mapping[batch_idx, token_idx, 1])
            if tok_s == tok_e: continue
            if any(tok_s < end and tok_e > start for start, end in spans):
                token_vectors.append(hidden_states[batch_idx, token_idx])
        if len(token_vectors) == 0: batch_vectors.append(q_sentences[batch_idx])
        else: batch_vectors.append(torch.stack(token_vectors).mean(dim=0))
    return torch.stack(batch_vectors)

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from stage1_idiom_signal.model_registry import split_train_val_by_mwe

def encode_fusion_records(records: list[dict[str, Any]], model_name: str, max_length: int, device: torch.device) -> torch.Tensor:
    cache_dir = REPO_ROOT / "stage1_idiom_signal/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cache_{clean_model_name}_target_pert.pt"
    if cache_path.exists():
        print(f"Loading cached embeddings from: {cache_path}")
        return torch.load(cache_path)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    
    chunks = []
    for start in range(0, len(records), ENCODE_BATCH_SIZE):
        batch = records[start : start + ENCODE_BATCH_SIZE]
        targets = [rec["target"] for rec in batch]
        mwes = [rec["mwe"] for rec in batch]
        
        target_enc = tokenizer(targets, padding=True, truncation=True, max_length=max_length, return_offsets_mapping=True, return_tensors="pt")
        offsets = target_enc.pop("offset_mapping").to(device)
        target_enc = {k: v.to(device) for k, v in target_enc.items()}
        
        with torch.no_grad():
            t_hidden = encoder(**target_enc).last_hidden_state
            q_sentences = mean_pool(t_hidden, target_enc["attention_mask"])
            spans_list = [find_compound_spans(t, m) for t, m in zip(targets, mwes)]
            
            # DEBUG BLOCK
            if start == 0:
                print("\n--- DEBUG: TARGET_PERTURBATION_MLP SPANS ---")
                for idx in range(min(3, len(targets))):
                    spans = spans_list[idx]
                    mapped_text = " | ".join([targets[idx][s:e] for s, e in spans]) if spans else "NONE (Fallback to Mean Pool)"
                    print(f"Sample {idx}: '{targets[idx]}'")
                    print(f"Extracted : [{mapped_text}]")
                print("--------------------------------------------\n")

            target_pooled = target_span_pool(t_hidden, offsets, spans_list, q_sentences) 
            
        rewrites_flat = []
        for rec in batch: rewrites_flat.extend([rec[key] for key in TEXT_KEYS])
        rewrite_enc = tokenizer(rewrites_flat, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        rewrite_enc = {k: v.to(device) for k, v in rewrite_enc.items()}
        
        with torch.no_grad():
            r_hidden = encoder(**rewrite_enc).last_hidden_state
            rewrite_pooled = mean_pool(r_hidden, rewrite_enc["attention_mask"]) 
            
        rewrite_pooled = rewrite_pooled.view(len(batch), 4, -1) 
        batch_pooled = torch.cat([target_pooled.unsqueeze(1), rewrite_pooled], dim=1)
        chunks.append(batch_pooled.cpu())
        
    del encoder; cleanup_memory(device)
    res = torch.cat(chunks, dim=0)
    print(f"Saving cached embeddings to: {cache_path}")
    torch.save(res, cache_path)
    return res

class SpanFusionDataset(Dataset):
    def __init__(self, pooled: torch.Tensor, labels: list[float], mwes: list[str]):
        pooled = pooled.float()
        t = pooled[:, 0]
        self.features = torch.cat([
            t, 
            torch.abs(t - pooled[:, 1]),
            torch.abs(t - pooled[:, 2]),
            torch.abs(t - pooled[:, 3]),
            torch.abs(t - pooled[:, 4]),
        ], dim=-1)
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.mwes = mwes
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def train_single_seed(dataset, seed, device, hidden_1, hidden_2, dropout_1, dropout_2, lr, weight_decay, batch_size, epochs, save_path=None):
    set_seed(seed)
    train_indices, val_indices = split_train_val_by_mwe(dataset.mwes, dataset.labels.tolist(), VAL_RATIO, seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    input_dim = dataset.features.size(-1)
    
    classifier = nn.Sequential(
        nn.Linear(input_dim, hidden_1), nn.BatchNorm1d(hidden_1), nn.GELU(), nn.Dropout(dropout_1),
        nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2), nn.GELU(), nn.Dropout(dropout_2),
        nn.Linear(hidden_2, 1),
    ).to(device)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_f1, best_epoch, best_metrics, best_state = -1.0, 0, None, None
    patience, patience_counter = 5, 0

    for epoch in range(1, epochs + 1):
        classifier.train()
        for feats, lbls in train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad(); loss = criterion(classifier(feats).squeeze(-1), lbls); loss.backward(); torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0); optimizer.step()
        
        classifier.eval(); all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats, lbls = feats.to(device), lbls.to(device)
                all_logits.append(classifier(feats).squeeze(-1)); all_labels.append(lbls)
                
        from sklearn.metrics import accuracy_score, f1_score
        probs = torch.sigmoid(torch.cat(all_logits)).cpu().numpy()
        lbls = torch.cat(all_labels).cpu().numpy()
        preds = (probs >= 0.5).astype(float)
        f1 = f1_score(lbls, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_metrics = {"f1": f1, "accuracy": accuracy_score(lbls, preds)}
            if save_path: best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
            
    if save_path and best_state:
        torch.save({"classifier_state_dict": best_state, "input_dim": input_dim, "hyperparameters": {"hidden_1": hidden_1, "hidden_2": hidden_2, "dropout_1": dropout_1, "dropout_2": dropout_2}, "seed": seed, "best_epoch": best_epoch, "metrics": best_metrics}, save_path)
    del classifier, optimizer; cleanup_memory(device)
    return {"seed": seed, "best_epoch": best_epoch, "best_val_f1": best_f1, "best_metrics": best_metrics}

def objective(trial, dataset, device):
    params = {"hidden_1": trial.suggest_categorical("hidden_1", [512, 1024, 1536]), "hidden_2": trial.suggest_categorical("hidden_2", [128, 256, 512]), "dropout_1": trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1), "dropout_2": trial.suggest_float("dropout_2", 0.1, 0.5, step=0.1), "lr": trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3]), "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2]), "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])}
    seed_f1s = [train_single_seed(dataset, seed, device, epochs=OPTUNA_EPOCHS, **params)["best_val_f1"] for seed in OPTUNA_SEEDS]
    return mean(seed_f1s)

def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(payload, f, indent=2, ensure_ascii=False)

def main():
    set_seed(OPTUNA_SEEDS[0]); OUTPUT_DIR.mkdir(parents=True, exist_ok=True); device = get_device()
    with MATCHED_PATH.open("r", encoding="utf-8") as f: records = json.load(f)
    labels = [float(r["label"]) for r in records]
    mwes = [r["mwe"] for r in records]
    pooled = encode_fusion_records(records, MODEL_NAME, MAX_LENGTH, device); dataset = SpanFusionDataset(pooled, labels, mwes)
    study = optuna.create_study(direction="maximize"); study.optimize(lambda t: objective(t, dataset, device), n_trials=N_TRIALS)
    best_params = study.best_trial.params
    per_seed = [train_single_seed(dataset, s, device, epochs=FINAL_EPOCHS, **best_params) for s in FINAL_SEEDS]
    best_run = max(per_seed, key=lambda r: r["best_val_f1"])
    train_single_seed(dataset, best_run["seed"], device, epochs=FINAL_EPOCHS, save_path=OUTPUT_DIR / "idiom_signal.pt", **best_params)
    
    save_json(OUTPUT_DIR / "config.json", {"model_name": MODEL_NAME, "input_format": "Target Span-Pool + Perturbation Diffs", "hyperparameters": best_params, "final_seeds": list(FINAL_SEEDS), "optuna_epochs": OPTUNA_EPOCHS, "final_epochs": FINAL_EPOCHS})
    save_json(OUTPUT_DIR / "metrics.json", {"name": "target_perturbation_mlp", "mean_val_f1": mean(r["best_val_f1"] for r in per_seed), "best_seed": best_run["seed"], "best_val_f1": best_run["best_val_f1"], "best_metrics": best_run["best_metrics"], "hyperparameters": best_params, "per_seed": per_seed})
    
    print(f"Done. Mean F1: {mean(r['best_val_f1'] for r in per_seed):.4f}")

if __name__ == "__main__": main()
