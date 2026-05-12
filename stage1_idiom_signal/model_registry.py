from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List
import torch
import torch.nn as nn

# --- ROBUST MARKING UTILITIES ---

CYRILLIC_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e',
    'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj',
    'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's',
    'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č',
    'џ': 'dž', 'ш': 'š'
}

def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.casefold()
    for cyr, lat in CYRILLIC_TO_LATIN.items():
        text = text.replace(cyr, lat)
    text = text.replace('ʻ', "'").replace('’', "'").replace('`', "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))

def get_universal_tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\u4e00-\u9FFF]|[^\W_]+", text, flags=re.UNICODE)]

def mark_compound(sentence: str, compound: str) -> str:
    """Robustly adds <mwe> tags, handling discontinuous spans and complex scripts."""
    if not compound: return sentence
    
    # 1. Exact match attempt
    compound_clean = re.sub(r'\s+\d+$', '', compound)
    exact_match = re.search(re.escape(compound_clean), sentence, flags=re.IGNORECASE)
    if exact_match:
        s, e = exact_match.span()
        return f"{sentence[:s]}<mwe>{sentence[s:e]}</mwe>{sentence[e:]}"

    # 2. Robust Token & Sequence Matching Fallback
    s_tokens = get_universal_tokens(sentence)
    c_tokens = get_universal_tokens(compound_clean)
    
    if not s_tokens or not c_tokens: return sentence

    matched_spans = []
    matched_c_indices = set()
    
    for c_idx, (c_word, _, _) in enumerate(c_tokens):
        norm_c = normalize_text(c_word)
        best_score = 0.0
        best_span = None
        for s_word, s_start, s_end in s_tokens:
            norm_s = normalize_text(s_word)
            score = 1.0 if (norm_c in norm_s and len(norm_c) >= 3) else SequenceMatcher(None, norm_c, norm_s).ratio()
            if score > best_score:
                best_score = score
                best_span = (s_start, s_end)
        
        if best_score >= 0.60 and best_span:
            matched_spans.append(best_span)
            matched_c_indices.add(c_idx)

    # If coverage is too low, return raw sentence
    if len(matched_c_indices) / len(c_tokens) < 0.35 or not matched_spans:
        return sentence

    # 3. Apply tags from Right to Left (Reverse) to prevent index shifting
    matched_spans = sorted(list(set(matched_spans)), key=lambda x: x[0])
    marked_sentence = sentence
    for s, e in reversed(matched_spans):
        marked_sentence = f"{marked_sentence[:s]}<mwe>{marked_sentence[s:e]}</mwe>{marked_sentence[e:]}"
        
    return marked_sentence

# --- MODEL ENGINES ---

class BaseMModelEngine:
    def prepare_texts(self, lang: str, sent: str, comp: str, loader: Any) -> List[str]:
        raise NotImplementedError
        
    def get_features(self, pooled_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def build_classifier(self, input_dim: int, params: Dict[str, Any]) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, params["hidden_1"]), 
            nn.BatchNorm1d(params["hidden_1"]), 
            nn.GELU(), 
            nn.Dropout(params["dropout_1"]),
            nn.Linear(params["hidden_1"], params["hidden_2"]), 
            nn.BatchNorm1d(params["hidden_2"]), 
            nn.GELU(), 
            nn.Dropout(params["dropout_2"]),
            nn.Linear(params["hidden_2"], 1),
        )

class BaselineEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        return [sent]
    def get_features(self, pooled_vectors):
        if pooled_vectors.dim() == 3: return pooled_vectors[:, 0, :]
        return pooled_vectors

class LangMarkedEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        marked = mark_compound(sent, comp)
        return [f"[LANG={lang}] {marked} [MWE] {comp}"]
    def get_features(self, pooled_vectors):
        if pooled_vectors.dim() == 3: return pooled_vectors[:, 0, :]
        return pooled_vectors

class LangMarkedPerturbationEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        marked = mark_compound(sent, comp)
        rewrites = loader.get_rewrites(sent)
        return [
            f"[LANG={lang}] {marked} [MWE] {comp}",
            sent,
            rewrites["lex_sub"], rewrites["comp_drop"], rewrites["paraphrase"], rewrites["literalize"]
        ]
    def get_features(self, pooled_vectors):
        h = pooled_vectors[:, 0] 
        t = pooled_vectors[:, 1]
        diffs = [torch.abs(t - pooled_vectors[:, i]) for i in range(2, 6)]
        return torch.cat([h] + diffs, dim=-1)

class PerturbationEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        rewrites = loader.get_rewrites(sent)
        return [
            sent,
            rewrites["lex_sub"], rewrites["comp_drop"], rewrites["paraphrase"], rewrites["literalize"]
        ]
    def get_features(self, pooled_vectors):
        t = pooled_vectors[:, 0]
        diffs = [torch.abs(t - pooled_vectors[:, i]) for i in range(1, 5)]
        return torch.cat(diffs, dim=-1)

class TargetPerturbationEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        rewrites = loader.get_rewrites(sent)
        return [
            sent, 
            rewrites["lex_sub"], rewrites["comp_drop"], rewrites["paraphrase"], rewrites["literalize"]
        ]
    def get_features(self, pooled_vectors):
        t = pooled_vectors[:, 0]
        diffs = [torch.abs(t - pooled_vectors[:, i]) for i in range(1, 5)]
        return torch.cat([t] + diffs, dim=-1)

MODEL_REGISTRY: Dict[str, BaseMModelEngine] = {
    "baseline_mlp": BaselineEngine(),
    "lang_marked_mlp": LangMarkedEngine(),
    "lang_marked_perturbation_mlp": LangMarkedPerturbationEngine(),
    "perturbation_mlp": PerturbationEngine(),
    "target_perturbation_mlp": TargetPerturbationEngine(),
}
