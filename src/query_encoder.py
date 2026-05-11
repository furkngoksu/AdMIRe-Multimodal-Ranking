from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.config import QueryEncoderConfig


class FrozenPCAProjection(nn.Module):
    """Frozen PCA projection from cached Q_raw to the shared query dimension."""

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)

    def forward(self, q_raw: torch.Tensor) -> torch.Tensor:
        projected = (q_raw - self.mean) @ self.components.t()

        if projected.size(-1) < self.output_dim:
            padding = projected.new_zeros(
                projected.size(0),
                self.output_dim - projected.size(-1),
            )
            projected = torch.cat([projected, padding], dim=-1)

        return projected


class QueryEncoder(nn.Module):
    # Robust thresholds for noisy multilingual datasets
    TOKEN_SIMILARITY_THRESHOLD = 0.60
    IDIOM_COVERAGE_THRESHOLD = 0.35

    # Cyrillic to Latin map for Serbian and similar languages
    CYRILLIC_TO_LATIN = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e',
        'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj',
        'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's',
        'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č',
        'џ': 'dž', 'ш': 'š'
    }

    def __init__(self, config: QueryEncoderConfig = QueryEncoderConfig()) -> None:
        super().__init__()

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
        )
        self.encoder = AutoModel.from_pretrained(config.model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        hidden_dim = self.encoder.config.hidden_size
        
        if getattr(config, "query_projection_type", "linear") == "pca":
            self.projection = FrozenPCAProjection(
                mean=torch.zeros(1, hidden_dim * 2),
                components=torch.zeros(config.shared_dim, hidden_dim * 2),
                output_dim=config.shared_dim,
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, config.shared_dim),
                nn.LayerNorm(config.shared_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Karakter standardizasyonu, transliterasyon ve aksan temizliği."""
        if not text: return ""
        text = text.casefold()
        
        # Cyrillic -> Latin (Serbian/Russian cases)
        for cyr, lat in cls.CYRILLIC_TO_LATIN.items():
            text = text.replace(cyr, lat)
            
        # Uzbek apostrophes standardization
        text = text.replace('ʻ', "'").replace('’', "'").replace('`', "'")
        
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(char for char in decomposed if not unicodedata.combining(char))

    @staticmethod
    def get_universal_tokens(text: str) -> List[Tuple[str, int, int]]:
        """Extracts language-agnostic tokens (handles CJK and alphabetic scripts)."""
        tokens = []
        # Pattern: Chinese char block OR alphabetic word characters
        pattern = re.compile(r"[\u4e00-\u9FFF]|[^\W_]+", flags=re.UNICODE)
        for match in pattern.finditer(text):
            tokens.append((match.group(0), match.start(), match.end()))
        return tokens

    @staticmethod
    def clean_compound(compound: str) -> str:
        """Cleans trailing numbers (Turkish dataset artifacts)."""
        return re.sub(r'\s+\d+$', '', compound)

    def find_compound_spans(self, sentence: str, compound: str) -> List[Tuple[int, int]]:
        """
        Multi-span alignment with fallback. Returns empty list if coverage is too low.
        """
        compound = self.clean_compound(compound)
        
        # Phase 1: Exact Match
        exact_match = re.search(re.escape(compound), sentence, flags=re.IGNORECASE)
        if exact_match:
            return [exact_match.span()]

        # Phase 2: Morphological Token Alignment
        s_tokens = self.get_universal_tokens(sentence)
        c_tokens = self.get_universal_tokens(compound)

        if not s_tokens or not c_tokens:
            return [] # Triggers Fallback

        matched_spans = []
        matched_c_indices = set()

        for c_idx, (c_word, _, _) in enumerate(c_tokens):
            norm_c = self.normalize_text(c_word)
            best_score = 0.0
            best_span = None

            for s_word, s_start, s_end in s_tokens:
                norm_s = self.normalize_text(s_word)
                
                # Substring match for joined words (e.g., xomsemiz)
                if norm_c in norm_s and len(norm_c) >= 3:
                    score = 1.0
                else:
                    score = SequenceMatcher(None, norm_c, norm_s).ratio()
                
                if score > best_score:
                    best_score = score
                    best_span = (s_start, s_end)

            if best_score >= self.TOKEN_SIMILARITY_THRESHOLD and best_span:
                matched_spans.append(best_span)
                matched_c_indices.add(c_idx)

        # Phase 3: Coverage Validation
        coverage = len(matched_c_indices) / len(c_tokens)
        if coverage < self.IDIOM_COVERAGE_THRESHOLD or not matched_spans:
            return [] # Triggers Fallback

        return sorted(list(set(matched_spans)), key=lambda x: x[0])

    @staticmethod
    def mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool over real tokens, excluding padding and special tokens."""
        real_token_mask = (
            (attention_mask == 1)
            & (offset_mapping[:, :, 0] != offset_mapping[:, :, 1])
        )
        mask = real_token_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def target_span_pool(
        hidden_states: torch.Tensor,
        offset_mapping: torch.Tensor,
        spans_list: List[List[Tuple[int, int]]],
        q_sentences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool vectors from the idiom parts. 
        Falls back to the full sentence mean if no idiom parts were found.
        """
        batch_vectors: list[torch.Tensor] = []

        for batch_idx, spans in enumerate(spans_list):
            # Fallback: if spans are empty, use the clean sentence vector
            if not spans:
                batch_vectors.append(q_sentences[batch_idx])
                continue

            token_vectors: list[torch.Tensor] = []
            for token_idx in range(offset_mapping.size(1)):
                tok_s = int(offset_mapping[batch_idx, token_idx, 0])
                tok_e = int(offset_mapping[batch_idx, token_idx, 1])
                if tok_s == tok_e: continue

                is_idiom_part = any(tok_s < end and tok_e > start for start, end in spans)
                if is_idiom_part:
                    token_vectors.append(hidden_states[batch_idx, token_idx])

            if len(token_vectors) == 0:
                batch_vectors.append(q_sentences[batch_idx])
            else:
                batch_vectors.append(torch.stack(token_vectors).mean(dim=0))

        return torch.stack(batch_vectors)

    def encode_raw(self, sentences: List[str], compounds: List[str]) -> torch.Tensor:
        """Main encoding pipeline [Q_sentence ; Q_target]."""
        if len(sentences) != len(compounds):
            raise ValueError("sentences and compounds must have the same length.")

        spans_list = [
            self.find_compound_spans(sentence, compound)
            for sentence, compound in zip(sentences, compounds)
        ]
        
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping")

        device = next(self.encoder.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        offset_mapping = offset_mapping.to(device)

        hidden_states = self.encoder(**encoded).last_hidden_state

        q_sentence = self.mean_pool(
            hidden_states=hidden_states,
            attention_mask=encoded["attention_mask"],
            offset_mapping=offset_mapping,
        )

        q_target = self.target_span_pool(
            hidden_states=hidden_states,
            offset_mapping=offset_mapping,
            spans_list=spans_list,
            q_sentences=q_sentence, # Passed for safe fallback
        )

        q_raw = torch.cat([q_sentence, q_target], dim=-1)
        return q_raw

    def forward(self, sentences: List[str], compounds: List[str]) -> torch.Tensor:
        """Final query projection."""
        q_raw = self.encode_raw(sentences=sentences, compounds=compounds)
        q = self.projection(q_raw)
        return q
