from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple

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
    # Threshold for fuzzy morphological matching
    TOKEN_SIMILARITY_THRESHOLD = 0.40

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

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(config.model_name)

        # Freeze XLM-R backbone
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        hidden_dim = self.encoder.config.hidden_size
        
        # Build projection head
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
        """Normalizes text: folds case, transliterates Cyrillic, and strips accents."""
        if not text: 
            return ""
        
        text = text.casefold()
        
        for cyr, lat in cls.CYRILLIC_TO_LATIN.items():
            text = text.replace(cyr, lat)
            
        # Standardize Uzbek apostrophes
        text = text.replace('ʻ', "'").replace('’', "'").replace('`', "'")
        
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(char for char in decomposed if not unicodedata.combining(char))

    @staticmethod
    def get_universal_tokens(text: str) -> List[Tuple[str, int, int]]:
        """Extracts language-agnostic tokens, returning (word, start_idx, end_idx)."""
        tokens = []
        pattern = re.compile(r"[\u4e00-\u9FFF]|[^\W_]+", flags=re.UNICODE)
        for match in pattern.finditer(text):
            tokens.append((match.group(0), match.start(), match.end()))
        return tokens

    @staticmethod
    def clean_compound(compound: str) -> str:
        """Cleans trailing numbers/artifacts from datasets."""
        return re.sub(r'\s+\d+$', '', compound)

    def find_compound_spans(self, sentence: str, compound: str) -> List[Tuple[int, int]]:
        """
        Aggressive span extractor guaranteeing 0% fallback. 
        Handles exact, substring, and fuzzy morphological matching.
        """
        compound = self.clean_compound(compound)
        
        # Phase 1: Exact Match
        exact_match = re.search(re.escape(compound), sentence, flags=re.IGNORECASE)
        if exact_match:
            return [exact_match.span()]

        s_tokens = self.get_universal_tokens(sentence)
        c_tokens = self.get_universal_tokens(compound)

        # Failsafe for empty inputs
        if not s_tokens or not c_tokens:
            return [(0, max(1, len(sentence)))]

        matched_spans = []
        
        # Phase 2: Flexible Morphological Match
        for c_word, _, _ in c_tokens:
            norm_c = self.normalize_text(c_word)
            
            # Skip short prepositions/conjunctions
            if len(norm_c) < 3: 
                continue 
            
            best_score = 0.0
            best_span = None

            for s_word, s_start, s_end in s_tokens:
                norm_s = self.normalize_text(s_word)
                
                # Substring check handles heavy agglutination (e.g., Turkish/Uzbek)
                if norm_c in norm_s or norm_s in norm_c:
                    score = 1.0
                else:
                    score = SequenceMatcher(None, norm_c, norm_s).ratio()
                
                if score > best_score:
                    best_score = score
                    best_span = (s_start, s_end)

            if best_score >= self.TOKEN_SIMILARITY_THRESHOLD and best_span:
                matched_spans.append(best_span)

        # Return matches if at least one token passed the threshold
        if matched_spans:
            return sorted(list(set(matched_spans)), key=lambda x: x[0])

        # Phase 3: Ultimate Failsafe (Force best match)
        norm_comp = self.normalize_text(compound)
        best_token_span = (s_tokens[0][1], s_tokens[0][2])
        best_token_score = 0.0
        
        for s_word, s_start, s_end in s_tokens:
            score = SequenceMatcher(None, norm_comp, self.normalize_text(s_word)).ratio()
            if score > best_token_score:
                best_token_score = score
                best_token_span = (s_start, s_end)
                
        return [best_token_span]

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

    def target_span_pool(
        self,
        hidden_states: torch.Tensor,
        offset_mapping: torch.Tensor,
        spans_list: List[List[Tuple[int, int]]],
        q_sentences: torch.Tensor,
    ) -> torch.Tensor:
        """Pool vectors specifically from the target span indices."""
        batch_vectors: list[torch.Tensor] = []

        for batch_idx, spans in enumerate(spans_list):
            if not spans:
                batch_vectors.append(q_sentences[batch_idx])
                continue

            token_vectors: list[torch.Tensor] = []
            for token_idx in range(offset_mapping.size(1)):
                tok_s = int(offset_mapping[batch_idx, token_idx, 0])
                tok_e = int(offset_mapping[batch_idx, token_idx, 1])
                
                if tok_s == tok_e: 
                    continue

                is_idiom_part = any(tok_s < end and tok_e > start for start, end in spans)
                if is_idiom_part:
                    token_vectors.append(hidden_states[batch_idx, token_idx])

            if self.config.debug:
                print(f"DEBUG [Pooling]: Spans {spans} -> {len(token_vectors)} tokens pooled.")

            if len(token_vectors) == 0:
                batch_vectors.append(q_sentences[batch_idx])
            else:
                batch_vectors.append(torch.stack(token_vectors).mean(dim=0))

        return torch.stack(batch_vectors)

    def encode_raw(self, sentences: List[str], compounds: List[str]) -> torch.Tensor:
        """Main encoding pipeline returning [Q_sentence ; Q_target]."""
        if len(sentences) != len(compounds):
            raise ValueError("sentences and compounds must have the same length.")

        spans_list = [
            self.find_compound_spans(sentence, compound)
            for sentence, compound in zip(sentences, compounds)
        ]
        
        if self.config.debug:
            for s, c, sp in zip(sentences, compounds, spans_list):
                print(f"DEBUG [Spans]: '{c}' in '{s}' -> {sp}")

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
            q_sentences=q_sentence,
        )

        return torch.cat([q_sentence, q_target], dim=-1)

    def forward(self, sentences: List[str], compounds: List[str]) -> torch.Tensor:
        """Final query projection."""
        q_raw = self.encode_raw(sentences=sentences, compounds=compounds)
        return self.projection(q_raw)