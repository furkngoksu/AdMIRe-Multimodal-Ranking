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


class QueryEncoder(nn.Module):
    FUZZY_SPAN_THRESHOLD = 0.80

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
        self.projection = nn.Linear(hidden_dim * 2, config.shared_dim)

    @staticmethod
    def find_compound_span(sentence: str, compound: str) -> Tuple[int, int]:
        """Case-insensitive character offset of compound in sentence."""
        start = sentence.lower().find(compound.lower())
        if start != -1:
            return start, start + len(compound)

        fuzzy_span = QueryEncoder.find_fuzzy_compound_span(
            sentence=sentence,
            compound=compound,
        )
        if fuzzy_span is None:
            raise ValueError(f"compound not found in sentence: {compound}")

        return fuzzy_span

    @staticmethod
    def normalize_for_match(value: str) -> str:
        """Normalize text for approximate span matching."""
        decomposed = unicodedata.normalize("NFKD", value.casefold())
        return "".join(
            char for char in decomposed
            if not unicodedata.combining(char)
        )

    @staticmethod
    def word_spans(text: str) -> list[tuple[str, int, int]]:
        """Return word tokens with original character spans."""
        return [
            (match.group(0), match.start(), match.end())
            for match in re.finditer(r"\w+", text, flags=re.UNICODE)
        ]

    @staticmethod
    def find_fuzzy_compound_span(
        sentence: str,
        compound: str,
    ) -> Tuple[int, int] | None:
        """Find a close inflected/surface variant of the compound in the sentence."""
        compound_tokens = QueryEncoder.word_spans(compound)
        sentence_tokens = QueryEncoder.word_spans(sentence)

        window_size = len(compound_tokens)
        if window_size == 0 or len(sentence_tokens) < window_size:
            return None

        normalized_compound = QueryEncoder.normalize_for_match(compound)
        best_score = 0.0
        best_span: tuple[int, int] | None = None

        for start_idx in range(0, len(sentence_tokens) - window_size + 1):
            window = sentence_tokens[start_idx : start_idx + window_size]
            window_text = " ".join(token for token, _, _ in window)
            normalized_window = QueryEncoder.normalize_for_match(window_text)

            score = SequenceMatcher(
                None,
                normalized_compound,
                normalized_window,
            ).ratio()

            if score > best_score:
                best_score = score
                best_span = (window[0][1], window[-1][2])

        if best_span is None or best_score < QueryEncoder.FUZZY_SPAN_THRESHOLD:
            return None

        return best_span

    @staticmethod
    def mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool over real sentence tokens, excluding padding and special tokens."""
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
        spans: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Mean pool over tokens overlapping the compound span."""
        batch_vectors: list[torch.Tensor] = []

        for batch_idx, span in enumerate(spans):
            target_start, target_end = span
            token_vectors: list[torch.Tensor] = []

            for token_idx in range(offset_mapping.size(1)):
                tok_s = int(offset_mapping[batch_idx, token_idx, 0])
                tok_e = int(offset_mapping[batch_idx, token_idx, 1])

                if tok_s == tok_e:
                    continue

                overlaps_target = tok_s < target_end and tok_e > target_start

                if overlaps_target:
                    token_vectors.append(hidden_states[batch_idx, token_idx])

            if len(token_vectors) == 0:
                raise ValueError("No tokenizer tokens overlap the compound span.")

            batch_vectors.append(torch.stack(token_vectors).mean(dim=0))

        return torch.stack(batch_vectors)

    def encode_raw(self, sentences: List[str], compounds: List[str]) -> torch.Tensor:
        """Build [Q_sentence ; Q_target] before the final query projection."""
        if len(sentences) != len(compounds):
            raise ValueError("sentences and compounds must have the same length.")

        spans = [
            self.find_compound_span(sentence, compound)
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
            spans=spans,
        )

        q_raw = torch.cat([q_sentence, q_target], dim=-1)
        return q_raw

    def forward(self, sentences: List[str], compounds: List[str]) -> torch.Tensor:
        """Encode sentence-compound pairs into query vectors [B, shared_dim]."""
        q_raw = self.encode_raw(sentences=sentences, compounds=compounds)
        q = self.projection(q_raw)

        q = F.normalize(q, p=2, dim=-1)

        if self.config.debug:
            print(
                "[QueryEncoder] "
                f"device={q.device} "
                f"batch={len(sentences)} "
            )
            print("[QueryEncoder] sentence encoded, target pooled, combined into Q")

        return q
