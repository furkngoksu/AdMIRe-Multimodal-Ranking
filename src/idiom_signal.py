from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


def mark_mwe_in_sentence(sentence: str, mwe: str) -> str:
    """Wrap the first case-insensitive MWE match in <mwe> tags."""
    if not mwe:
        return sentence

    pattern = re.compile(re.escape(mwe), flags=re.IGNORECASE)
    match = pattern.search(sentence)
    if match is None:
        return sentence

    start, end = match.span()
    return f"{sentence[:start]}<mwe>{sentence[start:end]}</mwe>{sentence[end:]}"


def build_idiom_input(sentence: str, mwe: str, language: str) -> str:
    marked_sentence = mark_mwe_in_sentence(sentence=sentence, mwe=mwe)
    return f"[LANG={language}] {marked_sentence} [MWE] {mwe}"


class IdiomSignalModel(nn.Module):
    """Lightweight supervised idiomaticity classifier for Stage 1."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.dropout = dropout
        self.encoder_frozen = freeze_encoder

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_dim = self.encoder.config.hidden_size
        classifier_hidden_dim = max(1, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

        self.set_encoder_trainable(trainable=not freeze_encoder)

    def set_encoder_trainable(self, trainable: bool) -> None:
        self.encoder_frozen = not trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if self.encoder_frozen:
            self.encoder.eval()

    def train(self, mode: bool = True) -> "IdiomSignalModel":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    @staticmethod
    def mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode_texts(
        self,
        texts: Sequence[str],
        max_length: int = 128,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {name: value.to(device) for name, value in encoded.items()}

    def forward(
        self,
        sentences: Sequence[str] | None = None,
        mwes: Sequence[str] | None = None,
        *,
        languages: Sequence[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_length: int = 128,
        return_probabilities: bool = False,
    ) -> torch.Tensor:
        """
        Return logits by default. Set return_probabilities=True for M in [0, 1].
        """
        if input_ids is None:
            if sentences is None or mwes is None:
                raise ValueError("Provide either tokenized input_ids or sentence/MWE lists.")
            if len(sentences) != len(mwes):
                raise ValueError("sentences and mwes must have the same length.")

            if languages is None:
                languages = ["UNK"] * len(sentences)
            if len(languages) != len(sentences):
                raise ValueError("languages must have the same length as sentences.")

            texts = [
                build_idiom_input(sentence=sentence, mwe=mwe, language=language)
                for sentence, mwe, language in zip(sentences, mwes, languages)
            ]
            encoded = self._encode_texts(texts=texts, max_length=max_length)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.encoder_frozen:
            with torch.no_grad():
                hidden_states = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        pooled = self.mean_pool(hidden_states=hidden_states, attention_mask=attention_mask)
        logits = self.classifier(pooled)

        if return_probabilities:
            return torch.sigmoid(logits)
        return logits

    @torch.no_grad()
    def predict_m(
        self,
        sentence: str,
        mwe: str,
        language: str,
        max_length: int = 128,
    ) -> float:
        self.eval()
        probability = self(
            sentences=[sentence],
            mwes=[mwe],
            languages=[language],
            max_length=max_length,
            return_probabilities=True,
        )
        return float(probability.squeeze().detach().cpu().item())


class IdiomSignal(IdiomSignalModel):
    """Compatibility wrapper returning M probabilities for Stage 2 fusion code."""

    def forward(
        self,
        sentences: Sequence[str],
        compounds: Sequence[str],
        languages: Sequence[str] | None = None,
        *,
        max_length: int = 128,
    ) -> torch.Tensor:
        return super().forward(
            sentences=sentences,
            mwes=compounds,
            languages=languages,
            max_length=max_length,
            return_probabilities=True,
        )


_LOADED_IDIOM_SIGNAL: IdiomSignalModel | None = None


def load_idiom_signal(
    output_dir: str | Path,
    device: str | torch.device | None = None,
) -> IdiomSignalModel:
    output_path = Path(output_dir)
    config_path = output_path / "config.json"
    checkpoint_path = output_path / "idiom_signal.pt"
    tokenizer_path = output_path / "tokenizer"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = IdiomSignalModel(
        model_name=config.get("model_name", "xlm-roberta-base"),
        dropout=float(config.get("dropout", 0.1)),
        freeze_encoder=True,
    )
    model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=resolved_device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)

    if "classifier_state_dict" in checkpoint:
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    global _LOADED_IDIOM_SIGNAL
    _LOADED_IDIOM_SIGNAL = model
    return model


def compute_m(
    sentence: str,
    mwe: str,
    language: str,
    *,
    model: IdiomSignalModel | None = None,
    max_length: int | None = None,
) -> float:
    active_model = model if model is not None else _LOADED_IDIOM_SIGNAL
    if active_model is None:
        raise ValueError("No idiom signal model is loaded. Call load_idiom_signal first.")

    resolved_max_length = 128 if max_length is None else max_length
    return active_model.predict_m(
        sentence=sentence,
        mwe=mwe,
        language=language,
        max_length=resolved_max_length,
    )
