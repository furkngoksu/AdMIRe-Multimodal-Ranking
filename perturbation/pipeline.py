
from __future__ import annotations

import os
import re
import json
import time
import hashlib
import threading
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import stopwordsiso
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel



CACHE_DIR = Path("outputs/m_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_NAME = "xlm-roberta-base"

LANG_ROUTE = {  # language → (model_key, "deepseek" | "openrouter")
    "Russian":   ("deepseek-chat",            "deepseek"),
    "Serbian":   ("deepseek-chat",            "deepseek"),
    "Kazakh":    ("deepseek-chat",            "deepseek"),
    "Slovak":    ("deepseek-chat",            "deepseek"),
    "Georgian":  ("deepseek-chat",            "deepseek"),
}
DEFAULT_ROUTE = ("google/gemini-2.5-flash", "openrouter")

PARAMS = {
    "deepseek-chat":            {"temperature": 1.0, "top_p": 1.0},
    "google/gemini-2.5-flash":  {"temperature": 0.5, "top_p": 1.0},
}

# Map free-form language label (e.g. "Portuguese-Brazil") to canonical name
LANG_CANONICAL = {
    "Portuguese-Brazil": "Portuguese", "Portuguese-Portugal": "Portuguese",
    "Spanish-Ecuador": "Spanish",
}

LANG_ISO = {
    "Chinese": "zh", "English": "en", "Georgian": "ka", "Greek": "el",
    "Igbo": "ig", "Kazakh": "kk", "Norwegian": "no", "Portuguese": "pt",
    "Russian": "ru", "Serbian": "sr", "Slovak": "sk", "Slovenian": "sl",
    "Spanish": "es", "Turkish": "tr", "Uzbek": "uz",
}


# ============================================================================
# Helpers
# ============================================================================

def canon_lang(lang: str) -> str:
    return LANG_CANONICAL.get(lang, lang)

def hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8")); h.update(b"\x00")
    return h.hexdigest()


# ============================================================================
# Disk cache (JSONL append + dict load)
# ============================================================================

class JsonlCache:
    """Thread-safe append-only JSONL cache. Reads existing file on construction."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                        if "key" in rec:
                            self._mem[rec["key"]] = rec["value"]
                    except json.JSONDecodeError:
                        continue

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._mem.get(key)

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            if key in self._mem:
                return  # already cached, don't double-write
            self._mem[key] = value
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ============================================================================
# LLM dispatcher
# ============================================================================

class Dispatcher:
    def __init__(self) -> None:
        ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.deepseek = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com") if ds_key else None
        self.openrouter = OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1") if or_key else None

    def model_for(self, lang: str) -> Tuple[str, str]:
        return LANG_ROUTE.get(canon_lang(lang), DEFAULT_ROUTE)

    def call(self, lang: str, system: str, user: str, max_tokens: int = 350,
             max_retries: int = 3) -> Tuple[str, Optional[str]]:
        model_id, provider = self.model_for(lang)
        client = self.deepseek if provider == "deepseek" else self.openrouter
        if client is None:
            return "", f"missing api key for provider={provider}"
        params = dict(PARAMS[model_id])
        params["max_tokens"] = max_tokens
        kwargs = dict(model=model_id,
                      messages=[{"role": "system", "content": system},
                                {"role": "user",   "content": user}],
                      **params)
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                return content.strip(), None
            except Exception as e:
                last_err = str(e)
                time.sleep(1.5 * (attempt + 1))
        return "", last_err


# ============================================================================
# Prompts
# ============================================================================

def sys_edit(lang: str) -> str:
    return (f"You are a controlled text-editing tool. The text is in {lang}. "
            f"You MUST output ONLY the rewritten sentence in {lang}, on a single "
            f"line, with no quotes, no explanation, no prefix, no labels. "
            f"Preserve the rest of the sentence verbatim.")

def sys_desc(lang: str) -> str:
    return (f"You are a literal-meaning describer. The expression is in {lang}. "
            f"Write the description IN {lang}. Output ONE short literal sentence "
            f"in {lang} describing the surface meaning of the expression. "
            f"No idiomatic interpretation. No English. Output only the {lang} sentence.")

def sys_neighbors(lang: str) -> str:
    return (f"You generate semantically related vocabulary in {lang}. "
            f"Output ONLY valid JSON, no commentary, no markdown. "
            f"All output words must be in {lang} (single words, no phrases).")

def sys_judge(lang: str) -> str:
    return (f"You are a strict evaluator of literalization rewrites in {lang}. "
            f"Output ONLY valid JSON: "
            f'{{"surface_ok": true|false, "grammatical_ok": true|false}}.')


def usr_lex(s, c, lang):
    return (f"Language: {lang}\nSentence: {s}\nTarget expression: {c}\n\n"
            f"Rewrite the sentence by replacing exactly ONE word INSIDE the target "
            f"expression with a close, plausible alternative word in the same language. "
            f"Change nothing else.")

def usr_drop(s, c, lang):
    return (f"Language: {lang}\nSentence: {s}\nTarget expression: {c}\n\n"
            f"Rewrite the sentence by REMOVING ONE CONTENT WORD from the target "
            f"expression while keeping the sentence grammatical. "
            f"Do not change anything outside the target expression.")

def usr_lit(s, c, lang):
    return (f"Language: {lang}\nSentence: {s}\nTarget expression: {c}\n\n"
            f"Interpret the target expression literally, using only the ordinary "
            f"dictionary meanings of its words and ignoring any idiomatic, "
            f"figurative, or implied meaning. ALWAYS rewrite the sentence by "
            f"replacing the target expression with a phrase describing this "
            f"literal interpretation. Do NOT add parenthetical glosses or "
            f"translations — output the rewritten sentence in {lang} directly. "
            f"The result may be absurd, surreal, or pragmatically odd; this is "
            f"expected.")

def usr_para(s, c, lang):
    return (f"Language: {lang}\nSentence: {s}\nTarget expression: {c}\n\n"
            f"Paraphrase the sentence so the meaning is preserved. "
            f"The target expression and its meaning must be kept intact. "
            f"Use different surrounding wording.")

def usr_desc(c, lang):
    return (f"Expression ({lang}): {c}\n\n"
            f"Describe the LITERAL surface meaning of this expression in one short "
            f"sentence written in {lang}. Ignore any idiomatic meaning.")

def usr_neighbors(c, lang):
    return (f"Expression ({lang}): \"{c}\"\n\n"
            f"Output JSON with exactly these keys, each a list of 3 single {lang} words "
            f"semantically related to the LITERAL surface meaning (NOT idiomatic):\n"
            f"  category_same   — words for the same semantic category as the noun(s) in the expression\n"
            f"  attributes      — words for attributes/states of the literal scene\n"
            f"  actions         — words for actions associated with the literal scene\n"
            f"  superordinate   — superordinate / hypernym words\n\n"
            f"Example for English 'rotten apple': "
            f'{{"category_same":["pear","banana","peach"],'
            f'"attributes":["fresh","ripe","unripe"],'
            f'"actions":["eat","peel","bite"],'
            f'"superordinate":["fruit","produce","food"]}}')

def usr_judge(orig, lit, comp, lang):
    return (f"Language: {lang}\n"
            f"Compound: {comp}\n"
            f"Original: {orig}\n"
            f"Literalized: {lit}\n\n"
            f"Q1 (surface_ok): Does the literalized version interpret the compound at "
            f"its SURFACE / PHYSICAL meaning (NOT its figurative meaning)?\n"
            f"Q2 (grammatical_ok): Is the literalized sentence grammatical in {lang}?\n\n"
            f"Reply with JSON only.")


# ============================================================================
# Verifiers
# ============================================================================

WORD_RE = re.compile(r"[\w'']+", re.UNICODE)


def tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def find_compound_span(sent_tokens: List[str], compound_tokens: List[str]) -> Optional[Tuple[int, int]]:
    """Return (start, end_exclusive) span of compound in sent_tokens (prefix match per word).
    Allows up to 1 token gap between consecutive compound words."""
    if not compound_tokens:
        return None
    n = len(sent_tokens)
    for start in range(n):
        i = start
        ci = 0
        gaps = 0
        while i < n and ci < len(compound_tokens):
            if sent_tokens[i].startswith(compound_tokens[ci]):
                ci += 1
                i += 1
            else:
                if ci == 0:
                    break
                gaps += 1
                i += 1
                if gaps > 1:
                    break
        if ci == len(compound_tokens):
            return (start, i)
    return None


def verify_lex_sub(orig: str, out: str, compound: str) -> Tuple[bool, str]:
    if not out or out.strip() == orig.strip():
        return False, "empty or unchanged"
    o = tokens(orig); v = tokens(out); c = tokens(compound)
    if len(o) != len(v):
        return False, f"length changed ({len(o)}->{len(v)})"
    diffs = [i for i in range(len(o)) if o[i] != v[i]]
    if len(diffs) != 1:
        return False, f"{len(diffs)} tokens changed (expected 1)"
    span = find_compound_span(o, c)
    if not span:
        return True, "compound span not found, accepting (lenient)"
    if not (span[0] <= diffs[0] < span[1]):
        return False, f"changed token {diffs[0]} outside compound span {span}"
    return True, "ok"


def verify_comp_drop(orig: str, out: str, compound: str) -> Tuple[bool, str]:
    if not out or out.strip() == orig.strip():
        return False, "empty or unchanged"
    o = tokens(orig); v = tokens(out); c = tokens(compound)
    if len(v) >= len(o):
        return False, f"length not reduced ({len(o)}->{len(v)})"
    if len(o) - len(v) != 1:
        return False, f"more than 1 token dropped ({len(o)-len(v)})"
    span = find_compound_span(o, c)
    if not span:
        return True, "compound span not found, accepting (lenient)"
    j = 0; dropped_idx = None
    for i in range(len(o)):
        if j < len(v) and o[i] == v[j]:
            j += 1
        else:
            if dropped_idx is not None:
                return False, "multiple drops"
            dropped_idx = i
    if dropped_idx is None or not (span[0] <= dropped_idx < span[1]):
        return False, f"dropped token outside compound span"
    return True, "ok"


def verify_paraphrase(orig: str, out: str) -> Tuple[bool, str]:
    if not out: return False, "empty"
    if out.strip() == orig.strip(): return False, "unchanged"
    if len(out) < 0.4 * len(orig) or len(out) > 2.5 * len(orig):
        return False, "length out of range"
    return True, "ok"


def verify_lit_desc(out: str) -> Tuple[bool, str]:
    if not out or len(out.strip()) < 3:
        return False, "empty or too short"
    return True, "ok"


# ============================================================================
# LLM-as-judge for literalize
# ============================================================================

def parse_judge_json(text: str) -> Optional[dict]:
    if not text: return None
    # try to extract first JSON object in text
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m: return None
    try:
        d = json.loads(m.group(0))
        return {"surface_ok": bool(d.get("surface_ok", False)),
                "grammatical_ok": bool(d.get("grammatical_ok", False))}
    except json.JSONDecodeError:
        return None


def judge_literalize(disp: Dispatcher, orig: str, lit: str, comp: str, lang: str) -> dict:
    raw, err = disp.call(lang, sys_judge(lang), usr_judge(orig, lit, comp, lang), max_tokens=80)
    parsed = parse_judge_json(raw)
    if parsed is None:
        # one strict-format retry
        raw2, _ = disp.call(lang, sys_judge(lang),
                            usr_judge(orig, lit, comp, lang) +
                            "\n\nIMPORTANT: Output ONLY the JSON object, nothing else.",
                            max_tokens=80)
        parsed = parse_judge_json(raw2) or {"surface_ok": False, "grammatical_ok": False}
        return {"raw": raw2, "parsed": parsed, "err": err, "retried": True}
    return {"raw": raw, "parsed": parsed, "err": err, "retried": False}


# ============================================================================
# Generation orchestration with retries + caching
# ============================================================================

@dataclass
class GenResult:
    text: str
    valid: bool
    reason: str
    raw_outputs: List[str] = field(default_factory=list)
    judge: Optional[dict] = None


_PERT_KEYS = {"lex_sub", "comp_drop", "literalize", "paraphrase", "literal_desc"}


def generate_perturbations(disp: Dispatcher, sentence: str, compound: str, lang: str,
                           cache: JsonlCache) -> Dict[str, GenResult]:
    key = hash_key(canon_lang(lang), compound, sentence)
    cached = cache.get(key)
    if cached:
        # Skip metadata keys (sentence/compound/lang) added in newer cache format
        return {k: GenResult(**v) for k, v in cached.items() if k in _PERT_KEYS}

    out: Dict[str, GenResult] = {}

    def gen(name, sysmsg, usrmsg, verifier, retry_extra=""):
        raws = []
        reason = "no successful API response"
        for attempt in range(2):  # initial + 1 retry
            text, err = disp.call(lang, sysmsg, usrmsg + retry_extra, max_tokens=350)
            raws.append(text)
            if err:
                reason = f"api error: {err[:120]}"
                continue
            valid, reason = verifier(text)
            if valid:
                return GenResult(text=text, valid=True, reason=reason, raw_outputs=raws)
            retry_extra = "\n\nIMPORTANT: Your previous attempt was rejected because: " + reason + ". Try again, strictly following all rules."
        return GenResult(text=raws[-1] if raws else "", valid=False, reason=reason, raw_outputs=raws)

    out["lex_sub"] = gen("lex_sub", sys_edit(lang), usr_lex(sentence, compound, lang),
                          lambda t: verify_lex_sub(sentence, t, compound))
    out["comp_drop"] = gen("comp_drop", sys_edit(lang), usr_drop(sentence, compound, lang),
                            lambda t: verify_comp_drop(sentence, t, compound))
    out["paraphrase"] = gen("paraphrase", sys_edit(lang), usr_para(sentence, compound, lang),
                             lambda t: verify_paraphrase(sentence, t))
    out["literal_desc"] = gen("literal_desc", sys_desc(lang), usr_desc(compound, lang),
                                verify_lit_desc)

    # Literalize uses code-light pre-check + LLM judge
    lit_raws = []
    lit_text = ""
    lit_judge = None
    lit_valid = False
    lit_reason = "judge failed"
    for attempt in range(2):
        text, err = disp.call(lang, sys_edit(lang), usr_lit(sentence, compound, lang), max_tokens=350)
        lit_raws.append(text)
        if not text or text.strip() == sentence.strip():
            lit_reason = "empty or unchanged"; continue
        judge = judge_literalize(disp, sentence, text, compound, lang)
        if judge["parsed"]["surface_ok"] and judge["parsed"]["grammatical_ok"]:
            lit_text = text; lit_judge = judge
            lit_valid = True; lit_reason = "ok"
            break
        lit_judge = judge; lit_text = text
        lit_reason = f"judge: {judge['parsed']}"
    out["literalize"] = GenResult(text=lit_text, valid=lit_valid, reason=lit_reason,
                                  raw_outputs=lit_raws, judge=lit_judge)

    cache.put(key, {
        "sentence": sentence,
        "compound": compound,
        "lang": canon_lang(lang),
        **{k: asdict(v) for k, v in out.items()}
    })
    return out


def generate_literal_set(disp: Dispatcher, compound: str, lang: str,
                         cache_set: JsonlCache) -> List[str]:
    key = hash_key(canon_lang(lang), compound)
    cached = cache_set.get(key)
    if cached:
        return cached["L"]

    raw, err = disp.call(lang, sys_neighbors(lang), usr_neighbors(compound, lang), max_tokens=400)
    parsed = None
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m: parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        parsed = None
    if not parsed:
        raw2, _ = disp.call(lang, sys_neighbors(lang),
                            usr_neighbors(compound, lang) + "\n\nOutput ONLY valid JSON.",
                            max_tokens=400)
        try:
            m = re.search(r"\{.*\}", raw2, re.DOTALL)
            if m: parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            parsed = None
    items = []
    if parsed:
        for k in ("category_same", "attributes", "actions", "superordinate"):
            v = parsed.get(k, [])
            if isinstance(v, list):
                items.extend(str(x).strip() for x in v if isinstance(x, (str, int)))
    L = list(dict.fromkeys(tokens(compound) + [w.lower() for w in items if w]))
    cache_set.put(key, {"L": L, "raw": raw, "parsed": parsed})
    return L


# ============================================================================
# XLM-R encoding utilities
# ============================================================================

class XLMREncoder:
    def __init__(self, name: str = ENCODER_NAME, device: Optional[str] = None):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def encode_sentence(self, text: str) -> Tuple[torch.Tensor, dict]:
        """Returns (last_hidden [seq_len, dim], encoding dict with offset_mapping)."""
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True,
                       truncation=True, max_length=256)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state[0].cpu()  # [seq_len, dim]
        return out, {"offsets": offsets, "tokens": self.tok.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())}

    @torch.no_grad()
    def encode_pooled(self, text: str) -> torch.Tensor:
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        mask = enc["attention_mask"][0].float()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        h = self.model(**enc).last_hidden_state[0].cpu()  # [seq, dim]
        v = (h * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(min=1.0)
        return v

    @torch.no_grad()
    def encode_window(self, sentence: str, char_span: Tuple[int, int],
                      window_words: int) -> Optional[torch.Tensor]:
        """Mean-pool XLM-R hidden states over a region around `char_span`,
        widened by `window_words` whitespace-words on each side. Returns None
        if no subword tokens fall inside the region (caller falls back)."""
        word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", sentence)]
        if not word_spans: return None
        cs, ce = char_span
        # word indices that overlap the char span
        overlap = [i for i, (s, e) in enumerate(word_spans)
                   if not (e <= cs or s >= ce)]
        if not overlap:
            # find the closest word to span instead
            return None
        start_w = max(0, overlap[0] - window_words)
        end_w   = min(len(word_spans), overlap[-1] + 1 + window_words)
        wider_cs = word_spans[start_w][0]
        wider_ce = word_spans[end_w - 1][1]
        h, meta = self.encode_sentence(sentence)
        mask = []
        for (s, e) in meta["offsets"]:
            if s == 0 and e == 0:
                mask.append(False); continue
            mask.append(not (e <= wider_cs or s >= wider_ce))
        if not any(mask):
            return None
        idxs = [i for i, m in enumerate(mask) if m]
        return h[idxs].mean(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float(); b = b.float()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ============================================================================
# Compound span on subword tokens
# ============================================================================

def find_char_span(sentence: str, compound: str) -> Optional[Tuple[int, int]]:
    """Locate compound span in sentence by case-insensitive prefix-matching its words.
    Returns (char_start, char_end_exclusive) or None."""
    sent_low = sentence.lower()
    words = compound.lower().split()
    if not words: return None
    # 1) try simple lowercase substring
    s = sent_low.find(compound.lower())
    if s >= 0:
        return (s, s + len(compound))
    # 2) prefix-match per word, allow up to 1-token gap, in order
    word_iter = re.finditer(r"\S+", sent_low)
    word_list = [(m.start(), m.end(), m.group(0)) for m in word_iter]
    n = len(word_list)
    for start in range(n):
        i = start
        ci = 0
        gaps = 0
        last_end = None
        while i < n and ci < len(words):
            if word_list[i][2].startswith(words[ci]):
                ci += 1
                if ci == 1: first_start = word_list[i][0]
                last_end = word_list[i][1]
                i += 1
            else:
                if ci == 0:
                    break
                gaps += 1
                i += 1
                if gaps > 1: break
        if ci == len(words):
            return (first_start, last_end)
    return None


def compound_token_mask(offsets, char_span: Tuple[int, int]) -> List[bool]:
    """True where the subword token overlaps the compound's char span."""
    cs, ce = char_span
    mask = []
    for (s, e) in offsets:
        if s == 0 and e == 0:
            mask.append(False)  # special token
            continue
        # overlap test
        mask.append(not (e <= cs or s >= ce))
    return mask


# ============================================================================
# Per-word vector pooling (subword aggregation)
# ============================================================================

def pool_per_word(hidden: torch.Tensor, offsets: List[Tuple[int, int]],
                  text: str) -> List[Tuple[str, torch.Tensor, Tuple[int, int]]]:
    """Group consecutive subwords belonging to the same whitespace-word and mean-pool their vectors.
    Returns list of (word_text, mean_vector, char_span)."""
    out = []
    word_iter = re.finditer(r"\S+", text)
    word_spans = [(m.start(), m.end(), m.group(0)) for m in word_iter]
    for ws, we, wt in word_spans:
        idxs = []
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0: continue  # special tokens
            if s >= ws and e <= we and not (e <= ws or s >= we):
                idxs.append(i)
        if not idxs: continue
        v = hidden[idxs].mean(0)
        out.append((wt, v, (ws, we)))
    return out


# ============================================================================
# Stopwords helper
# ============================================================================

def is_stopword(word: str, lang: str) -> bool:
    iso = LANG_ISO.get(canon_lang(lang))
    if not iso: return False
    try:
        sw = stopwordsiso.stopwords(iso)
    except Exception:
        return False
    return word.lower() in sw


# ============================================================================
# Feature computation
# ============================================================================

@dataclass
class MFeatures:
    S_sub: float; S_comp: float; S_lit: float; S_ctrl: float; S_lsc: float
    B_sub: float; B_comp: float; B_lit: float
    feature_mask: Dict[str, int]


def _span_or_full(enc: XLMREncoder, text: str, char_span: Optional[Tuple[int, int]],
                  window: int, fallback_text: str) -> Tuple[torch.Tensor, str]:
    """Encode either a windowed compound span or fall back to whole-sentence pooling."""
    if char_span is not None:
        v = enc.encode_window(text, char_span, window)
        if v is not None:
            return v, "span"
    return enc.encode_pooled(fallback_text), "full"


def compute_features(enc: XLMREncoder, sentence: str, compound: str, lang: str,
                     perts: Dict[str, GenResult], L_words: List[str],
                     window: int = 0) -> Tuple[MFeatures, dict]:
    """Span-focused features. `window=0` uses just the compound;
    `window=1` widens by 1 whitespace-word on each side, etc.
    For variants where the compound can't be located (literalize, sometimes paraphrase),
    falls back to whole-sentence mean-pool."""
    # 1) Locate compound in the original sentence
    orig_span = find_char_span(sentence, compound)
    src_lookup = {
        "lex_sub":   perts["lex_sub"].text   or sentence,
        "comp_drop": perts["comp_drop"].text or sentence,
        "literalize": perts["literalize"].text or sentence,
        "paraphrase": perts["paraphrase"].text or sentence,
    }
    # For each variant, try to locate the compound (or its modified form) inside it.
    # lex_sub: one word changed → original compound likely no longer matches verbatim,
    #   but prefix-matching find_char_span tolerates that for 1-word substitutions.
    # comp_drop: one word dropped → reduced form often still found by prefix match.
    # paraphrase: usually preserved.
    # literalize: typically rewritten — find_char_span returns None → fallback.
    spans = {k: find_char_span(t, compound) for k, t in src_lookup.items()}

    # Encode original at chosen window (or fallback)
    v_orig, origin = _span_or_full(enc, sentence, orig_span, window, sentence)
    v_sub,  src_sub  = _span_or_full(enc, src_lookup["lex_sub"],   spans["lex_sub"],   window, src_lookup["lex_sub"])
    v_drop, src_drop = _span_or_full(enc, src_lookup["comp_drop"], spans["comp_drop"], window, src_lookup["comp_drop"])
    v_lit,  src_lit  = _span_or_full(enc, src_lookup["literalize"], spans["literalize"], window, src_lookup["literalize"])
    v_para, src_para = _span_or_full(enc, src_lookup["paraphrase"], spans["paraphrase"], window, src_lookup["paraphrase"])

    S_sub  = cosine(v_orig, v_sub)
    S_comp = cosine(v_orig, v_drop)
    S_lit  = cosine(v_orig, v_lit)
    S_ctrl = cosine(v_orig, v_para)

    # Signed differences — no clamping. With span-focused encoding, S_ctrl ≈ 1 by
    # construction (paraphrase preserves the compound), so B values are ≥ 0 anyway.
    B_sub  = S_ctrl - S_sub
    B_comp = S_ctrl - S_comp
    B_lit  = S_ctrl - S_lit

    # 2) S_lsc — Liu & Hwa Eq.1 with XLM-R + LLM-built L
    h_sent, meta = enc.encode_sentence(sentence)
    span = find_char_span(sentence, compound)
    pooled = pool_per_word(h_sent, meta["offsets"], sentence)

    # build C: words outside compound, not stopword
    if span is None:
        C_vecs = [v for (_, v, _) in pooled if not is_stopword(_, lang)]
    else:
        cs, ce = span
        C_vecs = []
        for word, v, (ws, we) in pooled:
            inside = not (we <= cs or ws >= ce)
            if inside: continue
            if is_stopword(word, lang): continue
            C_vecs.append(v)

    # build L_vecs: encode each L word with light template "{word}." (skip stopwords)
    L_vecs = []
    L_used = []
    for w in L_words:
        if is_stopword(w, lang): continue
        if not w.strip(): continue
        # simple bare word
        L_vecs.append(enc.encode_pooled(w))
        L_used.append(w)

    if C_vecs and L_vecs:
        C = torch.stack(C_vecs); L = torch.stack(L_vecs)
        # normalized cosine matrix
        C_n = torch.nn.functional.normalize(C, dim=1)
        L_n = torch.nn.functional.normalize(L, dim=1)
        sim = (C_n @ L_n.T)  # [|C|, |L|]
        S_lsc = float(sim.mean().item())
    else:
        S_lsc = 0.0

    # Failure mask: only zero out S_lsc when we couldn't build C or L.
    # B_* are kept signed; classifier learns from variance, not from a hard threshold.
    mask = {
        "B_sub":  1,
        "B_comp": 1,
        "B_lit":  1,
        "S_lsc":  1 if (C_vecs and L_vecs) else 0,
    }

    feats = MFeatures(S_sub=S_sub, S_comp=S_comp, S_lit=S_lit, S_ctrl=S_ctrl, S_lsc=S_lsc,
                      B_sub=B_sub, B_comp=B_comp, B_lit=B_lit, feature_mask=mask)

    # Debug: list non-idiom non-stopword context words actually included in C
    if span is None:
        c_words = [w for (w, _, _) in pooled if not is_stopword(w, lang)]
    else:
        cs, ce = span
        c_words = [w for (w, _, (ws, we)) in pooled
                   if (we <= cs or ws >= ce) and not is_stopword(w, lang)]
    debug = {
        "compound_char_span": span,
        "window_size": window,
        "encoding_origin": {
            "orig": origin, "lex_sub": src_sub, "comp_drop": src_drop,
            "literalize": src_lit, "paraphrase": src_para,
        },
        "variant_spans": spans,
        "C_words": c_words,
        "L_used": L_used,
        "C_size": len(C_vecs),
        "L_size": len(L_vecs),
    }
    return feats, debug


# ============================================================================
# End-to-end runner for one example
# ============================================================================

def run_one(disp: Dispatcher, enc: XLMREncoder, sentence: str, compound: str, lang: str,
            llm_cache: JsonlCache, set_cache: JsonlCache, window: int = 0) -> dict:
    # Step 1: generate perturbations + L set
    perts = generate_perturbations(disp, sentence, compound, lang, llm_cache)
    L_words = generate_literal_set(disp, compound, lang, set_cache)

    # Step 2: compute features
    feats, debug = compute_features(enc, sentence, compound, lang, perts, L_words, window=window)

    # Step 3: untrained-classifier M (uniform weights, illustrative only)
    M_uniform = float(1.0 / (1.0 + np.exp(-(feats.B_sub + feats.B_comp + feats.B_lit + feats.S_lsc))))

    return {
        "lang": lang, "compound": compound, "sentence": sentence,
        "perturbations": {k: {"text": v.text, "valid": v.valid, "reason": v.reason,
                              "judge": v.judge, "raw_outputs": v.raw_outputs}
                          for k, v in perts.items()},
        "literal_set_L": L_words,
        "features": {
            "S_sub": feats.S_sub, "S_comp": feats.S_comp, "S_lit": feats.S_lit,
            "S_ctrl": feats.S_ctrl, "S_lsc": feats.S_lsc,
            "B_sub": feats.B_sub, "B_comp": feats.B_comp, "B_lit": feats.B_lit,
        },
        "feature_mask": feats.feature_mask,
        "debug": debug,
        "M_untrained_uniform": M_uniform,
        "models": {
            "generator": disp.model_for(lang)[0],
            "judge": disp.model_for(lang)[0],
            "encoder": ENCODER_NAME,
        },
    }
