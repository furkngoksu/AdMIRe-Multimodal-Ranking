from __future__ import annotations
import csv, json, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# --- GENERIC CONFIG ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Model Paths
MODEL_PATH = REPO_ROOT / "stage1_idiom_signal/fusion_mlp/outputs/idiom_signal.pt"
TRANSFORMER_NAME = "xlm-roberta-base"

# Data Directories
INPUT_LLM_DIR = REPO_ROOT / "llm_results_submission/enriched"
INPUT_TSV_DIR = REPO_ROOT / "admire2_submission/admire2_sentences"
OUTPUT_FILE = REPO_ROOT / "stage1_idiom_signal_final/output/submission_m_scores.tsv"

# Runtime Config
BATCH_SIZE = 16
TEXT_KEYS = ["hybrid_input", "target", "paraphrase", "lex_sub", "comp_drop", "literalize"]

class GenericLLMLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data: Dict[str, Dict[str, Any]] = {}
    def load(self):
        for path in self.data_dir.glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    sent = rec.get("sentence", "").strip()
                    if sent: self.data[sent] = rec
        print(f"[Loader] Indexed {len(self.data)} samples from {self.data_dir.name}")
    def get_rewrites(self, sentence: str) -> Optional[Dict[str, str]]:
        entry = self.data.get(sentence.strip())
        if not entry: return None
        return {k: entry.get(k, {}).get("text", sentence) for k in ["lex_sub", "comp_drop", "paraphrase", "literalize"]}

def mean_pool(hidden_states, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden_states * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

def mark_compound(sentence, compound):
    match = re.search(re.escape(compound), sentence, flags=re.IGNORECASE)
    if not match: return sentence
    s, e = match.span()
    return f"{sentence[:s]}<mwe>{sentence[s:e]}</mwe>{sentence[e:]}"

def build_classifier(input_dim, params):
    return nn.Sequential(
        nn.Linear(input_dim, params["hidden_1"]), nn.BatchNorm1d(params["hidden_1"]), nn.GELU(), nn.Dropout(params["dropout_1"]),
        nn.Linear(params["hidden_1"], params["hidden_2"]), nn.BatchNorm1d(params["hidden_2"]), nn.GELU(), nn.Dropout(params["dropout_2"]),
        nn.Linear(params["hidden_2"], 1),
    )

def main():
    # 1. Load M Model
    payload = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classifier = build_classifier(payload["input_dim"], payload["hyperparameters"]).to(DEVICE)
    classifier.load_state_dict(payload["classifier_state_dict"])
    classifier.eval()

    # 2. Load LLM Data
    loader = GenericLLMLoader(INPUT_LLM_DIR)
    loader.load()

    # 3. Prepare TSV Samples
    all_samples, missing_count = [], 0
    for path in sorted(INPUT_TSV_DIR.glob("*.tsv")):
        lang = path.stem.replace("submission_", "")
        with path.open("r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                sent, comp = row["sentence"].strip(), row["compound"]
                rewrites = loader.get_rewrites(sent)
                if not rewrites:
                    missing_count += 1
                    rewrites = {k: sent for k in ["lex_sub", "comp_drop", "paraphrase", "literalize"]}
                all_samples.append({"sample_id": f"{lang}_{i}", "hybrid_input": f"[LANG={lang}] {mark_compound(sent, comp)} [MWE] {comp}", "target": sent, **rewrites})
    
    if missing_count > 0:
        print(f"ERROR: {missing_count} missing samples in {INPUT_LLM_DIR.name}. Aborting."); sys.exit(1)

    # 4. Inference
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    encoder = AutoModel.from_pretrained(TRANSFORMER_NAME).to(DEVICE)
    encoder.eval()
    probs = []
    for i in tqdm(range(0, len(all_samples), BATCH_SIZE), desc="Processing"):
        batch = all_samples[i : i + BATCH_SIZE]
        texts = [b[k] for b in batch for k in TEXT_KEYS]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(**enc).last_hidden_state
            p = mean_pool(out, enc["attention_mask"]).view(len(batch), len(TEXT_KEYS), -1)
            h, t = p[:, 0], p[:, 1]
            feat = torch.cat([h, torch.abs(t-p[:,2]), torch.abs(t-p[:,3]), torch.abs(t-p[:,4]), torch.abs(t-p[:,5])], dim=-1)
            res = torch.sigmoid(classifier(feat).squeeze(-1)).cpu().tolist()
            probs.extend(res if isinstance(res, list) else [res])

    # 5. Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "m_prob"], delimiter="\t")
        writer.writeheader()
        for rec, p in zip(all_samples, probs):
            writer.writerow({"sample_id": rec["sample_id"], "m_prob": f"{p:.6f}"})
    print(f"[Done] Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
