from __future__ import annotations
import csv, json, sys
from pathlib import Path
from typing import Any, Dict, List
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# Proje kök dizinini ekle (2 seviye yukarı)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from stage1_idiom_signal.model_registry import MODEL_REGISTRY

# --- CONFIG ---
MODELS_TO_RUN = ["fusion_mlp", "marked_mlp", "baseline"]
BATCH_SIZE = 16
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_NAME = "xlm-roberta-base"

# Paths
INPUT_TRAIN_DIR = REPO_ROOT / "admire2_training"
INPUT_LLM_DIR = REPO_ROOT / "llm_results_training/enriched"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

class LLMLoader:
    def __init__(self, data_dir: Path):
        self.data: Dict[str, Dict[str, Any]] = {}
        if not data_dir.exists(): return
        for path in data_dir.glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    sent = rec.get("sentence", "").strip()
                    if sent: self.data[sent] = rec
    def get_rewrites(self, sentence: str) -> Dict[str, str]:
        entry = self.data.get(sentence.strip())
        if not entry: return {k: sentence for k in ["lex_sub", "comp_drop", "paraphrase", "literalize"]}
        return {k: entry.get(k, {}).get("text", sentence) for k in ["lex_sub", "comp_drop", "paraphrase", "literalize"]}

def mean_pool(hidden_states, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden_states * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = LLMLoader(INPUT_LLM_DIR)
    
    # 1. Prepare Data (EN/PT Nested Splits)
    all_rows = []
    for lang in ["EN", "PT"]:
        base = INPUT_TRAIN_DIR / lang
        if not base.exists(): continue
        for split in ["train", "dev", "test", "xeval"]:
            split_path = base / split
            if not split_path.exists(): continue
            for path in split_path.glob("*.tsv"):
                print(f"📖 Reading {lang}/{split}/{path.name}...")
                with path.open("r", encoding="utf-8") as f:
                    for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                        all_rows.append({
                            "id": f"{lang}_{split}_{i}", 
                            "lang": lang, 
                            "sent": row["sentence"].strip(), 
                            "comp": row["compound"]
                        })

    if not all_rows:
        print("❌ No training data found. Check paths!"); return

    # 2. Setup XLM-R
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    encoder = AutoModel.from_pretrained(TRANSFORMER_NAME).to(DEVICE)
    encoder.eval()

    # 3. Run Models
    for model_name in MODELS_TO_RUN:
        engine = MODEL_REGISTRY.get(model_name)
        weights_path = REPO_ROOT / f"stage1_idiom_signal/{model_name}/outputs/idiom_signal.pt"
        
        if not engine or not weights_path.exists():
            print(f"⚠️ [Skip] {model_name} (Missing engine or weights)"); continue
        
        print(f"\n🚀 Running Training Data Inference: {model_name}")
        payload = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        classifier = engine.build_classifier(payload["input_dim"], payload["hyperparameters"]).to(DEVICE)
        classifier.load_state_dict(payload["classifier_state_dict"])
        classifier.eval()

        probs = []
        for i in tqdm(range(0, len(all_rows), BATCH_SIZE), desc=model_name):
            batch = all_rows[i : i + BATCH_SIZE]
            batch_texts = []
            for b in batch:
                batch_texts.extend(engine.prepare_texts(b["lang"], b["sent"], b["comp"], loader))
            
            num_inputs = len(batch_texts) // len(batch)
            with torch.no_grad():
                enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
                hidden = encoder(**enc).last_hidden_state
                pooled = mean_pool(hidden, enc["attention_mask"]).view(len(batch), num_inputs, -1)
                feat = engine.get_features(pooled)
                res = torch.sigmoid(classifier(feat).squeeze(-1)).cpu().tolist()
                probs.extend(res if isinstance(res, list) else [res])

        # 4. Save
        out_path = OUTPUT_DIR / f"m_scores_{model_name}.tsv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "m_prob"], delimiter="\t")
            writer.writeheader()
            for b, p in zip(all_rows, probs):
                writer.writerow({"sample_id": b["id"], "m_prob": f"{p:.6f}"})
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
