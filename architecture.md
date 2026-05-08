Aşağıdaki gibi temiz çıkarılır. Bence en iyi format: **1) kısa mimari özeti + 2) Mermaid flowchart**. Çünkü diagram çok dallı; sadece sözle yazarsan kaybolur.

## Kısa sözlü extraction

Model iki aşamalı:

**Stage 1 — Idiomaticity Module / M modülü ayrı eğitiliyor**

Input:

```text
Sentence + Compound
```

M modülü, compound’un literal mı idiomatic mi olduğunu anlamaya çalışıyor.

İçeride şu sinyaller var:

```text
Controlled Semantic Perturbations
├── Lexical Substitution        → Bsub
├── Component Dependence        → Bcomp
├── Literalization              → Blit
└── Literal Semantic Compatibility → Slsc
```

Bu skorlar/sinyaller birleştiriliyor:

```text
Bsub, Bcomp, Blit, Slsc
→ Weighted Combiner / Lightweight Classifier
→ M
```

Stage 1 sonunda:

```text
M freeze edilir.
```

---

**Stage 2 — Ana ranking modeli joint train ediliyor**

Stage 2’de frozen olanlar:

```text
XLM-R encoder
Vision encoder
Caption/text encoder
M module
```

Trainable olanlar:

```text
query head / projection
query fusion
candidate fusion gate
projection layers
```

Query tarafı:

```text
Sentence + Compound
→ target-aware sentence formatting
→ XLM-R frozen encoder
→ sentence-level representation
→ target-focused representation
→ concat/combine
→ projection
→ Q
```

Sonra Q ile frozen M birleştiriliyor:

```text
Q prime = Q + alpha * Proj(M)
```

Candidate tarafı her aday için çalışıyor:

```text
Image_i   → frozen vision encoder → L_i
Caption_i → frozen text encoder   → T_i
```

Image ve caption embedding gate ile fuse ediliyor:

```text
g_i = sigmoid(L_i and T_i)
C_i = g_i * L_i + (1 - g_i) * T_i
```

Son scoring:

```text
S_i = cosine(Q prime, C_i)
```

5 aday score’lanıyor:

```text
Scores S1...S5
→ rank candidates
→ prediction / top-1 / nDCG
```

Loss:

```text
ranking loss over 5 candidates
```

---

## Mermaid flowchart

```mermaid
flowchart LR

%% =========================
%% Input
%% =========================

A[Sentence + Compound]

%% =========================
%% Query Side
%% =========================

subgraph QuerySide["Query Side"]
    A --> QF[Target-aware sentence formatting<br/>e.g. mark or highlight compound]
    QF --> XLMR[XLM-R Encoder<br/>frozen backbone]

    XLMR --> SR[Sentence-level Representation]
    XLMR --> TR[Target-focused Representation]

    SR --> COMB[Concat / Combine]
    TR --> COMB

    COMB --> QPROJ[Trainable query head / projection<br/>Projection to shared dim]
    QPROJ --> Q[Q]
end

%% =========================
%% Stage 1 - Idiomaticity Module
%% =========================

subgraph Stage1["Stage 1 — Train M module separately"]
    A --> CSP[Controlled Semantic Perturbations]

    CSP --> LS[Lexical Substitution]
    CSP --> CD[Component Dependence]
    CSP --> LIT[Literalization]
    CSP --> LSC[Literal Semantic Compatibility]

    LS --> BSUB[Bsub]
    CD --> BCOMP[Bcomp]
    LIT --> BLIT[Blit]
    LSC --> SLSC[Slsc]

    BSUB --> WCOMB[Weighted Combiner / Lightweight Classifier]
    BCOMP --> WCOMB
    BLIT --> WCOMB
    SLSC --> WCOMB

    WCOMB --> M[M]
end

M -. after Stage 1 freeze M .-> MFROZEN[M module frozen in Stage 2]

%% =========================
%% Query Fusion
%% =========================

Q --> QFUSION[Query Fusion<br/>Q prime = Q + alpha * Proj M]
MFROZEN --> QFUSION
QFUSION --> QP[Q prime]

%% =========================
%% Candidate Side
%% =========================

subgraph CandidateSide["Candidate Side — repeated for each candidate i"]
    IMG[Image_i] --> VENC[Vision Encoder<br/>CLIP / SigLIP<br/>frozen backbone]
    CAP[Caption_i] --> TENC[Text Encoder<br/>CLIP / SigLIP Text<br/>frozen backbone]

    VENC --> LI[L_i]
    TENC --> TI[T_i]

    LI --> GATE[Trainable candidate fusion gate<br/>g_i = sigmoid(L_i and T_i)]
    TI --> GATE

    LI --> CF[Fused Candidate<br/>C_i = g_i * L_i + 1 - g_i * T_i]
    TI --> CF
    GATE --> CF
end

%% =========================
%% Scoring and Ranking
%% =========================

QP --> SCORE[Similarity Scoring<br/>S_i = cosine(Q prime, C_i)]
CF --> SCORE

SCORE --> SCORES[Scores S1...S5]
SCORES --> RANK[Rank Candidates]
RANK --> OUT[Prediction / Top-1 / nDCG]

OUT -.-> LOSS[Ranking loss<br/>over 5 candidates]
LOSS -. trainable feedback .-> QFUSION
LOSS -. trainable feedback .-> GATE
LOSS -. trainable feedback .-> QPROJ
```

## Daha sade “mental model”

Bu sistem aslında şunu yapıyor:

```text
1. Compound idiom mu literal mı? 
   → M modülü bunu öğreniyor.

2. Sentence + compound için query embedding çıkarılıyor.
   → Q

3. M bilgisi query’ye ekleniyor.
   → Q prime

4. Her aday image + caption olarak temsil ediliyor.
   → C_i

5. Q prime ile her C_i cosine similarity ile karşılaştırılıyor.

6. En yüksek skorlu aday seçiliyor.
```

En basit haliyle:

```text
Sentence + Compound
        ↓
Query embedding + Idiomaticity signal
        ↓
Q prime
        ↓
Compare against 5 image-caption candidates
        ↓
Rank / Pick best candidate
```

## Freeze / trainable özeti

```text
Frozen in Stage 2:
- XLM-R encoder
- Vision encoder
- Caption/text encoder
- M module

Trainable in Stage 2:
- Query projection/head
- Query fusion: Q + alpha * Proj(M)
- Candidate fusion gate
- Projection layers
```

## Stage 1 supervised M baseline

Stage 1 trains only the idiomaticity signal module M. The input CSV must have:

```text
Language,MWE,Target,Label
```

`Label` is used only as the supervised training target:

```text
0 = literal / non-idiomatic
1 = idiomatic
```

The trained model output used by Stage 2 is not a hard class. It is:

```python
M = sigmoid(logit)
```

so every example receives a continuous idiomaticity score in `[0, 1]`.

Train Stage 1 from the repository root:

```bash
python -m training.train_idiom_signal --csv_path unique_mwe_target_label_language.csv --output_dir outputs/idiom_signal
```

Useful training options:

```bash
python -m training.train_idiom_signal \
  --csv_path unique_mwe_target_label_language.csv \
  --output_dir outputs/idiom_signal \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-4 \
  --max_length 128
```

By default the XLM-R encoder is frozen and only the classifier head is trained.
To fine-tune the encoder too, add:

```bash
--unfreeze-encoder
```

Saved artifacts:

```text
outputs/idiom_signal/idiom_signal.pt
outputs/idiom_signal/config.json
outputs/idiom_signal/tokenizer/
outputs/idiom_signal/metrics.json
```

Load the frozen Stage 1 model and compute M:

```python
from src.idiom_signal import compute_m, load_idiom_signal

model = load_idiom_signal("outputs/idiom_signal")
score = compute_m(
    sentence="He spilled the beans about the plan.",
    mwe="spilled the beans",
    language="EN",
    model=model,
)
print(score)
```

Run the smoke demo after training:

```bash
python scripts/demo_idiom_signal.py --model_dir outputs/idiom_signal
```

Stage 2 should load this saved M model, keep it frozen, and inject the continuous
score into the query representation. Stage 2 ranking is intentionally not
implemented in this Stage 1 baseline.

Kritik fikir şu:
**Model sadece sentence-image matching yapmıyor. Compound’un idiomatic/literal doğasını ayrı bir sinyal olarak query’ye enjekte ediyor.** Bu sayede örneğin “spill the beans” gibi ifadelerde literal bean image’a gitmemesi, idiomatic anlamı yakalaması hedefleniyor.
