# Stage 1 — Idiom Signal Models (M)

This module implements six distinct binary classification models to detect idiomaticity in multilingual contexts. These models generate the continuous scalar Idiom Signal ($M$) mapped to downstream multimodal ranking architectures.

## Model Philosophy & Feature Strategies

To effectively separate literal from idiomatic multi-word expression (MWE) usages, the feature extraction framework implements five distinct representation strategies:

1. **Contextual Awareness**: Extracting global semantic context from surrounding sentence sequences.
2. **Language Signaling (`lang`)**: Explicitly prefixing inputs with language indicators (e.g., `[LANG=tr]`) to condition cross-lingual representations.
3. **Structural Marking (`marked`)**: Enclosing candidate compounds within explicit structural boundaries (`<mwe>` and `</mwe>`).
4. **Target Extraction (`target`)**: Deriving targeted representation spans dynamically via token coordinate mapping (Span Pooling) without injecting structural boundary tokens.
5. **Semantic Break (`perturbation`)**: Computing explicit representation gaps ($V_{diffs}$) between the input sequence and LLM-generated literal alternatives.

## Model Architectures

The module compares six structural variants implementing subsets of the representation strategies:

| Model | Input Features | Pooling Mechanism | Rationale |
|---|---|---|---|
| **`baseline_mlp`** | Raw Context | Mean Pooling | Captures standard contextual embeddings without explicit target boundary localization. |
| **`lang_marked_simple`** | Lang + Marked | Mean Pooling | Combines explicit structural tags with a linear projection head. |
| **`lang_marked_mlp`** | Lang + Marked | Mean Pooling | Integrates structural tags using a deeper multi-layer perceptron (MLP) mapping. |
| **`lang_marked_perturbation_mlp`** | Lang + Marked + Diffs | Mean Pooling | Fuses explicit structural tags with semantic perturbation distance vectors. |
| **`perturbation_mlp`** | Semantic Diffs | Mean Pooling | Evaluates idiomaticity directly through the vector distance of semantic alternatives. |
| **`target_perturbation_mlp`** | Target + Diffs | Span Pooling + Mean Pooling | Combines uncorrupted target token representations with multi-aspect semantic gap vectors. |

## Feature Engineering & Vector Representation

Feature integration relies on constructing unified candidate vectors ($V$) prior to final classification layers:

- **Tagged Sequence Formulation**: `[LANG=tr] <mwe>çürük elma</mwe> [MWE] çürük elma`
- **Span Pooling Extraction**: Averages latent token states directly corresponding to character offsets of the target idiom.
- **Perturbation Difference Calculation**: Vector distances computed against literal semantic substitutions:
  $$V_{diff} = |V_{target} - V_{literal\_rewrite}|$$
- **Unified Input Tensor**: The complete input vector maps target span embeddings alongside four distinct semantic alternatives:
  $$V_{input} = [V_{target\_span} \,;\, V_{diff\_lexsub} \,;\, V_{diff\_compdrop} \,;\, V_{diff\_paraphrase} \,;\, V_{diff\_literal}]$$
