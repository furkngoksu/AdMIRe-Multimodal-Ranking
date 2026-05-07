# Q Construction Notes

## Main Idea

XLM-R does not produce a single sentence vector directly. Instead, it reads the
whole sentence and returns one contextual vector for each token.

Example:

```text
It     -> v1
took   -> v2
a      -> v3
lot    -> v4
of     -> v5
elbow  -> v6
grease -> v7
to     -> v8
...
```

These are contextual token vectors. This means the vector for `elbow` is not
produced independently from the sentence. XLM-R sees the full sentence context
while producing each token vector.

For the sentence:

```text
It took a lot of elbow grease to get the old engine running again.
```

the token vectors for `elbow` and `grease` are influenced by the surrounding
context and can reflect the idiomatic meaning of effort or work.

## Representation Construction

The query representation is built from two parts.

### 1. Sentence-Level Representation

We compute a sentence-level vector by averaging the contextual embeddings of the
real sentence tokens:

```text
Q_sentence = average(v1, v2, ..., vn)
```

This captures the general meaning of the sentence.

### 2. Target Compound Representation

We locate the target compound in the sentence using character offsets and the
tokenizer offset mappings.

For:

```text
compound = "elbow grease"
```

we identify the token vectors corresponding to the compound:

```text
elbow  -> v6
grease -> v7
```

Then we average these vectors:

```text
Q_target = average(v6, v7)
```

This captures the contextual representation of the target compound.

## Final Q Representation

The final raw query vector combines the sentence-level representation and the
target-compound representation:

```text
Q_raw = [Q_sentence ; Q_target]
```

Then we project it into the shared embedding space:

```text
Q = Linear(Q_raw)
```

Since XLM-R base has hidden size 768:

| Component | Dimension |
| --- | ---: |
| `Q_sentence` | 768 |
| `Q_target` | 768 |
| `Q_raw` | 1536 |
| `Q` | 512 |

Finally, `Q` is L2-normalized:

```text
Q = normalize(Q)
```

This prepares `Q` for cosine-similarity matching with candidate representations.

## Final Formula

```text
Q = normalize(W_q [Q_sentence ; Q_target])
```

where:

```text
Q_sentence = mean_pool(XLM-R(sentence))
Q_target   = mean_pool(XLM-R(sentence) tokens corresponding to compound)
```

## Design Decisions

| Decision | Choice |
| --- | --- |
| Text encoder | XLM-R |
| Input sentence | Original sentence |
| Target marker tokens | Not used |
| Target localization | Character span + tokenizer offset mapping |
| Sentence representation | Mean pooling over real sentence tokens |
| Target representation | Mean pooling over target compound tokens |
| Combination | Concatenation |
| Projection | 1536 -> 512 |
| Normalization | L2 normalization |
| Output | `Q` vector |

## Why This Design?

This design is used because:

- The sentence is kept in its natural form.
- No new special tokens are added to XLM-R.
- There is no randomly initialized marker-token embedding.
- The target compound is still explicitly represented through target-span
  pooling.
- The final `Q` vector contains both the general sentence meaning and the target
  compound meaning in context.

In short, `Q` represents:

```text
general sentence meaning + target compound meaning
```
