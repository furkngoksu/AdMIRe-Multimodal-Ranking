# Caption Encoder (T Block) Design Note

## Goal

The T block converts each candidate caption into a caption representation.

Each example contains five candidate captions:

```text
caption_1, caption_2, caption_3, caption_4, caption_5
```

The T block produces:

```text
T_1, T_2, T_3, T_4, T_5
```

where each `T_i` represents the meaning of candidate caption `i`.

## Position in the Pipeline

The T block belongs to the candidate side:

```text
Caption_i -> Text Encoder -> T_i
Image_i   -> Vision Encoder -> I_i

I_i + T_i -> C_i
```

It is independent from the idiomaticity signal `M`. The idiomaticity signal
modifies the query side:

```text
Q + M -> Q'
```

The final ranking compares:

```text
Q' with C_i
```

## Representation

The caption is encoded using a multilingual CLIP-compatible text encoder. This
encoder already returns a 512-dimensional representation aligned with the CLIP
ViT-B/32 image space, so no extra trainable projection is used. The output is
L2-normalized.

Formula:

```text
T_i = normalize(TextEncoder(caption_i))
```

## Encoder Choice

The current T block uses:

```text
sentence-transformers/clip-ViT-B-32-multilingual-v1
```

This model is a multilingual text encoder aligned to the original CLIP
ViT-B/32 image embedding space. Therefore it is a reasonable first choice if the
image branch uses the matching CLIP ViT-B/32 image encoder.

Important distinction:

```text
This is CLIP-compatible.
It is not a SigLIP text encoder.
```

If the image branch later uses a SigLIP-based image encoder, the T block should
be revisited so that the caption and image encoders come from the same
vision-language family.

For the first implementation pass, the current model is kept because it is
simple, multilingual, CLIP-aligned, and easy to run through `sentence-transformers`.

## Shape

For a batch with `B` examples and 5 candidates per example:

```text
Input:  captions = [B, 5]
Output: T        = [B, 5, 512]
```

## Design Decisions

| Decision | Choice |
| --- | --- |
| Input | Candidate captions |
| Encoder | Multilingual CLIP-compatible text encoder |
| Default model | `sentence-transformers/clip-ViT-B-32-multilingual-v1` |
| Encoder freeze | Yes |
| Projection | No |
| Normalize | Yes |
| Output shape | `[B, 5, 512]` |

## Why This Design?

Candidate representations are built from both images and captions. Since the
image branch will use a CLIP/SigLIP-style vision-language encoder, the caption
branch should also use a compatible text encoder.

The caption encoder already maps captions into the CLIP-aligned 512-dimensional
candidate space. Adding a trainable 512 -> 512 projection would introduce extra
parameters and can disturb the pretrained image-text alignment, so the T block
keeps the encoder output directly.
