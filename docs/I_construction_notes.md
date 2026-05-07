# Image Encoder (I Block) Design Note

## Goal

The I block converts each candidate image into an image representation.

Each example contains five candidate images:

```text
image_1, image_2, image_3, image_4, image_5
```

The I block produces:

```text
I_1, I_2, I_3, I_4, I_5
```

where each `I_i` represents the visual content of candidate image `i`.

## Position in the Pipeline

The I block belongs to the candidate side:

```text
Image_i   -> Vision Encoder -> I_i
Caption_i -> Text Encoder   -> T_i

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

## Encoder Choice

The T block uses:

```text
sentence-transformers/clip-ViT-B-32-multilingual-v1
```

This multilingual text encoder is aligned with image vectors from:

```text
sentence-transformers/clip-ViT-B-32
```

Therefore, the I block uses:

```text
sentence-transformers/clip-ViT-B-32
```

This keeps the candidate-side image and caption representations compatible.

## Representation

The image is encoded using the CLIP ViT-B/32 image encoder. The raw image
embedding is projected into the shared 512-dimensional space and L2-normalized.

Formula:

```text
I_i = normalize(W_I * ImageEncoder(image_i))
```

## Shape

For a batch with `B` examples and 5 candidates per example:

```text
Input:  images = [B, 5]
Output: I      = [B, 5, 512]
```

## Design Decisions

| Decision | Choice |
| --- | --- |
| Input | Candidate images |
| Encoder | `sentence-transformers/clip-ViT-B-32` |
| Encoder freeze | Yes |
| Projection | raw_dim -> 512 |
| Identity initialization | Yes, if raw_dim = 512 |
| Normalize | Yes |
| Output shape | `[B, 5, 512]` |

## Why This Design?

The candidate representation is built from both images and captions. Since the
T block uses a multilingual CLIP-compatible text encoder aligned with CLIP
ViT-B/32 image vectors, the I block uses the corresponding CLIP ViT-B/32 image
encoder.

The encoder is frozen for stability, while the projection layer remains
trainable. If the raw image embedding dimension is already 512, the projection
layer starts as identity so the pretrained CLIP image space is preserved at the
beginning of training.
