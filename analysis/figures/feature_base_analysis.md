# What is transferred: the 31 descriptors against BASE, and the shifts in embedding space

## What was reused

- Descriptors: `metrics/covariates_images.csv`, produced by `image_scripts/covariates.py` in one run for all images (photographs, BASE, CONTROL, AESTHETIC, replication adapters). The BASE scores were already there: nothing was recomputed.
- Pipeline: that of `direction_decomposition.py`, section 4 (the original figure): CLIP attributes as logits, every descriptor standardised on the mean and SD of the 600 photographs, "elegant" excluded, 31 descriptors.
- Sample: all 192 prompt × generation-seed cells (48 prompts), as in the original figure, with BASE (192 images), AESTHETIC and CONTROL as the mean of the three paired training seeds (A = 1254, B = 9865, C = 42160; 576 images per corpus).
- X: AESTHETIC − CONTROL between the two sets of photographs (89 + 89), identical in the three figures. Y: paired difference per cell, averaged over cells (and seeds).
- Intervals: bootstrap over prompts for each descriptor; bootstrap over the 31 descriptors for r; 5,000 samples. No significance test.

## Numbers

| figure | Y | Pearson r (95% CI) | same sign as the photographs |
|---|---|---|---|
| original | AESTHETIC − CONTROL | +0.71 (+0.57 / +0.82) | 27 of 31 |
| 1 | AESTHETIC − BASE | -0.01 (-0.31 / +0.27) | 14 of 31 |
| 2 | CONTROL − BASE | -0.43 (-0.69 / -0.13) | 13 of 31 |

Zeros in the signs: 0. Algebraic check (A − BASE) − (CONTROL − BASE) = A − CONTROL, descriptor by descriptor: maximum absolute error 5.00e-16 (BASE is shared by the two contrasts within each cell, so the identity holds up to floating-point rounding).

Mean absolute size of the differences (SD of the corpus): photographs 0.22; AESTHETIC − CONTROL 0.14; AESTHETIC − BASE 0.21; CONTROL − BASE 0.21.

## The four groups of descriptors

**AESTHETIC moves from BASE in the direction of the training difference, CONTROL does not (or moves the other way)** — 3 descriptors: monumental (photos +0.42; A−B +0.08; C−B -0.03), sky_brightness (photos -0.19; A−B -0.18; C−B +0.02), bright_share (photos -0.03; A−B -0.04; C−B +0.02)

**both adapters move from BASE in the direction of the training difference** — 11 descriptors: curved_organic (photos +0.61; A−B +0.22; C−B +0.12), concrete (photos -0.58; A−B -0.16; C−B -0.06), contrast (photos -0.37; A−B -0.25; C−B -0.10), vintage_photo (photos -0.26; A−B -0.24; C−B -0.04), saturation (photos +0.22; A−B +0.49; C−B +0.20), colourful (photos +0.11; A−B +0.37; C−B +0.36), colorfulness (photos +0.11; A−B +0.46; C−B +0.33), sharpness (photos +0.08; A−B +0.24; C−B +0.13), detail (photos +0.03; A−B +0.19; C−B +0.06), whole_building (photos -0.02; A−B -0.33; C−B -0.18), sunny (photos +0.00; A−B +0.38; C−B +0.28)

**CONTROL moves in the direction of the training difference, AESTHETIC does not** — 2 descriptors: dark_share (photos -0.13; A−B +0.03; C−B -0.07), brightness (photos +0.03; A−B -0.16; C−B +0.06)

**neither adapter moves in the direction of the training difference** — 15 descriptors: iconic_design (photos +0.58; A−B -0.11; C−B -0.30), interior (photos +0.45; A−B -0.35; C−B -0.44), contemporary (photos +0.43; A−B -0.04; C−B -0.33), cars (photos -0.41; A−B +0.17; C−B +0.40), low_angle (photos +0.41; A−B -0.16; C−B -0.36), glass (photos +0.37; A−B -0.32; C−B -0.41), complex_form (photos +0.35; A−B -0.32; C−B -0.49), urban (photos -0.26; A−B +0.09; C−B +0.18), people (photos -0.13; A−B +0.25; C−B +0.41), greenery (photos +0.10; A−B -0.03; C−B -0.01), real_photo (photos +0.07; A−B -0.14; C−B -0.32), water (photos +0.06; A−B -0.02; C−B -0.03), warmth (photos +0.05; A−B -0.58; C−B -0.39), white (photos -0.05; A−B +0.11; C−B +0.19), wood (photos -0.01; A−B +0.08; C−B +0.21)

Descriptors where the two adapters move from BASE in the **same** direction (whatever the training difference): 26 of 31; correlation between the two shifts across descriptors: +0.82.

## Embedding space (fig_embedding_shifts.png)

| adapter | along the source-set direction (→ AESTHETIC photographs) | along the shared direction of change |
|---|---|---|
| AESTHETIC seed A | +0.0416 | +0.1118 |
| CONTROL seed A | -0.0019 | +0.1202 |
| AESTHETIC seed B | +0.0413 | +0.1040 |
| CONTROL seed B | +0.0001 | +0.1229 |
| AESTHETIC seed C | +0.0320 | +0.1182 |
| CONTROL seed C | -0.0185 | +0.1048 |

The gap between the photograph centroids along the direction is 0.235. The vertical axis is the component of change that AESTHETIC and CONTROL share: it is large for both and says nothing about the corpus; the difference between the adapters is almost entirely horizontal.

## Reading

The descriptors are visual proxies (photographic measures and CLIP zero-shot attributes), not measures of architectural quality or beauty. The groups above separate what the AESTHETIC adapter does that CONTROL does not from what both fine-tunings do. Interpretation is left to the text of the paper.

