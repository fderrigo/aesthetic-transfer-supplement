# ARCH300 — replication across training seeds (DINOv2 embeddings)
Confirmatory analysis (ANALYSIS_PLAN.md, section 8). Model `facebook/dinov2-base`; frozen direction `baseline/direction_dinov2.npz` (datasets AESTHETIC-v2 / CONTROL-v2). Seed 20261005; 5000 bootstrap samples over prompts. 192 prompt × generation-seed cells (48 prompts), 159 of them shown to the participants in the original experiment.

## 1. Primary: AESTHETIC − CONTROL at equal training seed

Projection on the frozen direction (CLS token; the two training sets are 0.235 apart along it). Baseline value, unpaired, 159 cells: +0.0429.
| training seed | AESTHETIC LoRA | CONTROL LoRA | D (95% CI), all cells | transfer ratio | cells with D > 0 | D (95% CI), cells shown to the participants |
|---|---|---|---|---|---|---|
| A = 1254 | RUN-AESTHETIC-4 | REP-CTL-S1254 | **+0.0435 (+0.0272 / +0.0612)** | 18% | 126 of 192 | +0.0456 (+0.0272 / +0.0649) |
| B = 9865 | REP-AES-S9865 | RUN-CONTROL-4 | **+0.0412 (+0.0255 / +0.0578)** | 18% | 137 of 192 | +0.0431 (+0.0263 / +0.0605) |
| C = 42160 | REP-AES-S42160 | REP-CTL-S42160 | **+0.0504 (+0.0354 / +0.0674)** | 21% | 147 of 192 | +0.0494 (+0.0340 / +0.0671) |

**Criterion fixed in advance** (the three contrasts positive, each interval above zero): **MET**. Mean of the three +0.0451, range +0.0412 to +0.0504, standard deviation across training seeds 0.0048. With three training seeds no test between seeds is possible and none is reported.
Original unpaired contrast on the same 192 cells (AESTHETIC seed A − CONTROL seed B): +0.0415 (+0.0278 / +0.0570).

## 2. Corpus effect against training-seed effect

| same corpus, different training seed | difference along the direction (95% CI) |
|---|---|
| AESTHETIC: A − B | +0.0003 (-0.0100 / +0.0110) |
| AESTHETIC: A − C | +0.0096 (-0.0007 / +0.0213) |
| AESTHETIC: B − C | +0.0093 (-0.0028 / +0.0226) |
| CONTROL: A − B | -0.0020 (-0.0136 / +0.0093) |
| CONTROL: A − C | +0.0165 (+0.0053 / +0.0279) |
| CONTROL: B − C | +0.0185 (+0.0062 / +0.0320) |

R = mean |corpus contrast| / mean |between-seed difference| = **4.8** (well above 1 = along the direction the corpus moves the images more than the training seed does).
| per cell, mean over the comparisons | only the corpus changes | only the training seed changes |
|---|---|---|
| |difference of the projection on the direction| | 0.0679 | 0.0523 |
| distance between the two images in embedding space (any direction) | 0.680 | 0.629 |


## 3. Variance components (descriptive)

| source | standard deviation of the means along the direction |
|---|---|
| corpus | 0.0319 |
| training seed | 0.0078 |
| corpus x seed | 0.0021 |
| prompt | 0.2064 |
| generation seed within prompt | 0.0498 |
| residual | 0.0539 |

With three training seeds the components that involve the training seed are imprecise; they are shown to give the order of magnitude, not as estimates.

## 4. Sensitivities frozen in advance

| training seed | amount ‖Δ_A‖ − ‖Δ_C‖ (95% CI) | direction, cosine difference (95% CI) | specificity (95% CI) |
|---|---|---|---|
| A = 1254 | -0.008 (-0.035 / +0.019) | +0.058 (+0.037 / +0.079) | +0.0568 (+0.0350 / +0.0807) |
| B = 9865 | -0.019 (-0.046 / +0.006) | +0.056 (+0.036 / +0.077) | +0.0545 (+0.0330 / +0.0771) |
| C = 42160 | +0.028 (+0.009 / +0.047) | +0.068 (+0.048 / +0.089) | +0.0724 (+0.0520 / +0.0955) |


## 5. Determinism control

`REP-AES-S1254-R` repeats `RUN-AESTHETIC-4` exactly (same corpus, seed, parameters). Per cell, between the image of the original LoRA and the image of the repeated one: embedding distance 0.321 (0.297 / 0.346); |projection difference| 0.0219; signed difference -0.0021 (-0.0074 / +0.0031); identical embedding in 0% of the cells. For comparison: distance when only the training seed changes 0.629, when only the corpus changes 0.680.

## 6. Computational proxy (not human evidence)

Rating predictor trained on the phase-1 photographs only, applied to the generated images. It is a proxy computed by a model: it does not say what people would prefer, and no participant saw the new images.
| training seed | predicted rating, AESTHETIC − CONTROL (95% CI) |
|---|---|
| A = 1254 | +0.117 (+0.076 / +0.157) |
| B = 9865 | +0.113 (+0.070 / +0.161) |
| C = 42160 | +0.101 (+0.062 / +0.145) |


## Sensitivity — mean of the patch tokens instead of the CLS token

| quantity | CLS token | mean of patch tokens |
|---|---|---|
| D, training seed A | +0.0435 (+0.0272 / +0.0612) | +0.0347 (+0.0222 / +0.0480) |
| D, training seed B | +0.0412 (+0.0255 / +0.0578) | +0.0407 (+0.0264 / +0.0566) |
| D, training seed C | +0.0504 (+0.0354 / +0.0674) | +0.0505 (+0.0359 / +0.0657) |
| criterion | met | met |
| R (corpus / training seed) | 4.8 | 4.0 |

