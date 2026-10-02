# ARCH300 — what the source-set direction contains (decomposition)

Exploratory analysis (ANALYSIS_PLAN.md, section 9). 600 photographs: AESTHETIC 89, CONTROL 89, neither 422. Covariates: 10 photographic measures, 11 framing/scene and 10 architectural-content attributes (CLIP ViT-L/14 zero-shot), 36 catalogue-metadata indicators. Seed 20261007; 5000 bootstrap samples. "Accounted for" is linear and correlational.

## 1. How much of the direction each block describes

Position of each photograph along the direction (relative source-set affinity) predicted from the covariates; cross-validated R² over the 600 photographs (0 = nothing, 1 = everything).

| covariates | R² (95% CI) | unique part |
|---|---|---|
| photography | **0.10** (0.05 / 0.15) | +0.01 |
| framing and scene | **0.21** (0.14 / 0.27) | +0.02 |
| architectural content | **0.50** (0.44 / 0.55) | +0.19 |
| catalogue metadata | **0.25** (0.19 / 0.31) | +0.04 |
| photography + framing and scene | **0.27** (0.21 / 0.33) | – |
| image-derived (three blocks) | **0.55** (0.49 / 0.59) | – |
| all four blocks | **0.58** (0.53 / 0.63) | – |

CLIP attribute "elegant building vs ugly building" alone (a judgement of beauty, kept out of the blocks): R² -0.00 (-0.02 / 0.00).

## 2. How much of the AESTHETIC − CONTROL separation each block accounts for

The covariate model is fitted on the 422 photographs of neither set and applied to the two sets. Observed gap along the direction: +0.0363. Accounted share = gap predicted from the covariates / observed gap.

| covariates | share of the gap accounted for (95% CI) |
|---|---|
| photography | **8%** (-4% / 19%) |
| framing and scene | **33%** (19% / 50%) |
| architectural content | **57%** (43% / 76%) |
| catalogue metadata | **21%** (5% / 36%) |
| photography + framing and scene | **35%** (21% / 51%) |
| image-derived (three blocks) | **62%** (47% / 82%) |
| all four blocks | **65%** (50% / 85%) |
| CLIP 'elegant vs ugly' alone (not in the blocks) | **2%** (1% / 4%) |

**Reading fixed in advance (9.4):** mixed: the measured variables account for part of the separation, neither most of it nor almost none (photography + framing and scene 35%; all four blocks 65%). A small share only says that these variables do not capture the direction, not that the remainder is an architectural structure.

## 3. The direction after removing what the image-derived covariates predict

The part of every embedding that is linearly predictable from the 31 image-derived covariates (model fitted on the 422 photographs of neither set; it predicts 9% of the variance of the embeddings out of sample) is removed from photographs and generated images, and the direction is computed again.

| quantity | original direction | residual direction |
|---|---|---|
| distance between the AESTHETIC and CONTROL centroids | 0.235 | 0.166 (71% of the original; cosine with the original +0.85) |
| follows the mean rating of the 422 photographs of neither set (Spearman ρ) | +0.39 | +0.24 |

Generated images, AESTHETIC − CONTROL at equal training seed (192 cells, bootstrap over prompts):

| training seed | along the original direction | of which carried by the covariate-predictable part | along the residual direction |
|---|---|---|---|
| A = 1254 | +0.0435 (+0.0278 / +0.0609); 18% of the gap | +0.0166 (+0.0043 / +0.0299); 38% of the contrast | **+0.0186** (+0.0061 / +0.0325); 11% of the residual gap |
| B = 9865 | +0.0412 (+0.0254 / +0.0584); 18% of the gap | +0.0387 (+0.0262 / +0.0517); 94% of the contrast | **+0.0080** (-0.0037 / +0.0194); 5% of the residual gap |
| C = 42160 | +0.0504 (+0.0353 / +0.0674); 21% of the gap | +0.0264 (+0.0123 / +0.0404); 52% of the contrast | **+0.0269** (+0.0161 / +0.0382); 16% of the residual gap |

## 4. The same variables in the generated images

For each covariate: AESTHETIC − CONTROL between the two sets of photographs, and between the generated images at equal training seed (mean of the three seeds; 95% CI over prompts), in standard deviations of the 600 photographs. Sorted by the size of the difference between the photograph sets.

| covariate | block | photographs | generated (95% CI) | same sign for the three seeds | same sign as the photographs |
|---|---|---|---|---|---|
| curved_organic | architectural content | +0.61 | +0.10 (+0.00 / +0.21) | yes | yes |
| concrete | architectural content | -0.58 | -0.10 (-0.17 / -0.04) | no | yes |
| iconic_design | architectural content | +0.58 | +0.19 (+0.10 / +0.29) | yes | yes |
| interior | framing and scene | +0.45 | +0.09 (+0.01 / +0.17) | yes | yes |
| elegant | judgement (not in the blocks) | +0.45 | +0.17 (+0.09 / +0.26) | yes | yes |
| contemporary | architectural content | +0.43 | +0.29 (+0.21 / +0.37) | yes | yes |
| monumental | architectural content | +0.42 | +0.11 (+0.05 / +0.18) | yes | yes |
| cars | framing and scene | -0.41 | -0.22 (-0.32 / -0.13) | yes | yes |
| low_angle | framing and scene | +0.41 | +0.20 (+0.10 / +0.30) | yes | yes |
| contrast | photography | -0.37 | -0.15 (-0.21 / -0.09) | yes | yes |
| glass | architectural content | +0.37 | +0.09 (+0.01 / +0.17) | yes | yes |
| complex_form | architectural content | +0.35 | +0.17 (+0.07 / +0.27) | yes | yes |
| urban | framing and scene | -0.26 | -0.09 (-0.17 / -0.00) | yes | yes |
| vintage_photo | framing and scene | -0.26 | -0.20 (-0.31 / -0.10) | yes | yes |
| saturation | photography | +0.22 | +0.28 (+0.18 / +0.39) | yes | yes |
| sky_brightness | photography | -0.19 | -0.21 (-0.27 / -0.14) | no | yes |
| dark_share | photography | -0.13 | +0.10 (-0.01 / +0.22) | no | no |
| people | framing and scene | -0.13 | -0.16 (-0.27 / -0.07) | no | yes |
| colourful | architectural content | +0.11 | +0.01 (-0.06 / +0.07) | no | yes |
| colorfulness | photography | +0.11 | +0.12 (+0.08 / +0.17) | yes | yes |
| greenery | framing and scene | +0.10 | -0.02 (-0.10 / +0.05) | no | no |
| sharpness | photography | +0.08 | +0.11 (-0.04 / +0.28) | no | yes |
| real_photo | framing and scene | +0.07 | +0.18 (+0.09 / +0.26) | yes | yes |
| water | framing and scene | +0.06 | +0.01 (-0.07 / +0.09) | no | yes |
| warmth | photography | +0.05 | -0.19 (-0.28 / -0.09) | no | no |
| white | architectural content | -0.05 | -0.08 (-0.16 / -0.00) | no | yes |
| brightness | photography | +0.03 | -0.23 (-0.35 / -0.11) | no | no |
| bright_share | photography | -0.03 | -0.06 (-0.10 / -0.03) | yes | yes |
| detail | photography | +0.03 | +0.13 (+0.01 / +0.25) | no | yes |
| whole_building | framing and scene | -0.02 | -0.14 (-0.23 / -0.06) | no | yes |
| wood | architectural content | -0.01 | -0.13 (-0.19 / -0.05) | yes | yes |
| sunny | framing and scene | +0.00 | +0.10 (+0.05 / +0.16) | yes | yes |

Across the 31 image-derived covariates, the profile of the generated images follows that of the photographs with Pearson r **+0.71** (+0.56 / +0.82; Spearman ρ +0.71); same sign for 27 of 31. Mean size of the differences: photographs 0.22, generated images 0.14 standard deviations.

