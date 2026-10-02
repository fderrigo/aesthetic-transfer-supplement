# ARCH300 — computational analysis of the images (DINOv2 embeddings)
Exploratory analysis (ANALYSIS_PLAN.md, section 6). Model `facebook/dinov2-base`; preprocessing: whole image, short side 224px, multiples of 14px, bicubic, ImageNet mean/std. Export: `expfinal`. Seed 20261001; 10000 permutations, 2000 bootstrap samples.
The embedding describes what the network sees in an image (content, composition, photographic rendering) without telling them apart: the results say whether and how much the sets differ for the model, not why.

## Part A — What distinguishes the photographs selected by the consensus

600 corpus photographs; AESTHETIC 89, CONTROL 89 (the sets the models were trained on). Embedding: DINOv2 (facebook/dinov2-base), CLS token, L2-normalised, dimension 768. Distances are Euclidean between unit vectors (two unrelated photographs are typically ≈ 1.30 apart).
**A1. Centroids and distances.**
| comparison | distance between centroids | reference | p |
|---|---|---|---|
| AESTHETIC – CONTROL | **0.235** (bootstrap 95% CI 0.229 / 0.316) | labels exchanged at random: 0.138 (95th pct 0.159) | < .001 |
| AESTHETIC – CONTROL, labels exchanged within building type | 0.235 | 0.158 (95th pct 0.185); 27 types contain both sets | < .001 |
| AESTHETIC – rest of the corpus | **0.240** | random subsets of 89 photographs: 0.106 (95th pct 0.120) | < .001 |
| CONTROL – photographs in neither set | 0.104 | descriptive | – |

Standardised distance AESTHETIC – CONTROL = distance / pooled dispersion = **0.26** (how far apart the two centroids are compared with how spread each set is; the two clouds overlap almost entirely when this is well below 1).
**A2. Dispersion** (mean distance of the photographs from the centroid of their set):
| set | photographs | dispersion |
|---|---|---|
| AESTHETIC | 89 | 0.906 |
| CONTROL | 89 | 0.921 |
| corpus | 600 | 0.921 |

AESTHETIC − CONTROL = -0.015 (bootstrap 95% CI -0.038 / +0.007; permutation p = .152). Negative = the selected photographs are more alike than the control ones.
**A3. Is the difference usable?**
| question | result | p (permutation) |
|---|---|---|
| Can the two sets be told apart from the embedding alone? (ridge classifier, 10-fold CV) | AUC **0.78** (0.5 = chance, 1 = perfect) | .005 |
| Do the embeddings predict the mean rating of a photograph? (ridge, nested 10-fold CV, 600 photographs) | r **0.64**, R² 0.41 | .005 |
| Does the direction CONTROL → AESTHETIC track the ratings of the 422 photographs in neither set? | Spearman ρ **+0.39** | < .001 |

The direction from the CONTROL centroid to the AESTHETIC centroid (length 0.235) is the "aesthetic direction" used in Part B. It is defined on the photographs only and never sees the generated images or the phase-2 votes.
**A4. Relative source-set affinity.** Definition: for an image, the mean cosine similarity with the AESTHETIC photographs minus the mean cosine similarity with the CONTROL photographs. Positive = the image resembles the selected set more than the control set. For a photograph that belongs to one of the two sets the photograph itself is left out of its own set (leave-one-out), otherwise it would resemble its set by construction. With unit vectors the affinity equals the position along the aesthetic direction multiplied by its length (0.235), so it is the same measure used in Part B.
| photographs | n | mean affinity | share with positive affinity |
|---|---|---|---|
| AESTHETIC | 89 | +0.0301 | 78% |
| CONTROL | 89 | -0.0061 | 36% |
| neither set | 422 | -0.0127 | 33% |

AESTHETIC − CONTROL = **+0.0363** (bootstrap 95% CI +0.0245 / +0.0475; labels exchanged, leave-one-out repeated each time: mean +0.0000, p < .001). The affinity alone tells the two sets apart with AUC **0.74** (leave-one-out) and follows the mean rating of the 600 photographs with Spearman ρ **+0.47** (p < .001). One value per photograph: `source_affinity.csv`.

## Part B — Shifts of the generated images

159 complete triplets (48 prompts) of active generated images. For each prompt/seed: Δ_A = AESTHETIC − BASE and Δ_C = CONTROL − BASE (embeddings). Intervals: bootstrap over prompts; p values: sign-flip permutation of the prompt means.
**B1. Is the AESTHETIC shift coherent with the one observed between the training sets?** Projection on the aesthetic direction (unit vector; the two training sets are 0.235 apart along it):
| shift | mean projection | 95% CI | p | mean cosine with the direction |
|---|---|---|---|---|
| BASE → AESTHETIC | +0.0443 | +0.0270 / +0.0635 | < .001 | +0.058 |
| BASE → CONTROL | +0.0014 | -0.0120 / +0.0155 | .537 | +0.002 |
| **AESTHETIC − CONTROL** | **+0.0429** | +0.0275 / +0.0599 | < .001 | +0.055 |

Embedding transfer ratio = (AESTHETIC − CONTROL shift) / (gap between the training sets) = **18%** (95% CI 12% / 25%). 98 of 159 triplets have a positive difference.
**B2. Does each model move towards its own training set?** Mean projection of the shift on the direction from the corpus centroid to each training set:
| shift | towards AESTHETIC set | towards CONTROL set |
|---|---|---|
| BASE → AESTHETIC | +0.0434 | -0.0179 |
| BASE → CONTROL | +0.0011 | -0.0010 |

Specificity (own set minus other set, AESTHETIC row minus CONTROL row) = +0.0592 (95% CI +0.0365 / +0.0816), p < .001. Size of the shifts: ‖Δ_A‖ = 0.678, ‖Δ_C‖ = 0.699; cosine between Δ_A and Δ_C = +0.51 (how much of the change is common to the two fine-tuned models, whatever the training photographs).
**B3. Alignment and human votes.**
| question | result | 95% CI | p | n |
|---|---|---|---|---|
| Triplets where AESTHETIC moves further along the direction than CONTROL: is AESTHETIC also rated higher? (Spearman) | ρ **+0.23** | +0.09 / +0.38 | .003 | 159 |
| … and chosen more often in the pairwise comparisons? (Spearman) | ρ **+0.04** | -0.15 / +0.20 | .648 | 159 |
| Within a triplet, is the image further along the direction rated higher? (Pearson) | r **+0.13** | +0.05 / +0.21 | – | 477 |

**B4. The rating predictor learned on the photographs, applied to the generated images.**
| question | result | p |
|---|---|---|
| Predicted rating vs human rating of the generated images (Spearman, all images) | ρ +0.62 | < .001 |
| Same, within triplet (Pearson) | r +0.31 | – |
| Predicted rating, AESTHETIC − CONTROL | +0.105 points (95% CI +0.066 / +0.145) | < .001 |

Human votes for comparison (same triplets, exclusions applied): AESTHETIC − CONTROL = +0.105 points; AESTHETIC chosen in 51.1% of the decisive pairwise comparisons (mean over triplets).
**B5. Representation transfer score, and amount separated from direction.** Definition: for a triplet (same prompt and seed), the relative source-set affinity (A4) of the AESTHETIC image minus that of the CONTROL image. The BASE image cancels out, so the score is the AESTHETIC − CONTROL difference of B1 multiplied by the length of the aesthetic direction; divided by the squared gap between the training sets it is the transfer ratio of that triplet. Mean score **+0.0101** (95% CI +0.0064 / +0.0143, p < .001); median +0.0053; 98 of 159 triplets positive. One row per triplet: `triplet_transfer.csv`.
A shift along the direction is the product of how far the image moved from BASE (amount, ‖Δ‖) and where it moved (direction, cosine with the aesthetic direction). The two are reported separately:
| quantity | BASE → AESTHETIC | BASE → CONTROL | difference | 95% CI | p |
|---|---|---|---|---|---|
| amount of shift ‖Δ‖ | 0.678 | 0.699 | -0.021 | -0.052 / +0.008 | .129 |
| direction (cosine with the aesthetic direction) | +0.058 | +0.002 | **+0.055** | +0.035 / +0.076 | < .001 |

Exact split of the AESTHETIC − CONTROL shift of B1 (the two parts add up to it):
| part | mean | 95% CI | p | share of the total |
|---|---|---|---|---|
| due to direction (the two models given the same amount of shift) | +0.0410 | +0.0267 / +0.0568 | < .001 | 96% |
| due to amount (the two models given the same direction) | +0.0019 | -0.0013 / +0.0053 | .464 | 4% |

Relation with the human votes, per triplet (Spearman, bootstrap over prompts):
| AESTHETIC − CONTROL difference in | vs rating difference | vs share of AESTHETIC choices |
|---|---|---|
| amount of shift | ρ +0.07 (-0.07 / +0.19), p .405 | ρ -0.15 (-0.29 / +0.01), p .063 |
| direction of shift | ρ +0.24 (+0.08 / +0.39), p .003 | ρ +0.04 (-0.13 / +0.22), p .621 |
| representation transfer score (B3) | ρ +0.23 (+0.09 / +0.38), p .003 | ρ +0.04 (-0.15 / +0.20), p .648 |

**B6. Sensitivity: all the generated images.** Primary analysis above: the 477 active images shown to the participants (159 triplets). Here the same embedding quantities on all 576 generated images (192 triplets, 48 prompts), including those never shown; there are no human votes for the added images, so only the embedding results can be compared.
| quantity | primary: shown to the participants (159 triplets) | sensitivity: all generated (192 triplets) | only the triplets with an image not shown (33 triplets) |
|---|---|---|---|
| BASE → AESTHETIC along the direction | +0.0443 (+0.0270 / +0.0635), p < .001 | +0.0416 (+0.0275 / +0.0592), p < .001 | +0.0288 (+0.0082 / +0.0496), p .045 |
| BASE → CONTROL along the direction | +0.0014 (-0.0120 / +0.0155), p .537 | +0.0001 (-0.0123 / +0.0122), p .989 | -0.0061 (-0.0274 / +0.0170), p .619 |
| AESTHETIC − CONTROL | +0.0429 (+0.0275 / +0.0599), p < .001 | +0.0415 (+0.0276 / +0.0570), p < .001 | +0.0349 (+0.0155 / +0.0553), p .009 |
| embedding transfer ratio | 18% (12% / 25%) | 18% (12% / 24%) | 15% (7% / 23%) |
| triplets with a positive score | 98 of 159 (62%) | 121 of 192 (63%) | 23 of 33 (70%) |
| specificity | +0.0592 (+0.0365 / +0.0816), p < .001 | +0.0574 (+0.0379 / +0.0786), p < .001 | +0.0488 (+0.0226 / +0.0771), p .011 |
| amount of shift ‖Δ_A‖ / ‖Δ_C‖ | 0.678 / 0.699 | 0.701 / 0.719 | 0.815 / 0.814 |
| direction: cosine of Δ_A / Δ_C with the aesthetic direction | +0.058 / +0.002 | +0.055 / +0.001 | +0.042 / -0.003 |
| cosine between Δ_A and Δ_C | +0.51 | +0.52 | +0.55 |


## Sensitivity — mean of the patch tokens instead of the CLS token

| quantity | CLS token | mean of patch tokens |
|---|---|---|
| A: standardised distance AESTHETIC – CONTROL | 0.26 | 0.23 |
| A: p, labels exchanged / within building type / random subsets | < .001 / < .001 / < .001 | < .001 / < .001 / < .001 |
| A: AUC AESTHETIC vs CONTROL | 0.78 | 0.66 |
| A: ratings predicted from embeddings (r) | 0.64 | 0.61 |
| B: embedding transfer ratio | 18% (p < .001) | 29% (p < .001) |
| B: specificity | +0.0592 (p < .001) | +0.0649 (p < .001) |
| A: relative source-set affinity, AUC (leave-one-out) / ρ with the ratings | 0.74 / +0.47 | 0.70 / +0.42 |
| B: amount difference ‖Δ_A‖ − ‖Δ_C‖ / direction difference (cosine) | -0.021 (p .129) / +0.055 (p < .001) | -0.014 (p .156) / +0.087 (p < .001) |
| B: share of the AESTHETIC − CONTROL shift due to direction | 96% | 98% |
| B: embedding transfer ratio on all generated images (192 triplets) | 18% (p < .001) | 29% (p < .001) |
| B: alignment vs rating difference (ρ) / vs pairwise share (ρ) / within triplet (r) | +0.23 / +0.04 / +0.13 | +0.22 / -0.07 / +0.17 |

