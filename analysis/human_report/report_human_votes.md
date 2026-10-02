# ARCH300 — pilot study analysis
*Investigating Aesthetic Transfer in Generative Models for Architectural Design: A Pilot Study on Consensus-Based Fine-Tuning and Human Evaluation* (ANALYSIS_PLAN.md). Estimates with 95% intervals; p values are uncorrected indications, not confirmatory tests.

## 1. Consensus (phase 1)

**Agreement between raters** (600 photographs, participant-centred): ICC(1) single rater **0.14**, ICC(k) photograph mean **0.86**, split-half **0.82**. ICC(1) says how much two raters agree on one photograph; ICC(k) how stable the consensus (the mean of all ratings) is.
**Agreement of each rater with the others**: mean r = **0.35** (95% CI 0.32 / 0.38), 161 raters.
**Aesthetic signal** — photos chosen for AESTHETIC minus photos in CONTROL, per rater: **+1.02 points** (95% CI +0.91 / +1.12), p = < .001, 159 raters. This is the difference in taste the fine-tuning could transfer (large by construction: AESTHETIC was selected with these ratings).
**Aesthetic signal with an independent rater** (leave-one-rater-out: the best 89 photos are re-selected without the rater, then scored with the rater's own votes): **+0.92 points** (95% CI +0.81 / +1.02), 159 raters. This is the unbiased estimate; the value above is inflated by regression to the mean.

## 2. Aesthetic transfer — AESTHETIC, CONTROL and BASE (all participants)

Ratings: 9413 from 158 participants · pairwise: 4796 choices (4010 decisive, 786 no preference = 16.4%) from 160 participants.
| condition | N ratings | mean | SD | participant-centred mean |
|---|---|---|---|---|
| AESTHETIC | 3126 | 3.93 | 1.74 | +0.025 |
| BASE | 3134 | 4.01 | 1.77 | +0.056 |
| CONTROL | 3153 | 3.86 | 1.76 | -0.080 |

**1. Linear mixed model** (random intercepts participant, prompt, scene; random condition slopes by participant):
| contrast | difference (points) | 95% CI | standardised d | p |
|---|---|---|---|---|
| **AESTHETIC − CONTROL** | +0.100 | +0.034 / +0.166 | +0.056 | .003 |
| BASE − CONTROL | +0.134 | +0.071 / +0.197 | +0.075 | < .001 |
| AESTHETIC − BASE | -0.034 | -0.101 / +0.033 | -0.019 | .320 |

Variance components: participant 1.206, prompt 0.340, scene 0.054, slopeA 0.017, slopeB 0.003, residual 1.584.
**AESTHETIC − CONTROL on the scale of the generated scenes**: +0.16 SD of the scene means (95% CI +0.05 / +0.26; SD between scenes = 0.63 points). **Pairwise preference expected from the ratings**: 52.2% for AESTHETIC (95% CI 50.8% / 53.7%; 50% = none), to be compared with the Bradley–Terry estimate below.
**2. Ordinal model** (odds of a higher score; participant fixed effects; SE clustered by participant):
| contrast | odds ratio | 95% CI | p |
|---|---|---|---|
| AESTHETIC vs CONTROL | 1.15 | 1.04–1.26 | .007 |
| BASE vs CONTROL | 1.20 | 1.09–1.31 | < .001 |

**3. Bradley–Terry model** (logistic, SE clustered by participant; last column = probability of being chosen, position-neutral):
| term | odds ratio | 95% CI | p | P(chosen) |
|---|---|---|---|---|
| position: left chosen | 0.97 | 0.89–1.06 | .543 | 49.3% |
| **AESTHETIC vs CONTROL** | 1.09 | 0.99–1.21 | .094 | 52.1% |
| BASE vs CONTROL | 1.13 | 1.04–1.24 | .005 | 53.1% |
| AESTHETIC vs BASE | 0.96 | 0.87–1.06 | .416 | 49.0% |

Mixed version (random participant and scene, variational Bayes): Intercept OR 0.97 (0.91–1.04); AESTHETIC OR 1.09 (1.01–1.18); BASE OR 1.14 (1.05–1.23).
**5. Equivalence (TOST)** — ratings: 90% CI +0.045 / +0.155 vs margin ±0.3 → **equivalent**; pairwise: 90% CI OR 1.00–1.19 vs 0.8–1.25 → **equivalent**.
**6. Bayesian evidence (approximate, BIC / unit-information prior)** — BF01 (evidence for *no* difference) ratings **0.5**, pairwise **23.3** (BF01 > 3 moderate, > 10 strong evidence for no effect; < 1/3 evidence for an effect). P(AESTHETIC > CONTROL): ratings 100%, pairwise 95%.

## 2b. What is transferred — building types and image characteristics

**16. AESTHETIC − CONTROL by building category** (participant-centred; bootstrap over participants; * = CI excludes 0):
| category | participants | difference | 95% CI |  |
|---|---|---|---|---|
| industrial | 6 | -1.00 | -1.83 / +0.00 |  |
| library | 25 | -0.30 | -0.76 / +0.20 |  |
| market | 19 | -0.29 | -0.68 / +0.13 |  |
| cinema | 22 | -0.26 | -0.82 / +0.23 |  |
| public | 13 | -0.12 | -0.46 / +0.23 |  |
| museum | 109 | -0.06 | -0.33 / +0.21 |  |
| house | 147 | -0.01 | -0.23 / +0.22 |  |
| restaurant | 21 | +0.02 | -0.43 / +0.50 |  |
| culture | 135 | +0.04 | -0.16 / +0.25 |  |
| housing | 140 | +0.08 | -0.14 / +0.31 |  |
| station | 87 | +0.09 | -0.19 / +0.38 |  |
| sport | 46 | +0.14 | -0.25 / +0.53 |  |
| office | 119 | +0.20 | -0.01 / +0.43 |  |
| tower | 40 | +0.23 | -0.15 / +0.62 |  |
| religious | 101 | +0.29 | +0.01 / +0.56 | * |
| school | 64 | +0.40 | +0.10 / +0.71 | * |
| civic | 13 | +0.46 | -0.31 / +1.23 |  |
| chapel | 24 | +0.52 | +0.00 / +1.08 |  |

**17. Image covariates** — skipped (no `--metrics` folder).

## 3. Personal transfer — the raters who built the consensus

Phase-2 participants: 160 · linked to phase 1: **33** · declared phase 1 (not linked): 16 · phase 2 only: 111.
**7. Primary contrast by group** (ratings: LMM points; pairwise: Bradley–Terry OR):
| group | participants | AESTHETIC−CONTROL (95% CI) | p | pairwise OR (95% CI) | p |
|---|---|---|---|---|---|
| linked to phase 1 | 32 | +0.047 (-0.08/+0.18) | .482 | 1.06 (0.84–1.32) | .637 |
| declared phase 1 | 16 | +0.000 (-0.19/+0.19) | 1.000 | 0.93 (0.72–1.22) | .621 |
| phase 2 only | 110 | +0.127 (+0.05/+0.20) | .001 | 1.12 (0.99–1.27) | .067 |

Interaction AESTHETIC × linked-to-phase-1: -0.076 points (95% CI -0.234/+0.082), p = .348.
**8. Individual alignment** — does the participant's own phase-1 preference for the photographs selected for AESTHETIC (vs CONTROL) predict their phase-2 preference for AESTHETIC images?
- phase-1 preference vs phase-2 rating difference AESTHETIC−CONTROL: Spearman ρ = **+0.06** (95% CI -0.27/+0.40), p = .765, n = 31.
- phase-1 preference vs phase-2 share of AESTHETIC choices vs CONTROL: Spearman ρ = **-0.14** (95% CI -0.47/+0.19), p = .448, n = 31.
- mean phase-1 preference for the AESTHETIC photographs: +0.88 points (97% of these participants preferred them).
- interaction AESTHETIC × phase-1 preference (per SD): +0.087 points (95% CI -0.046/+0.220), p = .200.
**9. Agreement with the phase-1 consensus** (correlation of the participant's ratings with the leave-one-out image means):
- mean agreement r = 0.35; agreement vs phase-2 AESTHETIC−CONTROL preference: ρ = **+0.11**, p = .570, n = 31.

## 4. Human evaluation

**10. Reliability of the phase-2 judgements** (participant-centred scores; images with ≥ 2 ratings):
| phase | images | ICC(1) single rater | ICC(k) image mean | split-half |
|---|---|---|---|---|
| phase 2 (generated images) | 477 | 0.24 | 0.86 | 0.87 |

ICC(1) = how much raters agree on a single image; low values mean that small differences between conditions need many ratings.
**11. Convergent validity** — image mean rating vs share of pairwise wins: Spearman ρ = **+0.17**, p = < .001, 477 images.
**12. Order and fatigue** — AESTHETIC effect when ratings came first vs pairs first: +0.046 (p = .410); change of scores over the session (per 60 items): -0.332 (p = < .001); AESTHETIC × position: +0.025 (p = .795).
Position in pairwise choices: left (or top) chosen 49.4% of decisive choices (50% expected without bias).
**13. Sensitivity** — AESTHETIC − CONTROL under different data choices (LMM intercepts only; Bradley–Terry):
| data | AESTHETIC−CONTROL points (95% CI) | p | pairwise OR (95% CI) | p |
|---|---|---|---|---|
| as planned (exclusions applied) | +0.100 (+0.04/+0.16) | .002 | 1.09 (0.99–1.21) | .094 |
| no exclusions | +0.114 (+0.05/+0.18) | < .001 | 1.08 (0.98–1.19) | .128 |
| without answers < 500 ms | +0.100 (+0.04/+0.16) | .002 | 1.09 (0.99–1.21) | .094 |
| without the 10% fastest participants | +0.114 (+0.05/+0.18) | < .001 | 1.05 (0.95–1.17) | .336 |
| completed participants only | +0.100 (+0.04/+0.16) | .002 | 1.09 (0.99–1.21) | .094 |


## 2c. Transfer ratio and "no preference" answers

**Transfer ratio** — AESTHETIC − CONTROL in phase 2 (+0.100 points, mixed model) divided by the aesthetic signal of phase 1 (+0.92 points): **11%** (95% CI 4% / 18%; 0% = nothing transferred, 100% = the whole difference between the training photos is found between the generated images).
**"No preference" kept as half a choice** — AESTHETIC share per participant: 50.8% (95% CI 48.1% / 53.4%), p = .570, 160 participants.
**Davidson model** not estimable (IndexError).

## 5. Towards a full study

**14. Minimum detectable effect** (80% power, α = .05) with the current data: **±0.09 points** on the 1–7 scale; pairwise odds ratio **1.16** (≈ 54% vs 46%). Smaller true effects would likely be missed.
**15. Sample size for a full study** (paired comparison, 80% power, α = .05; SD of the per-participant AESTHETIC − CONTROL differences in the pilot = 0.47): to detect **0.3 points** ≈ **22 participants**; to detect the observed difference (+0.104 points) ≈ 167; to detect the observed pairwise preference (50.5% AESTHETIC) ≈ 12412. The pilot has 158 participants with both conditions rated. Orders of magnitude, assuming the pilot's variability.
