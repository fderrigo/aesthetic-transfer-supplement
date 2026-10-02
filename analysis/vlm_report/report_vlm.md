# ARCH300 — evaluation of the generated images by three vision-language models

Exploratory analysis (ANALYSIS_PLAN.md, section 7). Rubric `VLM_RUBRIC_V1`; models: anthropic `claude-opus-5-5`; google `gemini-3.8-flash`; openai `gpt-5.6-sol`. One image per request, blind to condition, prompt and seed. Seed 20261003; 10000 permutations, 2000 bootstrap samples.

The scores are observations of the models, not measures of beauty or of architectural quality. Each model was trained on images from the web and has its own taste. Composites: architectural appearance = mean of 4 items; representation quality = mean of 5 items. "Three models" = mean of the scores standardised within model (1 = one standard deviation of the images of the primary set); the columns in points are on the 1–7 scale of each model.


## 0. Data

| model | images evaluated | complete triplets, primary set | complete triplets, all | architectural appearance, mean (SD) | representation quality, mean (SD) |
|---|---|---|---|---|---|
| openai | 576 / 576 | 159 | 192 | 5.91 (0.35) | 5.92 (0.32) |
| anthropic | 576 / 576 | 159 | 192 | 4.07 (0.51) | 4.71 (0.43) |
| google | 576 / 576 | 159 | 192 | 6.40 (0.87) | 6.42 (0.66) |
| three models (z) | 576 / 576 | 159 | 192 | – | – |

Reliability of the instrument (test–retest, agreement between the models, use of the scale), computed without the key: `full_instrument_report.md`.

Intervals and p values come from two different procedures (bootstrap of the triplets by prompt; sign-flip of the prompt means), so an interval that just touches zero and a p value just below .05, or the reverse, can occur together: such results are borderline and are to be read as such.


## 1. Main comparison: AESTHETIC − CONTROL, same prompt and seed

| composite | model | difference (z) | 95% CI | p | difference in points (95% CI) | triplets |
|---|---|---|---|---|---|---|
| architectural_appearance | **three models (z)** | +0.149 | +0.028 / +0.277 | .016 (Holm .016) | – | 159 |
|  | openai | +0.062 | -0.123 / +0.276 | .457 | +0.022 (-0.046 / +0.088) | 159 |
|  | anthropic | +0.157 | -0.027 / +0.354 | .047 | +0.080 (-0.014 / +0.173) | 159 |
|  | google | +0.228 | +0.081 / +0.381 | .015 | +0.198 (+0.063 / +0.334) | 159 |
| representation_quality | **three models (z)** | +0.245 | +0.136 / +0.366 | < .001 (Holm .001) | – | 159 |
|  | openai | +0.344 | +0.143 / +0.549 | .014 | +0.109 (+0.042 / +0.176) | 159 |
|  | anthropic | +0.047 | -0.118 / +0.215 | .436 | +0.020 (-0.052 / +0.091) | 159 |
|  | google | +0.344 | +0.182 / +0.511 | < .001 | +0.226 (+0.123 / +0.334) | 159 |

Do the three models point the same way? architectural_appearance: yes (+, +, +); representation_quality: yes (+, +, +).


### Which block carries the difference

| model | architectural − representation (z) | 95% CI | p |
|---|---|---|---|
| three models (z) | -0.096 | -0.219 / +0.028 | .354 |
| openai | -0.282 | -0.482 / -0.057 | .062 |
| anthropic | +0.110 | -0.038 / +0.278 | .081 |
| google | -0.116 | -0.202 / -0.029 | .012 |

Positive = the AESTHETIC − CONTROL difference is larger for the architectural appearance than for the quality of the representation.


## 2. Secondary: the nine items

| item | three models (z) | 95% CI | p | p (Holm, 9) | openai, points | anthropic, points | google, points |
|---|---|---|---|---|---|---|---|
| architecture: spatial_geometric_coherence | +0.091 | -0.036 / +0.223 | .117 | .303 | -0.01 (p .951) | +0.09 (p .043) | +0.13 (p .127) |
| architecture: proportional_coherence | +0.206 | +0.076 / +0.351 | .003 | .017 | +0.05 (p .197) | +0.09 (p .088) | +0.22 (p .003) |
| architecture: apparent_constructability | +0.108 | -0.000 / +0.222 | .076 | .303 | +0.01 (p .975) | +0.06 (p .208) | +0.22 (p .026) |
| architecture: component_coherence | +0.164 | +0.052 / +0.281 | .002 | .017 | +0.04 (p .223) | +0.08 (p .065) | +0.23 (p .007) |
| representation: lighting | +0.123 | -0.010 / +0.248 | .195 | .303 | +0.08 (p .249) | -0.09 (p .025) | +0.21 (p < .001) |
| representation: material_rendering | +0.210 | +0.119 / +0.305 | < .001 | < .001 | +0.19 (p < .001) | -0.03 (p .870) | +0.25 (p .001) |
| representation: photographic_realism | +0.085 | -0.014 / +0.188 | .078 | .303 | +0.04 (p .747) | +0.01 (p .503) | +0.19 (p .033) |
| representation: photographic_composition | +0.271 | +0.133 / +0.405 | .004 | .021 | +0.13 (p .067) | +0.16 (p .031) | +0.14 (p .019) |
| representation: overall_visual_quality | +0.290 | +0.160 / +0.423 | < .001 | .002 | +0.11 (p .028) | +0.06 (p .201) | +0.33 (p < .001) |


### Other contrasts between conditions

| contrast | composite | three models (z) | openai (z) | anthropic (z) | google (z) |
|---|---|---|---|---|---|
| BASE − CONTROL | architectural_appearance | -0.244 (-0.364 / -0.124), p < .001 | -0.267 (-0.454 / -0.070), p .014 | -0.148 (-0.320 / +0.023), p .211 | -0.318 (-0.526 / -0.128), p .003 |
| BASE − CONTROL | representation_quality | +0.068 (-0.061 / +0.195), p .293 | +0.170 (-0.008 / +0.351), p .155 | +0.137 (-0.014 / +0.291), p .051 | -0.103 (-0.326 / +0.125), p .358 |
| AESTHETIC − BASE | architectural_appearance | +0.393 (+0.270 / +0.527), p < .001 | +0.329 (+0.141 / +0.541), p < .001 | +0.305 (+0.132 / +0.490), p .001 | +0.546 (+0.384 / +0.723), p < .001 |
| AESTHETIC − BASE | representation_quality | +0.177 (+0.056 / +0.304), p .012 | +0.174 (+0.004 / +0.355), p .114 | -0.090 (-0.253 / +0.069), p .161 | +0.447 (+0.277 / +0.625), p < .001 |


## 3. Agreement with the human votes and with the embedding analysis

**Per image.** Does an image with a higher VLM score also have a higher human rating (ratings centred on each participant's mean, exclusions applied)? "All images" compares images of different prompts; "within triplet" compares only the three images of the same prompt and seed.

| composite | model | all images (Spearman, 95% CI) | within triplet (Pearson, 95% CI) | images |
|---|---|---|---|---|
| architectural_appearance | three models (z) | ρ +0.00 (-0.14 / +0.15) | r +0.01 (-0.08 / +0.10) | 477 |
|  | openai | ρ -0.16 (-0.29 / -0.02) | r +0.00 (-0.09 / +0.10) | 477 |
|  | anthropic | ρ +0.11 (-0.04 / +0.25) | r +0.04 (-0.07 / +0.14) | 477 |
|  | google | ρ -0.05 (-0.17 / +0.10) | r -0.02 (-0.10 / +0.06) | 477 |
| representation_quality | three models (z) | ρ +0.07 (-0.05 / +0.20) | r +0.07 (-0.05 / +0.18) | 477 |
|  | openai | ρ -0.04 (-0.17 / +0.09) | r +0.08 (-0.03 / +0.18) | 477 |
|  | anthropic | ρ +0.15 (+0.01 / +0.28) | r +0.04 (-0.08 / +0.16) | 477 |
|  | google | ρ +0.02 (-0.11 / +0.15) | r +0.02 (-0.10 / +0.13) | 477 |

**Per triplet.** Where the VLM score favours AESTHETIC over CONTROL more, do the participants (and the embedding analysis) do the same?

| composite | model | vs rating difference | vs share of AESTHETIC choices | vs representation transfer score |
|---|---|---|---|---|
| architectural_appearance | three models (z) | ρ +0.13 (-0.04 / +0.28), p .099 | ρ +0.17 (+0.00 / +0.31), p .036 | ρ -0.27 (-0.41 / -0.11), p < .001 |
|  | openai | ρ +0.07 (-0.10 / +0.25), p .373 | ρ +0.16 (-0.02 / +0.33), p .044 | ρ -0.20 (-0.34 / -0.06), p .011 |
|  | anthropic | ρ +0.14 (-0.02 / +0.30), p .089 | ρ +0.18 (+0.03 / +0.32), p .020 | ρ -0.19 (-0.34 / -0.01), p .018 |
|  | google | ρ +0.08 (-0.07 / +0.22), p .321 | ρ -0.11 (-0.26 / +0.05), p .180 | ρ -0.12 (-0.30 / +0.06), p .122 |
| representation_quality | three models (z) | ρ +0.14 (-0.02 / +0.29), p .086 | ρ +0.16 (-0.00 / +0.32), p .040 | ρ -0.16 (-0.33 / +0.02), p .042 |
|  | openai | ρ +0.06 (-0.08 / +0.22), p .424 | ρ +0.12 (-0.03 / +0.26), p .129 | ρ -0.16 (-0.30 / -0.01), p .049 |
|  | anthropic | ρ +0.17 (+0.00 / +0.32), p .034 | ρ +0.22 (+0.06 / +0.37), p .006 | ρ -0.13 (-0.29 / +0.04), p .113 |
|  | google | ρ +0.06 (-0.09 / +0.22), p .468 | ρ +0.01 (-0.15 / +0.16), p .942 | ρ -0.09 (-0.25 / +0.07), p .280 |


## 4. Sensitivity: all the generated images

| composite | model | primary: triplets shown to the participants (z) | all triplets (z) |
|---|---|---|---|
| architectural_appearance | three models (z) | +0.149 (+0.028 / +0.277), p .016, n 159 | +0.113 (-0.012 / +0.236), p .089, n 192 |
|  | openai | +0.062 (-0.123 / +0.276), p .457, n 159 | -0.018 (-0.221 / +0.203), p .874, n 192 |
|  | anthropic | +0.157 (-0.027 / +0.354), p .047, n 159 | +0.179 (-0.013 / +0.372), p .073, n 192 |
|  | google | +0.228 (+0.081 / +0.381), p .015, n 159 | +0.178 (+0.039 / +0.315), p .014, n 192 |
| representation_quality | three models (z) | +0.245 (+0.136 / +0.366), p < .001, n 159 | +0.242 (+0.121 / +0.353), p < .001, n 192 |
|  | openai | +0.344 (+0.143 / +0.549), p .014, n 159 | +0.334 (+0.121 / +0.534), p .004, n 192 |
|  | anthropic | +0.047 (-0.118 / +0.215), p .436, n 159 | +0.087 (-0.070 / +0.251), p .290, n 192 |
|  | google | +0.344 (+0.182 / +0.511), p < .001, n 159 | +0.304 (+0.166 / +0.433), p < .001, n 192 |

The standardisation is the same in the two columns (mean and standard deviation of the primary set).

