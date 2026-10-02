# ARCH300 — replication across training seeds: evaluation by three vision-language models

Secondary analysis (ANALYSIS_PLAN.md, section 8.6). Rubric `VLM_RUBRIC_V1`; models: anthropic `claude-opus-5-5`; google `gemini-3.8-flash`; openai `gpt-5.6-sol`. Subset fixed before the replication images existed: 48 prompt × generation-seed cells (24 prompts) × 6 LoRA = 288 images, all evaluated in the same batch, blind to corpus, training seed, prompt and seed. Scores standardised within model over these images; intervals: bootstrap over prompts (5000 samples, seed 20261006). The scores are observations of the models, not measures of beauty or of architectural quality.

## 0. Data

| model | images evaluated | architectural appearance, mean (SD) | representation quality, mean (SD) |
|---|---|---|---|
| openai | 288 / 288 | 5.94 (0.39) | 5.92 (0.33) |
| anthropic | 288 / 288 | 4.09 (0.55) | 4.70 (0.40) |
| google | 288 / 288 | 6.46 (0.90) | 6.41 (0.65) |

Reliability of the instrument on this batch (test–retest, agreement between the models), computed without the key: `replication_instrument_report.md`.

## 1. Measures named in advance: AESTHETIC − CONTROL at equal training seed (three models, z)

A measure **reappears** if the contrast is positive for the three training seeds. Section 7 (original experiment, 159 triplets) is shown for reference. Three positive signs alone are weak evidence: with no real difference they occur by chance about once in eight (checked with shuffled labels before unblinding). The interval of the mean of the three seeds says how firm each result is.

| measure | section 7 | training seed A = 1254 | training seed B = 9865 | training seed C = 42160 | mean of the three seeds | positive for the three seeds |
|---|---|---|---|---|---|---|
| **representation_quality** | +0.245 | -0.162 (-0.372 / +0.055) | +0.224 (+0.022 / +0.419) | -0.228 (-0.429 / -0.022) | -0.056 (-0.182 / +0.073) | no |
| **proportional_coherence** | +0.206 | -0.173 (-0.428 / +0.055) | +0.033 (-0.153 / +0.241) | +0.004 (-0.167 / +0.172) | -0.046 (-0.156 / +0.059) | no |
| **component_coherence** | +0.164 | -0.269 (-0.540 / -0.004) | -0.061 (-0.265 / +0.147) | -0.059 (-0.236 / +0.139) | -0.130 (-0.245 / -0.019) | no |
| **material_rendering** | +0.210 | -0.199 (-0.408 / +0.007) | +0.232 (+0.051 / +0.408) | -0.029 (-0.210 / +0.151) | +0.001 (-0.118 / +0.119) | no |
| **photographic_composition** | +0.271 | -0.057 (-0.229 / +0.118) | +0.283 (+0.070 / +0.499) | -0.307 (-0.563 / -0.049) | -0.027 (-0.180 / +0.131) | no |
| architectural_appearance (not named, for reference) | +0.149 | -0.190 (-0.484 / +0.093) | -0.055 (-0.239 / +0.133) | -0.078 (-0.240 / +0.091) | -0.108 (-0.241 / +0.014) | no |

**0 of the 5 measures named in advance reappear** (positive for the three training seeds): none. Not reappearing: representation_quality, proportional_coherence, component_coherence, material_rendering, photographic_composition.

## 2. The same contrasts, model by model (points on the 1–7 scale of each model)

| measure | model | training seed A | training seed B | training seed C | positive for the three seeds |
|---|---|---|---|---|---|
| representation_quality | openai | -0.11 (-0.22 / -0.01) | +0.10 (-0.01 / +0.21) | -0.07 (-0.19 / +0.04) | no |
|  | anthropic | -0.01 (-0.14 / +0.15) | -0.04 (-0.15 / +0.08) | -0.19 (-0.30 / -0.07) | no |
|  | google | -0.07 (-0.22 / +0.08) | +0.30 (+0.08 / +0.52) | -0.00 (-0.25 / +0.27) | no |
| proportional_coherence | openai | -0.17 (-0.33 / -0.02) | -0.04 (-0.15 / +0.04) | -0.02 (-0.10 / +0.06) | no |
|  | anthropic | +0.06 (-0.08 / +0.23) | -0.04 (-0.21 / +0.12) | +0.00 (-0.12 / +0.12) | no |
|  | google | -0.08 (-0.29 / +0.15) | +0.25 (-0.04 / +0.54) | +0.06 (-0.27 / +0.44) | no |
| component_coherence | openai | -0.23 (-0.44 / -0.06) | -0.12 (-0.25 / -0.02) | -0.02 (-0.12 / +0.08) | no |
|  | anthropic | -0.06 (-0.27 / +0.19) | -0.10 (-0.31 / +0.12) | -0.10 (-0.27 / +0.06) | no |
|  | google | -0.17 (-0.44 / +0.10) | +0.27 (-0.02 / +0.58) | +0.04 (-0.33 / +0.44) | no |
| material_rendering | openai | -0.21 (-0.35 / -0.06) | +0.12 (-0.06 / +0.33) | +0.04 (-0.10 / +0.19) | no |
|  | anthropic | -0.02 (-0.23 / +0.17) | +0.00 (-0.17 / +0.17) | -0.12 (-0.25 / +0.00) | no |
|  | google | -0.08 (-0.25 / +0.10) | +0.31 (+0.06 / +0.56) | +0.04 (-0.23 / +0.33) | no |
| photographic_composition | openai | -0.02 (-0.17 / +0.12) | +0.25 (+0.10 / +0.40) | -0.23 (-0.42 / -0.04) | no |
|  | anthropic | +0.02 (-0.15 / +0.19) | +0.04 (-0.10 / +0.19) | -0.15 (-0.31 / +0.04) | no |
|  | google | -0.08 (-0.21 / +0.04) | +0.15 (-0.02 / +0.31) | -0.10 (-0.31 / +0.10) | no |
| architectural_appearance | openai | -0.15 (-0.34 / +0.01) | -0.12 (-0.26 / -0.02) | -0.05 (-0.17 / +0.05) | no |
|  | anthropic | -0.03 (-0.23 / +0.19) | -0.07 (-0.23 / +0.10) | -0.10 (-0.20 / +0.01) | no |
|  | google | -0.13 (-0.35 / +0.12) | +0.26 (-0.04 / +0.54) | +0.07 (-0.30 / +0.47) | no |

## 3. All nine items (for completeness; not a search for new differences)

| item (* = named in advance) | training seed A | training seed B | training seed C | positive for the three seeds |
|---|---|---|---|---|
| architecture: spatial_geometric_coherence | -0.193 (-0.500 / +0.114) | -0.022 (-0.233 / +0.198) | -0.085 (-0.250 / +0.085) | no |
| architecture: proportional_coherence * | -0.173 (-0.431 / +0.062) | +0.033 (-0.163 / +0.241) | +0.004 (-0.167 / +0.171) | no |
| architecture: apparent_constructability | -0.081 (-0.348 / +0.182) | -0.066 (-0.230 / +0.104) | -0.099 (-0.286 / +0.091) | no |
| architecture: component_coherence * | -0.269 (-0.547 / -0.013) | -0.061 (-0.264 / +0.149) | -0.059 (-0.233 / +0.137) | no |
| representation: lighting | -0.167 (-0.390 / +0.054) | +0.312 (+0.121 / +0.517) | -0.173 (-0.374 / +0.034) | no |
| representation: material_rendering * | -0.199 (-0.400 / +0.012) | +0.232 (+0.057 / +0.410) | -0.029 (-0.209 / +0.150) | no |
| representation: photographic_realism | -0.080 (-0.268 / +0.100) | -0.078 (-0.267 / +0.106) | -0.135 (-0.369 / +0.100) | no |
| representation: photographic_composition * | -0.057 (-0.230 / +0.119) | +0.283 (+0.071 / +0.503) | -0.307 (-0.555 / -0.039) | no |
| representation: overall_visual_quality | -0.144 (-0.361 / +0.065) | +0.288 (+0.095 / +0.479) | -0.233 (-0.415 / -0.051) | no |

## 4. Corpus effect against training-seed effect (three models, z)

Same corpus, different training seed: if these differences are as large as the contrasts of section 1, the VLM reading depends on the training run as much as on the corpus.

| composite | comparison | difference (95% CI) |
|---|---|---|
| architectural_appearance | AESTHETIC: A − B | -0.075 (-0.314 / +0.165) |
| architectural_appearance | AESTHETIC: A − C | +0.019 (-0.180 / +0.214) |
| architectural_appearance | AESTHETIC: B − C | +0.094 (-0.037 / +0.224) |
| architectural_appearance | CONTROL: A − B | +0.061 (-0.112 / +0.244) |
| architectural_appearance | CONTROL: A − C | +0.131 (-0.081 / +0.410) |
| architectural_appearance | CONTROL: B − C | +0.070 (-0.158 / +0.339) |
| architectural_appearance | **R = mean |corpus contrast| / mean |between-seed difference|** | **1.4** |
| representation_quality | AESTHETIC: A − B | -0.083 (-0.282 / +0.112) |
| representation_quality | AESTHETIC: A − C | +0.120 (-0.039 / +0.287) |
| representation_quality | AESTHETIC: B − C | +0.204 (+0.044 / +0.355) |
| representation_quality | CONTROL: A − B | +0.302 (+0.073 / +0.540) |
| representation_quality | CONTROL: A − C | +0.054 (-0.175 / +0.288) |
| representation_quality | CONTROL: B − C | -0.248 (-0.469 / -0.015) |
| representation_quality | **R = mean |corpus contrast| / mean |between-seed difference|** | **1.2** |

## 5. Consistency with the original experiment (check added after unblinding)

Section 7 compared the two original LoRA, which had different training seeds: AESTHETIC with seed A against CONTROL with seed B. The same unpaired contrast, computed on this batch (the 48 cells of the subset, images evaluated again), and the two CONTROL runs it depends on:

| measure | section 7 (159 triplets) | same unpaired contrast, this batch (48 cells) | original CONTROL LoRA − the two other CONTROL LoRA |
|---|---|---|---|
| architectural_appearance | +0.149 | -0.130 (-0.415 / +0.134) | +0.005 (-0.177 / +0.186) |
| representation_quality | +0.245 | +0.140 (-0.089 / +0.381) | -0.275 (-0.471 / -0.088) |
| proportional_coherence | +0.206 | -0.049 (-0.324 / +0.198) | -0.078 (-0.277 / +0.117) |
| component_coherence | +0.164 | -0.171 (-0.422 / +0.079) | -0.033 (-0.192 / +0.125) |
| material_rendering | +0.210 | +0.147 (-0.098 / +0.377) | -0.257 (-0.433 / -0.074) |
| photographic_composition | +0.271 | +0.148 (-0.037 / +0.326) | -0.243 (-0.393 / -0.094) |

The images of the two original LoRA in the subset were evaluated in section 7 and again here. Correlation between the two scores of the same image:

| model | images | architectural appearance (r) | representation quality (r) |
|---|---|---|---|
| openai | 96 | +0.77 | +0.73 |
| anthropic | 96 | +0.83 | +0.77 |
| google | 96 | +0.76 | +0.76 |

