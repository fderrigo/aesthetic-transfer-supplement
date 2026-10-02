# VLM evaluation — instrument report (full)

Read without the key: no condition, prompt or seed is known here. Rubric `VLM_RUBRIC_V1`. 691 evaluations requested per model (576 images, 115 repeated).

## 1. Completeness

| provider | model returned | valid evaluations | failed attempts | median time | input tokens per request | output tokens per request (reasoning included) |
|---|---|---|---|---|---|---|
| openai | gpt-5.6-sol | 691 / 691 | 1 | 3.2 s | 1863 | 147 |
| anthropic | claude-opus-5-5 | 691 / 691 | 20 | 3.1 s | 2906 | 97 |
| google | gemini-3.8-flash | 691 / 691 | 1 | 4.8 s | 1530 | 578 |

## 2. Use of the scale

Mean (standard deviation), range, and share of scores 6–7, on the images evaluated once (repeats left out). An item where almost every image is at 6–7 for every model is not informative.

| item | openai | anthropic | google | 6–7, all models |
|---|---|---|---|---|
| spatial_geometric_coherence | 6.13 (0.46), 4–7, 95% | 4.05 (0.64), 2–6, 1% | 6.46 (0.90), 3–7, 88% | 61% |
| proportional_coherence | 5.94 (0.32), 5–7, 92% | 4.47 (0.54), 3–6, 1% | 6.47 (0.79), 3–7, 90% | 61% |
| apparent_constructability | 5.66 (0.61), 3–7, 66% | 3.90 (0.67), 2–6, 1% | 6.22 (1.03), 2–7, 84% | 50% |
| component_coherence | 5.91 (0.39), 4–7, 90% | 3.84 (0.60), 2–6, 1% | 6.38 (0.95), 3–7, 87% | 59% |
| lighting | 5.94 (0.36), 4–7, 91% | 5.11 (0.45), 4–6, 16% | 6.61 (0.58), 4–7, 96% | 68% |
| material_rendering | 5.75 (0.51), 4–7, 73% | 4.66 (0.58), 3–6, 3% | 6.49 (0.81), 3–7, 88% | 55% |
| photographic_realism | 6.36 (0.73), 4–7, 88% | 3.89 (0.68), 2–7, 1% | 6.46 (1.03), 2–7, 85% | 58% |
| photographic_composition | 5.65 (0.54), 4–7, 65% | 5.19 (0.55), 3–6, 26% | 6.06 (0.52), 3–7, 92% | 61% |
| overall_visual_quality | 5.90 (0.34), 4–7, 89% | 4.63 (0.55), 3–6, 2% | 6.40 (0.82), 3–7, 89% | 60% |

| composite | openai | anthropic | google |
|---|---|---|---|
| architectural_appearance | 5.91 (0.36) | 4.06 (0.54) | 6.38 (0.89) |
| representation_quality | 5.92 (0.32) | 4.70 (0.44) | 6.40 (0.67) |

Correlation between the two composites within each model (if close to 1 the model does not separate architecture from representation):

| provider | Spearman ρ |
|---|---|
| openai | +0.44 |
| anthropic | +0.62 |
| google | +0.76 |

## 3. Test–retest

The same image sent twice in two independent requests (115 images per model).

| provider | pairs | items identical | items within 1 point | mean absolute difference (items) | ICC architectural composite | ICC representation composite | mean absolute difference of the composites (arch. / repr.) |
|---|---|---|---|---|---|---|---|
| openai | 115 | 83% | 100% | 0.17 | 0.82 | 0.75 | 0.16 / 0.17 |
| anthropic | 115 | 88% | 100% | 0.12 | 0.90 | 0.84 | 0.10 / 0.13 |
| google | 115 | 75% | 98% | 0.27 | 0.86 | 0.85 | 0.28 / 0.25 |

Test–retest ICC per item:

| item | openai | anthropic | google |
|---|---|---|---|
| spatial_geometric_coherence | 0.69 | 0.88 | 0.84 |
| proportional_coherence | 0.55 | 0.76 | 0.82 |
| apparent_constructability | 0.74 | 0.88 | 0.83 |
| component_coherence | 0.74 | 0.86 | 0.84 |
| lighting | 0.66 | 0.72 | 0.69 |
| material_rendering | 0.72 | 0.87 | 0.81 |
| photographic_realism | 0.73 | 0.77 | 0.83 |
| photographic_composition | 0.63 | 0.76 | 0.72 |
| overall_visual_quality | 0.73 | 0.75 | 0.83 |

## 4. Agreement between the three models

On the images evaluated once. Spearman ρ between pairs of models and ICC(2,1) (absolute agreement) across the three.

| measure | openai – anthropic | openai – google | anthropic – google | ICC (raw scores) | ICC (scores standardised within model) |
|---|---|---|---|---|---|
| spatial_geometric_coherence | +0.28 | +0.25 | +0.28 | 0.06 | 0.29 |
| proportional_coherence | +0.24 | +0.17 | +0.27 | 0.05 | 0.24 |
| apparent_constructability | +0.33 | +0.40 | +0.36 | 0.10 | 0.36 |
| component_coherence | +0.27 | +0.20 | +0.29 | 0.05 | 0.26 |
| lighting | +0.06 | +0.13 | +0.11 | 0.03 | 0.11 |
| material_rendering | +0.15 | +0.24 | +0.21 | 0.07 | 0.22 |
| photographic_realism | +0.09 | +0.27 | +0.24 | 0.05 | 0.22 |
| photographic_composition | +0.46 | +0.36 | +0.42 | 0.26 | 0.42 |
| overall_visual_quality | +0.23 | +0.20 | +0.25 | 0.06 | 0.24 |
| architectural_appearance | +0.34 | +0.35 | +0.32 | 0.06 | 0.34 |
| representation_quality | +0.20 | +0.23 | +0.27 | 0.06 | 0.26 |

