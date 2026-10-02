# VLM evaluation — instrument report (replication)

Read without the key: no condition, prompt or seed is known here. Rubric `VLM_RUBRIC_V1`. 346 evaluations requested per model (288 images, 58 repeated).

## 1. Completeness

| provider | model returned | valid evaluations | failed attempts | median time | input tokens per request | output tokens per request (reasoning included) |
|---|---|---|---|---|---|---|
| openai | gpt-5.6-sol | 346 / 346 | 2 | 3.1 s | 1863 | 146 |
| anthropic | claude-opus-5-5 | 346 / 346 | 0 | 2.9 s | 2906 | 97 |
| google | gemini-3.8-flash | 346 / 346 | 0 | 4.0 s | 1530 | 574 |

## 2. Use of the scale

Mean (standard deviation), range, and share of scores 6–7, on the images evaluated once (repeats left out). An item where almost every image is at 6–7 for every model is not informative.

| item | openai | anthropic | google | 6–7, all models |
|---|---|---|---|---|
| spatial_geometric_coherence | 6.19 (0.53), 3–7, 96% | 4.06 (0.64), 3–6, 0% | 6.55 (0.89), 2–7, 90% | 62% |
| proportional_coherence | 5.94 (0.31), 4–7, 93% | 4.50 (0.53), 3–6, 0% | 6.53 (0.81), 3–7, 90% | 61% |
| apparent_constructability | 5.74 (0.62), 2–7, 73% | 3.97 (0.68), 3–6, 0% | 6.32 (1.02), 2–7, 87% | 53% |
| component_coherence | 5.90 (0.43), 3–7, 90% | 3.86 (0.61), 3–6, 0% | 6.43 (0.97), 3–7, 87% | 59% |
| lighting | 5.92 (0.39), 5–7, 88% | 5.10 (0.41), 4–6, 14% | 6.62 (0.56), 5–7, 96% | 66% |
| material_rendering | 5.74 (0.47), 4–7, 74% | 4.70 (0.54), 3–6, 1% | 6.55 (0.73), 4–7, 91% | 55% |
| photographic_realism | 6.44 (0.75), 3–7, 90% | 3.89 (0.63), 2–6, 0% | 6.50 (1.04), 3–7, 85% | 58% |
| photographic_composition | 5.60 (0.53), 4–6, 61% | 5.15 (0.53), 4–6, 23% | 5.97 (0.49), 4–7, 88% | 57% |
| overall_visual_quality | 5.88 (0.35), 4–7, 88% | 4.64 (0.52), 3–6, 1% | 6.43 (0.82), 3–7, 88% | 59% |

| composite | openai | anthropic | google |
|---|---|---|---|
| architectural_appearance | 5.94 (0.39) | 4.09 (0.55) | 6.46 (0.90) |
| representation_quality | 5.92 (0.33) | 4.70 (0.40) | 6.41 (0.65) |

Correlation between the two composites within each model (if close to 1 the model does not separate architecture from representation):

| provider | Spearman ρ |
|---|---|
| openai | +0.43 |
| anthropic | +0.59 |
| google | +0.72 |

## 3. Test–retest

The same image sent twice in two independent requests (58 images per model).

| provider | pairs | items identical | items within 1 point | mean absolute difference (items) | ICC architectural composite | ICC representation composite | mean absolute difference of the composites (arch. / repr.) |
|---|---|---|---|---|---|---|---|
| openai | 58 | 81% | 99% | 0.20 | 0.80 | 0.50 | 0.18 / 0.20 |
| anthropic | 58 | 87% | 100% | 0.13 | 0.92 | 0.77 | 0.09 / 0.16 |
| google | 58 | 74% | 97% | 0.30 | 0.77 | 0.74 | 0.38 / 0.24 |

## 4. Agreement between the three models

On the images evaluated once. Spearman ρ between pairs of models and ICC(2,1) (absolute agreement) across the three.

| measure | openai – anthropic | openai – google | anthropic – google | ICC (raw scores) | ICC (scores standardised within model) |
|---|---|---|---|---|---|
| spatial_geometric_coherence | +0.32 | +0.16 | +0.28 | 0.06 | 0.29 |
| proportional_coherence | +0.19 | +0.22 | +0.27 | 0.05 | 0.24 |
| apparent_constructability | +0.43 | +0.39 | +0.36 | 0.10 | 0.38 |
| component_coherence | +0.28 | +0.19 | +0.32 | 0.06 | 0.31 |
| lighting | +0.07 | +0.12 | +0.06 | 0.02 | 0.08 |
| material_rendering | -0.04 | +0.13 | +0.20 | 0.04 | 0.12 |
| photographic_realism | +0.19 | +0.18 | +0.28 | 0.05 | 0.23 |
| photographic_composition | +0.39 | +0.38 | +0.27 | 0.22 | 0.37 |
| overall_visual_quality | +0.19 | +0.24 | +0.19 | 0.05 | 0.21 |
| architectural_appearance | +0.42 | +0.34 | +0.37 | 0.07 | 0.37 |
| representation_quality | +0.14 | +0.19 | +0.17 | 0.04 | 0.21 |

