# VLM evaluation — instrument report (pilot)

Read without the key: no condition, prompt or seed is known here. Rubric `VLM_RUBRIC_V1`. 54 evaluations requested per model (45 images, 9 repeated).

## 1. Completeness

| provider | model returned | valid evaluations | failed attempts | median time | input tokens per request | output tokens per request (reasoning included) |
|---|---|---|---|---|---|---|
| openai | gpt-5.6-sol | 54 / 54 | 0 | 4.4 s | 1926 | 214 |
| anthropic | claude-opus-5-5 | 54 / 54 | 0 | 5.3 s | 3054 | 254 |
| google | gemini-3.8-flash | 54 / 54 | 0 | 5.3 s | 1578 | 633 |

## 2. Use of the scale

Mean (standard deviation), range, and share of scores 6–7, on the images evaluated once (repeats left out). An item where almost every image is at 6–7 for every model is not informative.

| item | openai | anthropic | google | 6–7, all models |
|---|---|---|---|---|
| spatial_geometric_coherence | 6.20 (0.55), 4–7, 98% | 4.07 (0.58), 3–5, 0% | 6.38 (1.05), 3–7, 87% | 61% |
| proportional_coherence | 5.98 (0.34), 5–7, 93% | 4.49 (0.51), 4–5, 0% | 6.29 (0.89), 4–7, 84% | 59% |
| apparent_constructability | 5.56 (0.69), 3–7, 60% | 3.89 (0.65), 3–5, 0% | 6.07 (1.16), 3–7, 82% | 47% |
| component_coherence | 5.96 (0.42), 4–7, 93% | 3.84 (0.67), 3–5, 0% | 6.31 (1.04), 3–7, 82% | 59% |
| lighting | 5.89 (0.38), 5–7, 87% | 5.33 (0.52), 4–6, 36% | 6.51 (0.63), 5–7, 93% | 72% |
| material_rendering | 5.80 (0.50), 4–7, 80% | 4.67 (0.64), 3–6, 7% | 6.36 (0.91), 4–7, 84% | 57% |
| photographic_realism | 6.36 (0.83), 3–7, 89% | 3.96 (0.80), 3–6, 2% | 6.33 (1.17), 3–7, 80% | 57% |
| photographic_composition | 5.69 (0.47), 5–6, 69% | 5.29 (0.46), 5–6, 29% | 6.09 (0.36), 5–7, 98% | 65% |
| overall_visual_quality | 5.89 (0.38), 4–6, 91% | 4.73 (0.58), 4–6, 7% | 6.33 (0.90), 4–7, 84% | 61% |

| composite | openai | anthropic | google |
|---|---|---|---|
| architectural_appearance | 5.92 (0.42) | 4.07 (0.54) | 6.26 (0.99) |
| representation_quality | 5.92 (0.34) | 4.80 (0.50) | 6.32 (0.73) |

Correlation between the two composites within each model (if close to 1 the model does not separate architecture from representation):

| provider | Spearman ρ |
|---|---|
| openai | +0.30 |
| anthropic | +0.49 |
| google | +0.72 |

## 3. Test–retest

The same image sent twice in two independent requests (9 images per model).

| provider | pairs | items identical | items within 1 point | mean absolute difference (items) | ICC architectural composite | ICC representation composite | mean absolute difference of the composites (arch. / repr.) |
|---|---|---|---|---|---|---|---|
| openai | 9 | 79% | 99% | 0.22 | 0.85 | 0.55 | 0.17 / 0.27 |
| anthropic | 9 | 84% | 100% | 0.16 | 0.90 | 0.75 | 0.14 / 0.13 |
| google | 9 | 59% | 96% | 0.44 | 0.89 | 0.71 | 0.44 / 0.44 |

## 4. Agreement between the three models

On the images evaluated once. Spearman ρ between pairs of models and ICC(2,1) (absolute agreement) across the three.

| measure | openai – anthropic | openai – google | anthropic – google | ICC (raw scores) | ICC (scores standardised within model) |
|---|---|---|---|---|---|
| spatial_geometric_coherence | +0.35 | +0.20 | +0.27 | 0.09 | 0.39 |
| proportional_coherence | +0.33 | +0.18 | +0.37 | 0.08 | 0.33 |
| apparent_constructability | +0.61 | +0.56 | +0.46 | 0.18 | 0.56 |
| component_coherence | +0.37 | +0.12 | +0.28 | 0.06 | 0.31 |
| lighting | +0.05 | -0.03 | -0.00 | 0.01 | 0.02 |
| material_rendering | +0.17 | +0.34 | +0.15 | 0.08 | 0.23 |
| photographic_realism | -0.19 | +0.08 | +0.28 | 0.04 | 0.11 |
| photographic_composition | +0.32 | +0.18 | -0.03 | 0.09 | 0.16 |
| overall_visual_quality | -0.01 | -0.01 | +0.22 | 0.05 | 0.12 |
| architectural_appearance | +0.44 | +0.45 | +0.35 | 0.10 | 0.47 |
| representation_quality | -0.01 | +0.18 | +0.11 | 0.04 | 0.11 |

## 5. Notes (pilot only)

Share of architectural notes that use words of aesthetic appreciation (beaut, elegant, attractive, pleasing, aesthetic, stunning, striking…): openai 0%; anthropic 2%; google 0%.

Notes of the images with the lowest and the highest architectural composite, for each model:

* **openai**, architecture 4.00 (constructability 3), representation 4.40 — *architecture:* The repeated rounded window modules establish a clear facade rhythm, but warped junctions, irregular gaps, and ambiguous support conditions reduce coherence and plausibility. *representation:* Daylight, exposure, and frontal framing are convincing, while smeared details, inconsistent surfaces, and distorted peripheral objects weaken photographic realism.
* **openai**, architecture 4.75 (constructability 4), representation 5.40 — *architecture:* The repeated organic window modules form a coherent facade system, though several warped junctions and the open ground-level support arrangement appear visually uncertain. *representation:* Consistent daylight and a clear frontal composition support realism, while some softened details, irregular edges, and ambiguous surface transitions reduce material and photographic credibility.
* **openai**, architecture 5.25 (constructability 5), representation 6.00 — *architecture:* The stacked rectilinear masses and recessed window bands are coherent, though irregular panel seams, deformation, and several visually tenuous overhangs reduce clarity. *representation:* The centered frontal photograph has plausible daylight, perspective, and material detail, with minor distraction from glare, facade damage, and peripheral street elements.
* **openai**, architecture 6.50 (constructability 7), representation 6.20 — *architecture:* The brick masses, recessed glazing, ground-level supports, and connecting volumes form a clear and constructively plausible geometric system. *representation:* Crisp perspective, consistent daylight and shadows, and legible brick, glass, and landscape textures create a highly realistic architectural photograph.
* **openai**, architecture 6.75 (constructability 6), representation 6.00 — *architecture:* The stepped masses, vertical facade ribs, narrow openings, and projecting entrance canopy form a highly coherent and plausibly supported ensemble. *representation:* Strong daylight, clear perspective, and balanced frontal framing produce a convincing image, with only slight softness and surface smoothing reducing fine material realism.
* **openai**, architecture 6.75 (constructability 6), representation 6.20 — *architecture:* The strongly ordered vertical facade elements, stepped masses, openings, and projecting entrance canopy form a highly coherent and plausibly supported composition. *representation:* Consistent daylight, natural street detail, and a centered frontal viewpoint provide strong photographic realism, with minor distraction from poles, wires, vehicles, and edge cropping.
* **anthropic**, architecture 3.25 (constructability 3), representation 4.20 — *architecture:* The curved balcony bands and porthole windows follow a loose rhythm, but irregular window types, melting-looking AC units, inconsistent floor alignment between bays and oddly tapered ground piers weaken the coherence and plausibility. *representation:* Overcast lighting is consistent and the street-level framing is reasonable, but smeared window details, warped porthole glazing and soft, blobby elements give away a synthetic appearance.
* **anthropic**, architecture 3.25 (constructability 3), representation 4.20 — *architecture:* The porthole grid facade starts orderly but devolves into melted, merging bulges and irregular circular openings in the middle, with an ambiguous ground-floor support zone and odd blob-like transitions. *representation:* Frontal framing and daylight are plausible, but smeared concrete textures, warped window frames, inconsistent glass reflections and blurry street details reveal synthetic artifacts.
* **anthropic**, architecture 3.25 (constructability 3), representation 4.20 — *architecture:* Stacked blob-like pod modules with porthole windows lack legible structure, show inconsistent column alignment and melting transitions, and sit oddly on a thin podium with an unclear connection to the base. *representation:* Lighting is soft and plausible and the frontal framing is clear, but warped window interiors, smeared curtains, odd rooftop fins and mushy surface detail reveal synthetic artifacts.
* **anthropic**, architecture 5.00 (constructability 5), representation 5.00 — *architecture:* Stone masses and glazed volume read coherently overall, though the cantilevered stone corner over glass, the ambiguous glass-to-stone junctions and the odd plinth step at the right introduce minor inconsistencies. *representation:* Warm low sun and clear sky are plausible and stonework is legible, but the oversized, overly uniform gravel foreground and some smeary glass reflections and edges reduce photographic realism.
* **anthropic**, architecture 5.00 (constructability 5), representation 5.20 — *architecture:* The tapered concrete tower and low blocky volumes read coherently, though the tower base junction with the canopy wing and the faceted tower edges are slightly ambiguous. *representation:* Consistent sunny lighting and film-like color feel plausible, but some soft, smeared details in the cars, background buildings and tower texture, plus an awkward cropped building at left, reduce realism and composition.
* **anthropic**, architecture 5.00 (constructability 5), representation 5.20 — *architecture:* Symmetrical stepped streamline massing with towers and fluted bays reads coherently, though some junctions between the central drum, canopy and towers are ambiguous and fin rhythms are slightly irregular. *representation:* Consistent sunny lighting and centered frontal framing work well, but smeared cars, garbled signage, indistinct pedestrians and soft blurred details betray synthetic artifacts.
* **google**, architecture 3.25 (constructability 3), representation 4.20 — *architecture:* The modular facade logic is disrupted by severe melting and geometric warping across several central window pods. *representation:* While concrete textures and sunlight behave plausibly, prominent generative synthesis artifacts distort openings and framing in the center of the image.
* **google**, architecture 3.25 (constructability 3), representation 4.40 — *architecture:* The repetitive modular pods exhibit irregular organic warping, inconsistent seams, and ambiguous structural connection details. *representation:* The frontal composition and daylighting are strong, but noticeable generative smearing and melting artifacts degrade overall realism.
* **google**, architecture 3.50 (constructability 3), representation 4.20 — *architecture:* The bold cubic cantilevers on the left degrade into chaotic, melted, and unconstructible faceted geometry toward the upper right facade. *representation:* While lighting and framing are effective, distinct generative artifacts appear in the distorted pedestrians, background curtain wall, and warped panel seams.
* **google**, architecture 7.00 (constructability 7), representation 7.00 — *architecture:* The structure exhibits clear modernist tectonic logic with plausible cantilevers, weathered concrete framing, and coherent integration of timber louvers and glazing. *representation:* The image exhibits exceptional photographic realism with nuanced natural diffuse lighting, authentic material weathering, and rich foliage detail.
* **google**, architecture 7.00 (constructability 7), representation 7.00 — *architecture:* The modular facade system of balconies, perforated weathering steel panels, and vertical louvers is exceptionally well-ordered and constructively convincing. *representation:* The image demonstrates pristine architectural photography with precise orthographic framing, sharp detail, and natural daylighting.
* **google**, architecture 7.00 (constructability 7), representation 6.80 — *architecture:* The brick volumes, articulated ribbon windows, and recessed glazed bay form a fully coherent and structurally plausible institutional ensemble. *representation:* The photograph displays authentic direct sunlight with consistent cast shadows, natural material textures, and convincing photographic depth.

