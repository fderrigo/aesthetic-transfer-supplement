# ARCH300 — the captions of the two training sets

Exploratory analysis (ANALYSIS_PLAN.md, section 9.5). Caption snapshots of the frozen datasets `AESTHETIC-v2` (89 captions) and `CONTROL-v2` (89 captions), the same for the original and the replication runs. Mean length 35.3 and 35.1 words (Mann–Whitney p 0.89). 806 different words, 176 present in at least 5 captions. Captions describe the photographs: words can differ simply because the photographs differ.

## a. Words

Words whose frequency differs between the two sets after correction for the 176 words tested (Benjamini–Hochberg q < .05): **4**. Uncorrected p < .05: 10 words, against about 9 expected by chance alone.

The twenty words with the largest difference in share of captions:

| word | AESTHETIC captions | CONTROL captions | difference | p | q (BH) | generation prompts containing it (of 48) |
|---|---|---|---|---|---|---|
| green | 36 (40%) | 6 (7%) | +34% | 0.000 | 0.00 | 4 |
| perspective | 28 (31%) | 56 (63%) | -31% | 0.000 | 0.00 | 8 |
| sky | 79 (89%) | 54 (61%) | +28% | 0.000 | 0.00 | 4 |
| level | 41 (46%) | 61 (69%) | -22% | 0.004 | 0.10 | 7 |
| eye | 40 (45%) | 60 (67%) | -22% | 0.004 | 0.10 | 6 |
| glass | 39 (44%) | 22 (25%) | +19% | 0.011 | 0.22 | 11 |
| glazed | 4 (4%) | 18 (20%) | -16% | 0.002 | 0.08 | 1 |
| dark | 18 (20%) | 5 (6%) | +15% | 0.006 | 0.14 | 5 |
| long | 11 (12%) | 23 (26%) | -13% | 0.035 | 0.59 | 5 |
| river | 11 (12%) | 0 (0%) | +12% | 0.001 | 0.03 | 1 |
| front | 36 (40%) | 25 (28%) | +12% | 0.114 | 0.79 | 16 |
| concrete | 17 (19%) | 27 (30%) | -11% | 0.117 | 0.79 | 15 |
| street | 10 (11%) | 19 (21%) | -10% | 0.103 | 0.76 | 12 |
| tower | 17 (19%) | 25 (28%) | -9% | 0.216 | 0.95 | 7 |
| blue | 48 (54%) | 40 (45%) | +9% | 0.294 | 0.95 | 2 |
| lawn | 11 (12%) | 19 (21%) | -9% | 0.160 | 0.95 | 1 |
| volumes | 5 (6%) | 13 (15%) | -9% | 0.079 | 0.67 | 5 |
| windows | 5 (6%) | 13 (15%) | -9% | 0.079 | 0.67 | 8 |
| large | 18 (20%) | 26 (29%) | -9% | 0.224 | 0.95 | 6 |
| facade | 9 (10%) | 16 (18%) | -8% | 0.195 | 0.95 | 10 |

## b. Can the set be told from the caption alone?

Ridge classifier on the presence of the 176 words, 10-fold cross-validation: AUC **0.85** (95% CI 0.79 / 0.90); with the labels exchanged 0.49 (95% of the permutations between 0.37 and 0.60); permutation p = 0.001. For comparison, the images alone give AUC 0.78 (DINOv2 embeddings, section 6).

**Reading (9.5):** the captions tell the two sets apart better than chance: the text is a second channel, to be declared next to the images.

## c. Unbalanced words and the generation prompts

Of the 10 words with uncorrected p < .05, 10 also occur in the generation prompts: green (+34%; 4 prompts), perspective (-31%; 8 prompts), sky (+28%; 4 prompts), level (-22%; 7 prompts), eye (-22%; 6 prompts), glass (+19%; 11 prompts), glazed (-16%; 1 prompts), dark (+15%; 5 prompts), long (-13%; 5 prompts), river (+12%; 1 prompts).

