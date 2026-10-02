# Supplementary material — anonymous version for peer review

*From curated references to the generated image: transfer of human preferences in the fine-tuning of diffusion models
for architectural images* (pilot study)

Data, code, protocol and every result behind the manuscript, prepared for double-blind review. Nothing here identifies
the participants or the authors (see [ANONYMISATION.md](ANONYMISATION.md)).
[PAPER_MAP.md](PAPER_MAP.md) links every section, table, figure and number of the manuscript to the file that supports it.

## The study in six lines

1. **Phase 1.** 164 participants rated 600 Wikimedia Commons photographs of documented architecture on a 1–7 scale
   ("How aesthetically successful do you consider this architecture?"): 22,712 valid ratings.
2. **Two training sets of 89 photographs.** AESTHETIC: photographs selected by a preference rule (at least 10 ratings,
   mean at least 5, top 35%), defined after phase 1 and frozen before the datasets were built; not preregistered.
   CONTROL: photographs not selected, matched one to one on building type and style. The treatment is
   *preference-guided*: the rule has no threshold on the dispersion of the ratings, so consensus is described, not manipulated.
3. **Two LoRA adapters** of Stable Diffusion XL, one per set, with the same settings; BASE is the unmodified model.
4. **Generation.** 48 prompts × 4 seeds × 3 conditions = 576 images; after a quality rule applied to whole triplets, 159
   of the 192 triplets (477 images) form the primary set.
5. **Phase 2.** 158 participants, blind to the condition: 9,413 valid ratings and 4,736 pairwise comparisons (4,010 decisive).
6. **Exploratory probes:** DINOv2 embeddings, 31 image-derived covariates (10 photographic measures, 21 CLIP zero-shot
   attributes), three vision-language models (VLM), and a post-hoc sensitivity with three paired training seeds.

The study concerns **images**: two-dimensional representations of architecture, not built form.

## Where to look

| You want to check… | Go to |
|---|---|
| which file supports a statement of the manuscript | [`PAPER_MAP.md`](PAPER_MAP.md) |
| what was planned, when, and what changed | [`protocol/`](protocol/README.md) |
| what is not preregistered and the other limits, stated by the authors | [`protocol/TRANSPARENCY_NOTES.md`](protocol/TRANSPARENCY_NOTES.md) |
| the human data | [`data/`](data/README.md) |
| the analyses, their code and how to re-run them | [`analysis/`](analysis/README.md) |
| how the adapters were trained and the images generated | [`training/`](training/README.md) |
| the images | `analysis/vlm/images/` (the 576 images as shown to participants), `analysis/replication/images/` |

Every report has a companion CSV with all its numbers. Result files are named after their analysis.

## Results reported in the manuscript

| Result | Value | Source in this repository |
|---|---|---|
| Phase 1: individual judgements are heterogeneous, the aggregate is stable | ICC single rater 0.138; ICC(k) 0.858; correlation of each participant with the others .351 (n = 161) | `human_report/report_paper_participant_level.md` |
| Leave-one-out preference signal between the two sets | +0.915 points | `human_report/report_paper_participant_level.md` |
| **Primary contrast, ratings: AESTHETIC − CONTROL** | +0.104 points (95% CI +0.029 / +0.178; p = .0067), mean of the within-participant differences, n = 158; d_z = +0.22; +0.105 at triplet level | `human_report/report_paper_participant_level.md` |
| Pairwise choices AESTHETIC vs CONTROL | OR 1.090 (0.985–1.206); AESTHETIC chosen in 51.1% of the decisive comparisons (mean over the triplets) | `human_report/report_paper_participant_level.md` |
| BASE − CONTROL and AESTHETIC − BASE | +0.140 and −0.036 points | `human_report/report_paper_participant_level.md` |
| The two training sets differ in embedding space, with large overlap | centroid distance 0.235, standardised 0.26; AUC 0.78 | `embedding_report/report_dino.md`, A |
| The direction CONTROL → AESTHETIC follows the ratings of the 422 photographs in neither set | Spearman ρ +0.39 | `report_dino.md`, A3 |
| BASE → AESTHETIC moves along the source direction, BASE → CONTROL does not | +0.0443 and +0.0014; contrast +0.0429, transfer ratio 18% | `report_dino.md`, B1 |
| The difference is one of direction, not of amount of shift | 96% due to direction | `report_dino.md`, B5 |
| Post-hoc sensitivity: three training seeds, each applied to both corpora | +0.0435, +0.0412, +0.0504 on 192 cells | `replication_report/report_replication_dino.md`, 1 |
| What the source direction contains | photography R² 0.10, framing and scene 0.21, architectural content 0.50, metadata 0.25; share of the gap: 8%, 33%, 57%, all blocks 65% | `decomposition_report/report_direction_decomposition.md`, 1–2 |
| Profile of the 31 covariates, photographs vs generated images | r = 0.71; same sign for 27 of 31 | `report_direction_decomposition.md`, 4 |
| Residual direction | 71% of the length, cosine 0.85, ρ 0.39 → 0.24; +0.0186, +0.0080, +0.0269 by training seed | `report_direction_decomposition.md`, 3 |
| VLM triangulation on the 159 triplets | +0.149 SD (architectural appearance), +0.245 SD (representation quality); near-zero correlation with human ratings | `vlm_report/report_vlm.md`, 1 and 3 |
| The VLM contrasts do not keep a coherent direction with the paired training seeds; the original CONTROL adapter scores lower than other CONTROL adapters on representation quality | none of the five measures named in advance is positive for the three seeds | `vlm_report/report_replication_vlm.md`, 1 and 5 |
| The captions may be a second channel of transfer | the two sets can be told apart from the captions alone (AUC 0.85) | `decomposition_report/report_captions.md` |
| Curator composition (architects vs non-experts; high vs low expertise) | ρ 0.423 and 0.500; gaps +0.893 / +1.130 and +0.978 / +1.133 | computed from `data/export/pretraining_ratings.csv` and `participants.csv` (see `PAPER_MAP.md`) |

**Note on the human estimates.** The manuscript reports participant-level estimates: for each participant the mean
difference AESTHETIC − CONTROL over the images rated, then a one-sample t test on the 158 differences; Bradley–Terry odds
ratios for the decisive pairwise choices. `analysis/paper_human_estimates.py` recomputes them from the export and
`human_report/report_paper_participant_level.md` lists them: they coincide with the values of the manuscript.
`human_report/report_human_votes.md` is a second analysis of the same data with mixed models: close, not identical
estimates (AESTHETIC − CONTROL +0.100, 95% CI +0.034 / +0.166), same reading. Both reports also contain quantities that
the manuscript does not use (equivalence intervals, Davidson model, sample-size projections).

## Analyses in this repository that go beyond the manuscript

These were run in the same project and are included for completeness, whatever their outcome.

| Analysis | Result | Source |
|---|---|---|
| The 31 descriptors against BASE: do the adapters move towards the profile of their training set? | no. AESTHETIC − BASE is unrelated to the training difference (r = −0.01), CONTROL − BASE goes against it (r = −0.43); the two adapters move from BASE in the same direction for 26 of 31 descriptors. The r = 0.71 of the manuscript is a difference between two nearly parallel shifts, not a move of AESTHETIC towards its own photographs | `analysis/figures/feature_base_analysis.md`, figures in `analysis/figures/` |
| Does the percentile of the preference rule matter? | no: the photographs are selected by the mean threshold alone; the selected set is identical for any percentile between 20% and 50%, hence for the preliminary 33% and the final 35% | `human_report/report_selection_rule.md` |
| Determinism control: the AESTHETIC adapter trained twice with identical settings | different weights and different images, no systematic shift along the direction (distance 0.32, against 0.63 when only the training seed changes and 0.68 when only the corpus changes) | `replication_report/report_replication_dino.md`, 5 |
| Corpus effect against training-seed effect along the direction | ratio 4.8 | `report_replication_dino.md`, 2 |
| Details of the VLM evaluation on the paired seeds (subset fixed in advance: 24 prompts × 2 seeds × 6 adapters; contrasts by seed and by model) | between-seed differences as large as the corpus contrasts | `vlm_report/report_replication_vlm.md` |
| Words that differ between the captions of the two sets | more `green`, `sky`, `river` in AESTHETIC; more "eye-level perspective" in CONTROL | `decomposition_report/report_captions.md` |
| Equivalence intervals, Davidson model with ties, mixed models, sample-size projections for the human data | see the reports | `human_report/` |

## Status of the analyses

The study is a **pilot**. The primary endpoint is the human contrast AESTHETIC − CONTROL between the two final adapters,
which were trained as two independent workflows with distinct seeds: it compares two workflows, not the pure effect of
the corpus at a fixed seed. No prospective power calculation was made; a minimum detectable effect is reported a
posteriori. Embedding, covariate and VLM analyses were designed after the human data collection and are exploratory.
The sensitivity with three training seeds applied to both corpora is post hoc; its protocol and criterion were written
before the additional trainings (plan, section 8). It concerns the representational transfer only: **no participant saw
the images of the additional adapters.**

## What is not in this repository

- The **600 photographs**: third-party works under their own licences. `data/export/image_metadata.csv` gives source
  URL, author and licence of each; their embeddings and covariates are included.
- The **platform** used to collect the data (a web application): only the script that ran training and generation is
  included (`training/`), with the exports it produced.
- The **original PNG files** of the generated images: the repository holds the display version shown to the participants
  (JPEG, same size 1216×832) and the 480-px thumbnails used for the embeddings.
- The **weights** of the adapters (their SHA-256 are in `data/export/generated_images.csv`).
- The **version-control history**: this anonymous copy starts from a single commit. The dated history of the protocol
  can be shown to the editors on request.

## Reuse

Material provided for peer review. A licence will be attached to the public release that accompanies the published paper.
