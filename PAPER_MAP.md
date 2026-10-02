# From the manuscript to the files

Paths are relative to the root of the repository. "Export" is `data/export/`.

## Section 3 — Corpus and method

| Manuscript | Statement | File |
|---|---|---|
| 3.1, Tab. 1 | 600 photographs / 600 buildings; 76 countries; periods; 78 industrial; 264 in Europe; most frequent author in 7 cases | export `image_metadata.csv`, `source_images.csv` |
| 3.2, Tab. 1 | 164 participants, 22,712 valid ratings, 12–64 ratings per image (mean 37.9); roles (63 non-experts, 56 architects, 8 architecture students, 8 engineers, 29 in other categories aggregated in the anonymous copy) and expertise of the phase-1 sample | export `pretraining_ratings.csv` (rows with `excluded_from_analysis` false), `participants.csv` |
| 3.2 | preference rule; preliminary 33% and final 35%; 89 photographs; CONTROL matched on building type and style, seed 345145722; v2 = same membership, harmonised captions | export `selection_rules.json`, `aesthetic_dataset.csv`, `control_dataset.csv`; `protocol/timeline_from_audit_log.csv`; effect of the percentile: `analysis/human_report/report_selection_rule.md` |
| 3.2, 4.1 | curator composition: architects vs non-experts ρ = .423 (n = 599), gaps +0.893 / +1.130; expertise 4–5 vs 1–2 ρ = .500 (n = 600), gaps +0.978 / +1.133 | mean valid rating per photograph within each group, from export `pretraining_ratings.csv` and `participants.csv`; Spearman correlation across the photographs rated by both groups; gap = mean over the AESTHETIC photographs minus mean over the CONTROL ones |
| 3.3, Tab. 2 | base model and revision; LoRA settings; training seeds 1254 / 9865 of the two final adapters; seeds 1254, 9865, 42160 of the sensitivity | export `training_runs.csv`, `model_conditions.csv`; `training/README.md`; `analysis/replication/design.json` |
| 3.4 | 48 prompts × 4 seeds × 3 conditions; 1216 × 832, DPM++ 2M Karras, guidance 5, 40 steps | export `generation_plan.csv`, `generated_images.csv` (column `generation_parameters_json`); `protocol/generation_prompts.csv` |
| 3.4, Tab. 2 | screening: 33 triplets excluded; defects 21 BASE, 14 AESTHETIC, 13 CONTROL; one researcher; blind interface | export `generated_images.csv` (`review_defects`, `excluded_by_review_rule`, `exclusion_reason`); `protocol/timeline_from_audit_log.csv` (action `manual_correction`, reason "blind review"); `protocol/TRANSPARENCY_NOTES.md` |
| 3.5, Tab. 1 | 158 participants, 9,413 valid ratings, 15–22 ratings per image (mean 19.7); 4,736 comparisons, 726 ties, 4,010 decisive; 32 participants with a valid phase-1 session | export `posttraining_ratings.csv`, `pairwise_trials.csv`, `pretraining_ratings.csv` |
| 3.6 | primary estimate: mean of the within-participant differences, one-sample t test on 158 differences; triplet-level summary; Bradley–Terry odds ratio; minimum detectable effect (about 0.11 points, OR 1.16) | `analysis/paper_human_estimates.py`; `analysis/human_report/report_paper_participant_level.md` |
| 3.7 | DINOv2 embeddings, source direction, transfer ratio | `analysis/image_scripts/embeddings.py`, `analysis/embedding_analysis.py`, `analysis/baseline/direction_dinov2.npz`, `analysis/metrics/embeddings_dinov2_vitb14.npz` |
| 3.7 | the 31 covariates: formulas of the ten photographic descriptors, CLIP contrasts of the 21 attributes, the separate attribute "elegant" | `analysis/image_scripts/covariates.py` (complete descriptions and deterministic code); values in `analysis/metrics/covariates_images.csv` |
| 3.7 | VLM rubric with nine items; blind to condition, prompt and seed | `protocol/vlm_rubric_v1.txt`; `analysis/vlm/` |

## Section 4 — Results

| Manuscript | Statement | File |
|---|---|---|
| 4.1 | ICC single rater 0.138, ICC(k) 0.858; correlation of each participant with the others .351 (n = 161: three of the 164 participants have fewer than 10 valid ratings); leave-one-out signal +0.915; means 5.30 and 4.22 | `analysis/human_report/report_paper_participant_level.md`; `analysis/embedding_report/source_affinity_dino.csv` (mean rating per photograph and set) |
| 4.2, Tab. 3 | AESTHETIC − CONTROL at participant and triplet level, d_z, BASE − CONTROL, AESTHETIC − BASE, Bradley–Terry odds ratios, 51.1% as mean of the triplet shares, effect in SD of the scenes, ICC of phase 2, interaction with the participation in phase 1 | `analysis/human_report/report_paper_participant_level.md` (script `analysis/paper_human_estimates.py`): every value coincides with the manuscript. Sensitivity to filters and fast answers: `analysis/human_report/report_human_votes.md`, section 13 (mixed models) |
| Fig. 1 | three triplets: minimum, median, maximum of the representation transfer score | `analysis/embedding_report/triplet_transfer_dino.csv` (rows with `all_three_shown_to_participants` true, sorted by `representation_transfer_score`): prompt A07 seed 6127, C14 seed 25478, A08 seed 7. Images: `analysis/vlm/images/<opaque_id>.jpg`, ids in export `generated_images.csv` |
| 4.3 | centroid distance 0.235 vs 0.138; standardised 0.26; AUC 0.78; r = .64; ρ = .39 | `analysis/embedding_report/report_dino.md`, A1–A3 |
| 4.3 | BASE → AESTHETIC +0.0443; BASE → CONTROL +0.0014; contrast +0.0429; transfer ratio 18%; 62% of the triplets positive; direction, not amount | `report_dino.md`, B1 and B5 |
| 4.3 | three training seeds applied to both corpora: +0.0435 (+0.0272 / +0.0612), +0.0412 (+0.0255 / +0.0578), +0.0504 (+0.0354 / +0.0674) on 192 cells | `analysis/replication_report/report_replication_dino.md`, section 1 |
| 4.4 | R² of each block; share of the gap accounted for; "elegant vs ugly" | `analysis/decomposition_report/report_direction_decomposition.md`, sections 1–2 |
| 4.4, Fig. 2 | profile of the 31 covariates, r = .71, same sign for 27 of 31 | `report_direction_decomposition.md`, section 4; data of the figure in `covariate_profile_direction_decomposition.csv`; the figure redrawn, with the same contrast taken from BASE for each adapter, in `analysis/figures/` (`feature_base_analysis.md`) |
| 4.4 | residual direction: 0.235 → 0.166, 71%, cosine .85, ρ .39 → .24; +0.0186, +0.0080, +0.0269 | `report_direction_decomposition.md`, section 3 |
| 4.5 | VLM: +0.149 and +0.245 SD on the 159 triplets; 192 triplets; correlation with human ratings; test–retest and agreement between models | `analysis/vlm_report/report_vlm.md`, sections 1, 3, 4; `analysis/vlm_report/full_instrument_report.md` |
| 4.5, 5.5 | with paired training seeds the VLM contrasts do not keep a coherent direction | `analysis/vlm_report/report_replication_vlm.md`, section 1 |
| 5.2 | the original CONTROL adapter scores lower than other CONTROL adapters on representation quality | `analysis/vlm_report/report_replication_vlm.md`, section 5 |
| 5.5 | captions as a possible second channel of transfer | `analysis/decomposition_report/report_captions.md` |

## Declarations

| Manuscript | File |
|---|---|
| participation and ethics | `protocol/consent_text.md`; `protocol/TRANSPARENCY_NOTES.md`, section 6 |
| data availability | this repository; `ANONYMISATION.md` |

## Notes

- **Roles in the anonymised export.** Professional roles declared by fewer than ten participants overall (creative
  professionals, designers, construction professionals) are merged into `Other` in `participants.csv` to reduce the risk
  of re-identification (see `ANONYMISATION.md`); the manuscript reports the roles in the same aggregated form.
- **Analyses beyond the manuscript** (effect of the percentile, determinism control, corpus against seed, details of the
  VLM evaluation on the paired seeds, equivalence intervals) are listed in the main `README.md`.
