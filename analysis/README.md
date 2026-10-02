# Analysis: code, intermediate files and results

Everything here runs on the anonymised export in `../data/export` and reproduces the reports of this folder.
The plan that these scripts implement is `../protocol/ANALYSIS_PLAN.md`.

## Results (already computed)

| Folder | Report | Numbers | Plan |
|---|---|---|---|
| `human_report/` | `report_paper_participant_level.md` — the participant-level estimates reported in the manuscript | `results_paper_participant_level.csv` | sections 1–5 |
| `human_report/` | `report_human_votes.md` — the same data analysed with mixed models | `results_human_votes.csv` | sections 1–5 |
| `human_report/` | `report_selection_rule.md` — the preference rule applied with different percentiles | `results_selection_rule.csv` | — |
| `embedding_report/` | `report_dino.md` | `results_dino.csv`, `source_affinity_dino.csv` (one row per photograph), `triplet_transfer_dino.csv` (one row per prompt/seed) | 6 |
| `vlm_report/` | `report_vlm.md`; instrument reports computed without the key: `pilot_instrument_report.md`, `full_instrument_report.md` | `results_vlm.csv`, `triplet_vlm.csv`, `vlm_scores_unblinded.csv` | 7 |
| `replication_report/` | `report_replication_dino.md` | `results_replication_dino.csv`, `cells_replication_dino.csv` | 8.5 |
| `vlm_report/` | `report_replication_vlm.md`, `replication_instrument_report.md` | `results_replication_vlm.csv`, `cells_replication_vlm.csv`, `vlm_scores_unblinded_replication.csv` | 8.6 |
| `decomposition_report/` | `report_direction_decomposition.md`, `report_captions.md` | `results_direction_decomposition.csv`, `covariate_profile_direction_decomposition.csv`, `results_captions.csv`, `words_captions.csv` | 9 |
| `figures/` | `feature_base_analysis.md` — the 31 descriptors against BASE, and the shifts in embedding space; figures `fig_source_vs_aesthetic_minus_control.png` (the profile of the manuscript, Fig. 2, redrawn), `fig_source_vs_aesthetic_from_base.png`, `fig_source_vs_control_from_base.png`, the three side by side in `fig_source_contrasts_three_panels.png`, `fig_embedding_shifts.png` | `feature_base_contrasts.csv` | 9 (added) |

## Re-run

```
pip install -r requirements.txt
cd analysis
python paper_human_estimates.py   ../data/export
python selection_rule_check.py    ../data/export
python arch300_analysis.py        ../data/export --out human_report
python embedding_analysis.py      ../data/export metrics/embeddings_dinov2_vitb14.npz --out embedding_report
python vlm_analysis.py            ../data/export
python replication_analysis.py    ../data/export metrics/embeddings_replication_dinov2_vitb14.npz
python replication_vlm_analysis.py
python direction_decomposition.py ../data/export
python caption_analysis.py        ../data/export
python figures_transfer.py        ../data/export
```

All seeds are fixed in the scripts. The human analysis and the embedding analysis take ten to fifteen minutes each
(bootstrap and permutations); the others a few minutes. None of them needs a GPU or network access.

## Intermediate files

| File | Content | Produced by |
|---|---|---|
| `metrics/embeddings_dinov2_vitb14.npz` | DINOv2 ViT-B/14 embeddings (CLS token and mean of the patch tokens) of the 600 photographs and of the 576 generated images | `image_scripts/embeddings.py` |
| `metrics/embeddings_replication_dinov2_vitb14.npz` | the same for the 960 replication images | `image_scripts/embeddings.py` |
| `metrics/covariates_images.csv` | ten photographic measures and 22 CLIP ViT-L/14 zero-shot attributes for every photograph and generated image | `image_scripts/covariates.py` |
| `baseline/direction_dinov2.npz` | the source-set contrast direction, frozen before the replication | `baseline/freeze.py` |
| `baseline/BASELINE_MANIFEST.csv`, `baseline/RENAMED_FILES.csv` | SHA-256 of every file of the baseline when it was frozen (the hashes of the export and of the photographs refer to the original, non-anonymised files) | `baseline/freeze.py` |
| `metrics/image_metrics.csv`, `metrics/content_metrics.csv`, `metrics/triplet_key.csv` | earlier measures on the 576 generated images, computed on three-image sheets (superseded by `covariates_images.csv`) | `image_scripts/image_metrics.py`, `content.py`, `triplets.py` |
| `replication/design.json`, `replication/vlm_subset.csv` | training seed C, order of the trainings, and the VLM subset, drawn before any replication training | `replication/draw.py` |

To recompute the embeddings and the covariates of the **generated** images: `pip install torch transformers pillow
open_clip_torch`, then run the two scripts on `images/generated` and `replication/images/generated`. Those of the
photographs cannot be recomputed from this repository (the photographs are not redistributed; their URLs are in
`../data/export/image_metadata.csv`).

## Images

| Folder | Content |
|---|---|
| `vlm/images/` | the 576 generated images of the experiment, in the version shown to the participants and to the VLM (JPEG, 1216×832) |
| `images/generated/` | their 480-px thumbnails, used for the embeddings |
| `replication/images/generated/` | the 960 replication images, 480-px thumbnails |
| `replication/images/weblarge/` | the 192 replication images of the VLM subset, 1216×832 |

Files are named by `opaque_id`; `../data/export/generated_images.csv` gives condition, training run, prompt and seed.

## The blind VLM evaluation (`vlm/`)

| File | Role |
|---|---|
| `rubric_v1.txt` | the text sent with each image |
| `prepare.py`, `prepare_replication.py` | build the blind bundle: byte-identical copies under random names, a blind manifest, the list of repeats, and the key |
| `run.py`, `providers.py`, `check.py` | send one image per request to the three models; resumable; every attempt logged |
| `pilot_report.py` | instrument report (completeness, use of the scale, test–retest, agreement between models) computed **without** the key |
| `manifests/` | blind manifests and lists of repeats of the three runs (pilot, full, replication) |
| `outputs/<run>/<provider>.jsonl` | every attempt: model returned, time, outcome, token usage, parsed answer |
| `outputs/<run>/vlm_scores_long.csv` | the blind long table: anonymous id, model, nine scores |
| `key_DO_NOT_USE_<run>.csv` | the keys (anonymous id → image, condition). They were kept outside the process until unblinding and are published here so that the join can be verified |
| `config.json` | paths and model identifiers, without API keys. To send new requests, add your own keys |

The analysis scripts read the long table and the key; they do not call any API.

## Notes on reproducibility

- Bootstrap and permutation results are deterministic given the seeds in the scripts.
- The replication, VLM-replication and decomposition scripts were re-run on this anonymised copy: every estimate is
  identical to the reported one. `human_report/` was produced from this anonymised copy.
- `arch300_analysis.py` uses the participant timestamps only to order events, which the anonymisation preserves within
  each participant.
