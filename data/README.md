# Data

`export/` is the full export of the study (taken on 2026-10-02, after the replication), anonymised as described in
[`../ANONYMISATION.md`](../ANONYMISATION.md). It keeps the file names of the original export, so the analysis scripts
accept the folder as it is.

## Tables

| File | One row per | Notes |
|---|---|---|
| `participants.csv` | participant (311) | age range, professional role, self-assessed expertise (1–5), country, language, exclusion flag and reason, whether they have a phase-1 session, self-reported prior participation |
| `pretraining_sessions.csv`, `posttraining_sessions.csv` | session | phase, mode, randomisation seed, number of items and of answers |
| `pretraining_ratings.csv` | phase-1 rating (23,012) | photograph, score 1–7, position in the session, response time (ms), quality flags, `excluded_from_analysis` |
| `pretraining_summary.csv` | photograph (600) | number of ratings, mean, median, SD, interval, distribution of the scores |
| `posttraining_ratings.csv` | phase-2 rating (9,932) | generated image, condition, training run, prompt, generation seed, score, response time, exclusion flag |
| `pairwise_trials.csv` | phase-2 comparison (5,130) | conditions shown left and right, choice (`Left`, `Right`, `Tie`), response time, exclusion flag |
| `session_items.csv` | item presented | order of presentation |
| `quality_flags.csv` | data-quality flag | very fast answers and similar, with the decision taken |
| `image_metadata.csv` | photograph | building, architect, year, place, type, style, period, **source URL, author and licence** of the photograph |
| `source_images.csv` | photograph | file size, SHA-256, corpus batch, whether active |
| `captions.csv` | caption version | text, source, whether current and reviewed |
| `aesthetic_dataset.csv`, `control_dataset.csv` | photograph in a dataset version | selection reason, matching, **caption snapshot used for training** |
| `selection_rules.json` | — | the frozen selection rule |
| `training_runs.csv` | training run (17) | corpus, base model and commit, LoRA rank and alpha, learning rate, steps, resolution, **training seed**, configuration |
| `model_conditions.csv` | condition | BASE, CONTROL, AESTHETIC and the test conditions of the replication runs |
| `prompt_sets.csv`, `generation_plan.csv` | prompt; generation plan | every prompt set and plan, including earlier test ones |
| `generated_images.csv` | generated image (1,536) | condition, run, prompt, seed, plan, `opaque_id` (the file name of the image), SHA-256 of the original file, full generation parameters, defects marked in the screening, `is_active` |
| `consent_versions.csv` | consent version | the text shown to participants |
| `experiment.json`, `protocol.json` | — | experiment description, frozen configuration, history of the phases and of the protocol changes |
| `results_summary.csv` | — | descriptive results as computed by the platform |

## Keys between tables

- `participant_id` links participants, sessions, ratings, comparisons and flags.
- `image_code` (`ARCH_0001`…) identifies a photograph.
- `opaque_id` identifies a generated image; image files are named `<opaque_id without dashes>.jpg`.
- `prompt_code` + `seed` identify a prompt/seed cell (a "triplet" in the original experiment: BASE, CONTROL, AESTHETIC).
- `training_run` identifies the adapter: `RUN-AESTHETIC-4` and `RUN-CONTROL-4` are the adapters of the experiment;
  `REP-…` are the replication adapters (the suffix is the training seed; `REP-AES-S1254-R` repeats `RUN-AESTHETIC-4` exactly).
- `generation_plan_id` 22 and 23 are the plans of the experiment; 24 and 25 those of the replication.

## What "excluded" means

`excluded_from_analysis` is true for an answer or a participant removed by a rule or by a decision recorded in
`quality_flags.csv` / `participants.csv`. `is_active` false on a generated image means that its triplet was excluded by
the defect screening and never shown. The analyses apply these flags; their sensitivity analyses do not.

## Numbers to expect

600 photographs · 89 + 89 in the two training sets · 576 generated images of the experiment (477 shown) · 960 images
of the replication · 311 participant records over the two phases.
