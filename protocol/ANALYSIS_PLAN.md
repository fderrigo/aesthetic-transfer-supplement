# ARCH300 — Analysis plan of the pilot study

**Investigating Aesthetic Transfer in Generative Models for Architectural Design:
A Pilot Study on Consensus-Based Fine-Tuning and Human Evaluation**

Script: `arch300_analysis.py` (runs on the "Full experiment export" ZIP or its extracted folder). The platform page
*Pilot study analysis* shows the same structure with participant-level methods; the script adds the mixed models.

> **Change of framing (2026-09-30).** A first version of this plan, written on 2026-09-30, treated the study as a
> confirmatory test of one hypothesis (AESTHETIC better than CONTROL), with secondary tests corrected by Holm and a
> hold-out set of participants registered from that date. On the same day the study was reframed as a **pilot study**
> with the title above. From then on all analyses are estimates with uncertainty: no result is used to "confirm" or
> "refute", p values are reported as uncorrected indications, and the hold-out split is dropped. The previous version is
> kept in the project history.

## Design recap

- **Phase 1 — consensus.** Participants with and without architectural training rate photographs of 300 buildings
  (1–7). From the ratings two training sets of equal size are selected: **AESTHETIC** (the photographs preferred by the
  consensus: top 35% of the mean ratings and mean ≥ 5, at least 10 ratings) and **CONTROL** (a random sample matched on
  building type and style, drawn from the photographs **not** selected for AESTHETIC). CONTROL is therefore "the rest of
  the corpus", slightly below the corpus average (mean rating 4.21 against 4.34), not a random sample of the whole corpus:
  the contrast AESTHETIC − CONTROL is "the preferred photographs against the others".
- **Fine-tuning.** Three image-generation conditions from the same SDXL base model: **BASE** (no fine-tuning),
  **CONTROL** (LoRA on the CONTROL set), **AESTHETIC** (LoRA on the AESTHETIC set). Same prompts, seeds and settings; a
  *triplet* = the three images of one prompt/seed. A blind review (rule v1) excluded triplets with evident defects.
- **Phase 2 — human evaluation.** Blind 1–7 ratings and pairwise choices between two conditions of the same triplet
  (sides at random, "no preference" allowed); task order counterbalanced.

## Research questions

1. **Consensus.** Is there a shared aesthetic judgement of architectural photographs, and how large is the aesthetic
   signal that the consensus put into the AESTHETIC set?
2. **Aesthetic transfer.** How much of that judgement is carried by the fine-tuning into the generated images:
   AESTHETIC against CONTROL (effect of the aesthetic selection) and against BASE (effect of fine-tuning as such), in
   ratings and in pairwise choices? What is transferred (building types, image characteristics such as colour,
   framing and photographic style)?
3. **Personal transfer.** Do the raters who built the consensus recognise it: are they more in favour of AESTHETIC, and
   does each person's phase-1 preference predict their phase-2 preference?
4. **Human evaluation.** How reliable are the phase-2 judgements, do ratings and choices agree, and how much do
   position, task order and fatigue matter?
5. **Towards a full study.** What effect can a study of this size see, and how many participants would a full study need?

## Decisions

- 95% intervals throughout; p values reported without correction and read as indications (many analyses are run).
- Relevance threshold: **0.30 points** on the 1–7 scale (odds ratio **0.80–1.25** for pairwise choices), used for the
  equivalence check (TOST, 90% intervals) and as the target of the sample-size estimate. **The threshold was set on
  2026-09-30, after the data of the first 22 phase-2 participants had been seen**: it is a declared convention, not a
  pre-registered margin.
- Exclusions are the documented researcher decisions exported with the data (participants, sessions, single answers);
  every page, the export and the script apply the same three levels. The main estimates are re-run without exclusions
  and with other reasonable data choices (sensitivity). Excluding a participant removes their answers in **both** phases.
- **Open point, to be decided and written here before the final analysis: the rule for very fast answers.** Answers
  faster than the protocol threshold are only flagged; nothing is excluded without a decision. In the export of
  2026-09-28 no exclusion had been decided yet, so the consensus used for the AESTHETIC set included 327 phase-1 answers
  given in less than 0.5 s (1.4% of 23,012). The sensitivity analysis without answers < 500 ms is reported in any case.
- "Linked to phase 1" = the participant gave at least one phase-1 rating. AESTHETIC and CONTROL photographs = the
  datasets the fine-tuned models were actually trained on.
- Pairwise models use participant-robust errors with t tests on G − 1 degrees of freedom (G = participants).
- Repeated measures: mixed models with crossed random effects (script) and, on the platform, one value per participant
  and per triplet with bootstrap over participants and participant-robust errors.

## Analyses

### 1. Consensus (phase 1)
- Agreement between raters: ICC(1), ICC(k) and split-half reliability of the photograph means (participant-centred).
- Agreement of each rater with the others: correlation with the leave-one-out photograph means.
- Aesthetic signal: per rater, mean rating of the AESTHETIC photographs minus the CONTROL photographs. Measured in two
  ways: with the same ratings used for the selection (inflated by regression to the mean) and **leave-one-rater-out**
  (the best photographs are re-selected without the rater and scored with the rater's own votes): the second is the
  reference value.

### 2. Aesthetic transfer (phase 2)
- Ratings: linear mixed model `score ~ condition` (CONTROL = reference; random intercepts for participant, prompt and
  scene, random condition slopes by participant when they converge); ordinal model; per-participant and per-triplet
  differences, Wilcoxon, bootstrap, standardised effect size.
- Pairwise choices: Bradley–Terry model with a position term (participant-robust errors; mixed version with random
  participant and scene); per-participant and per-triplet share of AESTHETIC choices.
- All three contrasts: AESTHETIC − CONTROL, BASE − CONTROL, AESTHETIC − BASE.
- Is the transfer negligible? Equivalence (TOST) with the threshold above and Bayes factors BF01.
- **Transfer ratio**: phase-2 difference AESTHETIC − CONTROL divided by the leave-one-rater-out aesthetic signal
  (0% = nothing transferred, 100% = the whole difference between the training photographs).
  The two phases are not on the same footing (real photographs of different buildings against generated images that
  share prompt and seed), so the difference is also expressed **on the scale of phase 2**: in standard deviations of
  the generated scene means.
- **Coherence of the two tasks**: the pairwise preference expected from the rating difference
  (Φ(difference / (√2 · SD of a rating around the image mean))) is compared with the observed one.
- "No preference" answers: dropped in the Bradley–Terry model; kept in two checks — as half a choice per participant, and
  in the **Davidson model** (Bradley–Terry with "no preference" as a third outcome, position term, participant-robust errors).

### 2b. What is transferred
- AESTHETIC − CONTROL by building type (bootstrap intervals).
- Image characteristics as covariates (colour saturation, colourfulness, brightness, sharpness, whole-building framing,
  CLIP "elegance", prompt adherence): how much of the difference remains at equal image characteristics
  (requires the image-metric files in `metrics/`).

### 3. Personal transfer
- AESTHETIC − CONTROL (ratings and choices) for participants linked to phase 1, declared phase-1 participants and
  phase-2-only participants, and the difference linked − phase 2 only.
- Individual alignment: each linked participant's phase-1 preference for the AESTHETIC photographs against their
  phase-2 preference for AESTHETIC images (Spearman, bootstrap interval; interaction in the mixed model).
- Agreement with the consensus in phase 1 against the phase-2 preference.

### 4. Human evaluation
- Reliability of the phase-2 ratings (ICC, split-half).
- Agreement of the two tasks: image mean rating against share of pairwise wins; per participant and per triplet.
- Position bias, "no preference" answers, task order and fatigue (position in the session; first vs second half).
- Sensitivity of the transfer estimates: without exclusions; without answers < 500 ms; without the 10% fastest
  participants; completed participants only.

### 5. Towards a full study
- Minimum detectable effect with the current data (80% power, α = .05; t quantiles), ratings and pairwise choices.
- Participants needed for a paired comparison with 80% power to detect 0.30 points, the observed rating difference and
  the observed pairwise preference, assuming the variability observed in the pilot.

### 6. Computational analysis of the images (DINOv2 embeddings) — exploratory

Added on 2026-10-01, after the phase-2 votes had been seen: exploratory. Scripts: `image_scripts/embeddings.py`
(embeddings) and `embedding_analysis.py` (analysis and report in `embedding_report/`).

Fixed before running: model **DINOv2 ViT-B/14** (`facebook/dinov2-base`); every image seen whole (480-px thumbnail
resized to a 224-px short side, no crop); **CLS token, L2-normalised** as the embedding (mean of the patch tokens as a
sensitivity check); Euclidean distances between unit vectors; 10,000 permutations, 2,000 bootstrap samples, fixed seed.

**Part A — what distinguishes the photographs selected by the consensus** (600 corpus photographs; AESTHETIC and
CONTROL = the sets the models were trained on):
- centroids and distances: AESTHETIC – CONTROL, AESTHETIC – rest of the corpus (bootstrap intervals);
- dispersion of each set around its centroid, and the difference AESTHETIC − CONTROL;
- permutation tests: AESTHETIC against random subsets of the corpus of the same size; AESTHETIC/CONTROL labels
  exchanged freely and within building type (CONTROL is matched on building type and style);
- usability of the difference: cross-validated separability of the two sets (AUC) and prediction of the phase-1 mean
  rating from the embedding (ridge regression, nested cross-validation, permutation p);
- the **aesthetic direction** = unit vector from the CONTROL centroid to the AESTHETIC centroid, checked on the
  photographs that are in neither set.

**Part B — shifts of the generated images** (complete triplets of active images):
- for each prompt/seed, Δ_A = AESTHETIC − BASE and Δ_C = CONTROL − BASE; their projection on the aesthetic direction;
  the test is the **difference** Δ_A − Δ_C (any fine-tuning moves the images, so the AESTHETIC shift alone proves
  nothing); intervals by bootstrap over prompts, p by sign-flip permutation of the prompt means;
- embedding transfer ratio = that difference divided by the gap between the two training sets along the direction;
- specificity: whether each model moves towards its own training set more than towards the other;
- link with the human votes: per triplet, alignment difference against the rating difference AESTHETIC − CONTROL and
  the share of AESTHETIC choices; within triplet, position along the direction against the rating; the rating predictor
  learned on the photographs applied to the generated images.

Limits to state with the results: the direction is learned on real photographs and applied to generated images (domain
shift); the embedding mixes architecture, composition and photographic rendering and cannot tell them apart; one
fine-tuning run per condition; many quantities, uncorrected p values. No interpretation of single embedding dimensions.

### 7. Evaluation of the generated images by vision-language models (VLM) — exploratory

**Status: protocol written before any VLM request; nothing has been run.** Added after the end of the data collection and
after sections 1–6 were computed; it is therefore exploratory. Changes to this section are allowed only after the pilot
(7.5) and are recorded here with their date; the git history of this file is the record.

**Question.** Section 6 says that the AESTHETIC model moves the images towards the selected photographs, not what changes.
Here: is the difference between AESTHETIC and CONTROL images seen by independent observers in the *architectural
appearance* of what is depicted, in the *quality of the representation* (light, materials, photographic rendering), or in
both? The scores are not measures of beauty or of architectural quality.

**Check of 2026-10-01 (test image, not from the study):** the three keys are accepted, the three identifiers below are
listed and returned by the providers, image input and structured answer work. Defaults observed: OpenAI and Google use
reasoning tokens, Anthropic does not (no setting is changed).

**7.1 Models.** Three systems from three providers, so that a result does not depend on one family:
`gpt-5.6-sol` (OpenAI), `claude-opus-5-5` (Anthropic), `gemini-3.8-flash` (Google). Each identifier is checked with a test
request before the pilot; the identifier returned by the provider is saved with every answer. If one identifier is not
available it is replaced by the closest fixed version of the same provider, and the replacement is recorded here.

**7.2 Images.** The 576 generated images in the version shown to the participants (`/research/image/{id}`: JPEG, quality
82, 1216×832, no metadata), downloaded with a staff account and passed byte for byte, without crop, resize, recompression,
frame or collage. The 600 photographs are not evaluated. Copies are named `VLM_000001.jpg`… in random order; the table
`anon_id → image, condition, prompt, seed, active` (the key) stays outside the process, is never sent, is not in the
repository, and is not opened by the analysis before 7.7.

**7.3 Request.** One image, one new request, one structured answer. No conversation history, no tools, no web search,
no file name, no generation prompt, no mention of the study, of generated images or of conditions. Same text for the three
providers (`vlm/rubric_v1.txt`, version `VLM_RUBRIC_V1`); only the syntax of the request differs. Sampling and reasoning
settings: the defaults of each model, recorded (they are not forced to be equal; stability is measured by the retest).
Order of the requests (amended on 2026-10-01, before the first evaluation request, because the credit available for each
provider is limited): triplets in random order, the three images of a triplet one after the other in random order; in the
full run first the triplets shown to the participants, then the repeats, then the triplets with an image not shown. If a
provider stops for lack of credit, what is left is made of complete triplets, primary set first, and the run is resumed
later from where it stopped. Requests are independent, so the model cannot see the order; anonymous names are random. Saved for each request: raw answer, model identifier returned,
time, tokens, duration, errors. A refusal or an invalid answer is retried once, then recorded as missing; missing values
are never filled in.

**7.4 Measures.** Nine scores from 1 to 7, exactly as in `vlm/rubric_v1.txt`:

| block | fields |
|---|---|
| architectural appearance | `spatial_geometric_coherence`, `proportional_coherence`, `apparent_constructability`, `component_coherence` |
| image / representation quality | `lighting`, `material_rendering`, `photographic_realism`, `photographic_composition`, `overall_visual_quality` |

No overall architectural quality score is asked. The two composites are computed by us, not by the model:
`architectural_appearance` = mean of the first four, `representation_quality` = mean of the last five.

**7.5 Pilot.** 15 complete triplets (45 images) drawn from the 159 active triplets: 15 different prompts, the four seeds
balanced, reproducible draw (seed 20261002) that reads only prompt, seed, condition and active flag — no rating, no
pairwise result, no embedding score, no manual choice. 9 of the 45 images (20%) are sent a second time under another
identifier (test–retest). 54 evaluations × 3 models = 162 requests. In the pilot only, two short free-text fields
(`architectural_note`, `representation_note`) are added to check that the definitions are understood; pilot scores are
therefore not pooled with the full run and the 45 images are evaluated again.
The pilot is read **without opening the key** and does not say which condition is better. Checks: all nine fields returned
by the three models; the notes show that constructability is not read as "a beautiful building"; the scale is used (an
item with almost all answers at 6–7 is not informative and is corrected); test–retest agreement; agreement between the
three models. No minimum ICC is fixed in advance: the distributions are looked at first. If the rubric is changed it
becomes `VLM_RUBRIC_V2` and the pilot is repeated on a new draw.

**Result of the pilot (2026-10-01, read without the key; `vlm_report/pilot_instrument_report.md`).** 162 of 162
evaluations valid. The notes describe geometry, junctions, supports and artifacts; words of aesthetic appreciation appear
in 0–2% of the architectural notes: the definitions are understood. Use of the scale: Anthropic uses 3–6 (no ceiling);
OpenAI and Google give 6–7 to most images (80–98% of the scores for several items), so their items discriminate little
and the composites carry most of the information. Test–retest of the composites (9 images): ICC 0.85–0.90 for the
architectural composite, 0.55–0.75 for the representation composite. Agreement between the three models: moderate for the
architectural composite (Spearman 0.35–0.45; ICC 0.47 on standardised scores), highest for `apparent_constructability`
(0.46–0.61), close to zero for the representation composite (−0.01 to +0.18) and for `lighting` and
`overall_visual_quality`. The levels differ strongly between models (architectural composite: 4.1, 5.9, 6.3).
*Decisions taken before the full run and before unblinding:* (a) the rubric is **not changed** (`VLM_RUBRIC_V1` stays as
defined by the investigators); (b) because the levels and spreads differ, the "mean of the three models" of 7.7 is the
mean of the scores **standardised within model** (z scores over the images of the primary set), and the results for each
model on its own scale are reported next to it; (c) since the three models do not agree on the representation block,
results on `representation_quality` are read model by model, and a statement about that block is made only where the
three models point the same way; (d) the weak discrimination of two models and the low agreement on the representation
block are limits of the instrument and are reported as such, whatever the result.

**7.6 Full run.** 576 images × 3 models = 1,728 evaluations, plus a retest of about 20% (about 115 images × 3 = 345):
about 2,073 requests, without free text. Output: one long table with `anon_image_id, model_provider, model_id,
rubric_version, run`, the nine scores and `timestamp` — no condition, prompt, seed or human rating. The raw outputs are
committed **before** the key is opened.

**State of the full run (2026-10-01, before unblinding).** OpenAI 691 of 691 and Google 691 of 691 evaluations valid.
Anthropic stopped at 471 of 691 because the spending limit of the account was reached: it evaluated the first 471 images
of the primary set in the fixed order (157 complete triplets of 159), no repeat and no image of the triplets not shown.
*Decision recorded before unblinding:* the key is opened now. For Anthropic the primary analysis uses its 157 complete
triplets; test–retest and the sensitivity on all the images are available for OpenAI and Google only; "three models"
uses the images evaluated by all three. The remaining Anthropic requests follow the same manifest, are made by the same
script without any change, and are added when the limit is raised; the model never sees the key, so the blinding of the
evaluations is not affected. The report states which version it is.

**Update of 2026-10-01, after unblinding.** The Anthropic requests were resumed with the same script and manifest when
the limit was raised, and stopped again at 638 of 691 for the same reason: the primary set is now complete (159 triplets),
the repeats are complete, 15 of the 33 triplets not shown are done. The results of the primary analysis did not change in
any conclusion (differences in the third decimal). 53 requests remain.
Resumed once more the same day: 683 of 691 (the spending limit was reached again); 8 requests remain, all in the triplets
not shown to the participants.
**Run complete (2026-10-01):** after the limit was raised the last 8 requests were made: 691 of 691 valid evaluations for
each of the three models (2,073 in all), same script, same manifest. The reports are those of the complete run.

**7.7 Unblinding and analysis.** After the commit of 7.6 the outputs are joined with the key and with condition, prompt,
seed, triplet, human rating (centred on each participant's mean), pairwise share, representation transfer score (section
6) and active flag.
* *Primary set:* the 477 images seen by the participants (159 triplets). *Sensitivity:* all 576 (192 triplets).
* *Reliability:* test–retest (ICC and mean absolute difference per item and model); agreement between the three models
  (ICC of the composites and of the items).
* *Main comparison:* AESTHETIC − CONTROL within triplet, for the two composites, averaged over the three models and for
  each model separately. Intervals: bootstrap over prompts; p values: sign-flip permutation of the prompt means (as in
  section 6). The two composites are two tests (Holm).
* *Which block:* difference between the two composites in the size of AESTHETIC − CONTROL (standardised within model).
* *Secondary:* the nine items one by one (Holm over nine); BASE − CONTROL and AESTHETIC − BASE; agreement with the human
  votes (per image within triplet, and per triplet for the AESTHETIC − CONTROL differences: rating, pairwise share) and
  with the representation transfer score.
* Everything not listed here is reported as additional exploration.

**7.8 Limits stated in advance.** The models were trained on images from the web and have their own taste: a higher
score is an observation of the model, not a fact about the architecture. The models may recognise generated images and
score them differently from photographs; since all the evaluated images are generated and the comparison is within
triplet, this affects the level, not the comparison. Associations are not causes. The rubric measures coherence and
quality, it does not describe *what* is in the image (light, viewpoint, materials, vegetation…): a separate descriptive
block, asked in a separate request, is a possible extension and is not part of this protocol until it is written here.

**7.9 Files.** `vlm/config.template.json` (to be copied to `vlm/config.json`, which holds the keys and is never
committed), `vlm/rubric_v1.txt`; scripts to be written: `vlm/prepare.py` (download, anonymous copies, draw, key),
`vlm/run.py` (requests), `vlm_analysis.py` (7.7), report in `vlm_report/`. Images, bundle and key are excluded from the
repository.

### 8. Replication across training seeds ("Phase 3") — confirmatory for the embedding result

**Status: protocol written and committed on 2026-10-01, before any replication training.** No new human data are
collected: the votes of the original experiment remain the only human outcome.

**Why.** In the original experiment each condition has one LoRA, and the two were trained with different training seeds
(AESTHETIC: 1254; CONTROL: 9865). The AESTHETIC − CONTROL contrast therefore mixes the effect of the training corpus with
the randomness of the single training run. **Question:** does the contrast observed along the source-set direction
depend on the training corpus, or can a substantial part of it be explained by the training seed?

**8.1 Baseline frozen.** Git tag `baseline-v1`. `baseline/BASELINE_MANIFEST.csv` lists size and SHA-256 of every file of
the baseline (analysis folder, worker script, and the local files that are not in the repository: export, images, VLM
key). `baseline/direction_dinov2.npz` holds the source-set contrast direction (unit vector from the CONTROL centroid to
the AESTHETIC centroid of the training photographs, CLS and patch-mean versions), its length and the centroids; it
reproduces the reported result exactly (+0.0429, 159 triplets) and **is not recomputed**: every replication result uses
this file. No later analysis changes the baseline files.

**8.2 Design.** Three training seeds, each used once with each corpus (2 corpora × 3 seeds):

| training seed | AESTHETIC-v2 | CONTROL-v2 |
|---|---|---|
| A = 1254 | original (`RUN-AESTHETIC-4`) | new `REP-CTL-S1254` |
| B = 9865 | new `REP-AES-S9865` | original (`RUN-CONTROL-4`) |
| C = 42160 | new `REP-AES-S42160` | new `REP-CTL-S42160` |

Seed C and the order of the four trainings come from `replication/draw.py` (draw seed 20261004, file
`replication/design.json`): `REP-CTL-S42160`, `REP-AES-S9865`, `REP-AES-S42160`, `REP-CTL-S1254`.
Everything else is identical to the original runs: base model `stabilityai/stable-diffusion-xl-base-1.0` at commit
`462165984030d82259a11f4367a4eed129e94a7b`, the same 89 photographs and captions per corpus (frozen datasets
AESTHETIC-v2, CONTROL-v2), LoRA rank 16, alpha 16, learning rate 1e-4, batch 1, 1,500 steps, resolution 1024,
configuration `{"noise_offset":0.0357,"timestep_min":250}`, instance L4-1-24G. In the worker the training seed sets
every random source (order of the images, noise, timesteps), so two runs with the same seed and different corpora share
the same random sequence.

*Blocking checks (a replication run is used only if all hold; verified on 2026-10-01 for the original runs):* worker
script SHA-256 `266541cf200e590dd5a1d6fd68429c50858b0c48eb64172507220882f833b95f`; torch 2.4.1+cu121, diffusers 0.30.3,
peft 0.12.0, transformers 4.44.2, accelerate 0.34.2; GPU NVIDIA L4; run parameters identical to the table above. If a
check fails for the new runs, the two original runs are trained again in the new environment (6 new LoRA instead of 4)
and the three pairs are formed from the new runs only.

*Determinism (recommended control, one more run):* no two existing runs share corpus, seed and configuration, so whether
a training is reproducible bit for bit is not known. `REP-AES-S1254-R` repeats `RUN-AESTHETIC-4` exactly. If the weights
are identical (same SHA-256) the training is deterministic. If not, its 192 images give the size of the variation with
*everything* equal, the floor against which the seed and corpus effects are read. If this run is not made, the report
says that determinism was not verified.

*Platform (2026-10-01):* the four runs were created on the platform with the parameters copied from the original runs
(checked field by field: no difference except code, corpus and seed). Cloud jobs were allowed only in the dataset-selection
and training phases; the platform was changed so that, after the training phase, a training run that no experimental
condition uses and the generation of a test plan can still be started (nothing else changes; the worker script is
untouched, same SHA-256).

*Execution (2026-10-02):* the four runs completed on NVIDIA L4 with worker script, library versions and parameters
identical to the original runs (blocking checks passed, `replication/check_images.py`). The control run
`REP-AES-S1254-R` was also made: same final loss as `RUN-AESTHETIC-4` (0.0863) but different weights (different SHA-256):
the training is not reproducible bit for bit, so its images are generated and analysed (8.5, determinism control).
The 768 images were generated by a test plan with the parameters of the main plans; a first attempt stopped at the first
image because the platform refused imports after the start of the collection (now allowed for test plans only; no image
had been stored). Result files are named after their analysis (`report_replication_dino.md`, …); the baseline files of
the embedding report were renamed without changing their content (`baseline/RENAMED_FILES.csv`).

**8.3 Generation.** For each new LoRA: the same 48 prompts (prompt set of the main plans) × the same 4 generation seeds
(25478, 85, 6127, 7), 1216×832, DPM++ 2M Karras, 40 steps, guidance 5: 192 images per LoRA, 768 new images (960 with the
determinism run). On the platform: one test condition per run and one test generation plan (its images are never shown
to participants and do not enter the vote pages or the analyses of sections 1–5). BASE and the two original LoRA are not
generated again: their 576 images are those of the baseline.

**8.4 Embeddings.** Same model, same preprocessing and same image version as the baseline (the 480-px thumbnails served
by the platform; `image_scripts/embeddings.py`). No other embedding model is added.

**8.5 Analysis (DINOv2, primary).** Unit: the prompt × generation-seed cell (192 cells). y(corpus, training seed, cell) =
projection of the image on the frozen direction (CLS).
* *Primary endpoint:* for each training seed s, D_s = mean over the cells of y(AESTHETIC, s) − y(CONTROL, s) (the BASE
  image cancels). 95% interval: bootstrap over prompts. All 192 cells; *sensitivity:* the 159 cells shown to the
  participants in the original experiment.
* *Criterion, fixed in advance.* The contrast is said to **replicate across training seeds** if (i) D_A, D_B and D_C are
  all positive and (ii) each of the three intervals excludes zero. With three seeds no test "between seeds" is possible
  (a sign test cannot go below p = .25) and none is reported: the three values, their mean, range and standard deviation
  are reported as they are, whatever they are, including negative ones.
* *Corpus effect against seed effect (secondary, fixed in advance).* For each corpus, the three differences between
  training seeds, y(corpus, s) − y(corpus, s′), with their intervals; ratio R = mean of the three |D_s| / mean of the six
  |between-seed differences|. R well above 1 = the corpus moves the images along the direction more than the training
  seed does. The original unpaired contrast (AESTHETIC seed A − CONTROL seed B) is shown next to the three paired ones.
* *Variance components (descriptive):* corpus, training seed, corpus × training seed, prompt, generation seed within
  prompt, from the 6 × 192 values; with three training seeds the seed components are imprecise and are reported with that
  warning.
* *Already frozen sensitivities only:* mean of the patch tokens instead of CLS; amount of shift against direction
  (as in 6, B5); specificity (as in 6, B2); embedding transfer ratio D_s / gap.
* *Proxy, labelled as such:* the rating predictor trained on the phase-1 photographs, applied to the new images. It is a
  computational proxy, not human evidence, and is not used to say what people would prefer.

**8.6 VLM stage (secondary), only if criterion 8.5 is met.** Same blind protocol and rubric as section 7
(`VLM_RUBRIC_V1`, one image per request, same three model identifiers if still available; an identifier no longer
available is recorded and that model is left out). Subset fixed now, before any embedding of the new images:
`replication/vlm_subset.csv`, 24 prompts × 2 generation seeds, balanced by building category (48 cells) × 6 LoRA = 288
images, all evaluated in the same batch (the images of the two original LoRA are evaluated again, their old scores are not
reused), plus 20% repeats. Scores standardised within model over the 288 images. *Question:* do the measures that
emerged in section 7 reappear with the same sign in the three paired contrasts? Measures named in advance:
`representation_quality`, `proportional_coherence`, `component_coherence`, `material_rendering`,
`photographic_composition` (the architectural composite is reported next to them). No search for new differences. If
they do not reappear, this is reported: part of the reading of section 7 would belong to the specific training run.

*Result of 8.5 (2026-10-02):* criterion met (D = +0.0435, +0.0412, +0.0504, each interval above zero;
`replication_report/report_replication_dino.md`). The VLM stage was therefore run.
*State of the VLM run (2026-10-02, before unblinding):* blind manifest committed before the first request; Anthropic 346
of 346 and Google 346 of 346 valid; OpenAI stopped at 149 of 346 (no credit left), i.e. the first 24 cells in the random
order. *Decision:* the key is **not** opened until the OpenAI evaluations are complete (nothing requires opening it
earlier); the remaining requests use the same script and manifest. The analysis script was tested with shuffled corpus
labels only. Under shuffled labels three positive signs occur by chance about once in eight: the report therefore gives,
next to the sign rule fixed above, the interval of the mean of the three seeds.
*Run complete (2026-10-02):* after the credit was added the remaining OpenAI requests were made: 346 of 346 valid for each
of the three models (1,038 evaluations). Raw outputs committed, then the key was opened.

*Determinism control (2026-10-02):* `REP-AES-S1254-R` (192 images, same checks passed) was added to the report. Trained
twice with everything equal, the LoRA gives different images: embedding distance 0.32 per cell, against 0.63 when only the
training seed changes and 0.68 when only the corpus changes; along the frozen direction the mean difference is −0.002
(interval −0.007 / +0.003), i.e. no systematic shift. None of the estimates already reported changed; the intervals of the
later sections moved by at most 0.003 (the bootstrap stream is shared).

*Result of 8.6 (2026-10-02; `vlm_report/report_replication_vlm.md`):* **none of the five measures named in advance
reappears** (none is positive for the three training seeds); the mean of the three seeds is close to zero for four of
them and negative for `component_coherence`. Between-seed differences within the same corpus are as large as the corpus
contrasts (R = 1.4 and 1.2 for the two composites, against 4.8 for the embedding direction). *Check added after
unblinding and labelled as such in the report:* on this batch the original unpaired contrast of section 7 (AESTHETIC seed
A − CONTROL seed B) is again positive for the representation block (+0.14), and the original CONTROL LoRA scores lower
than the two other CONTROL LoRA on that block (−0.28, interval below zero): the advantage read in section 7 belongs for
a good part to that specific CONTROL run. The same images scored in section 7 and here correlate 0.73–0.83.
*Reading, by the rule of 8.7:* embedding contrast replicated, VLM pattern not replicated — the contribution is the
representational transfer, not the specific visual characteristics; section 7 is to be presented as a result of the
original pair of runs, not as a property of the corpus.

**8.7 Reading of the outcome, fixed in advance.** Embedding contrast replicated and VLM pattern at least partly
replicated: the curation of the corpus produces a replicable orientation of the generative behaviour. Embedding contrast
replicated, VLM pattern not: the contribution is the representational transfer, not the specific visual characteristics.
Neither replicated: the interpretation of sections 6 and 7 is scaled down, the randomness of the fine-tuning being a
dominant part of the phenomenon. From here on no analysis is added because a result is not strong enough: an addition
must answer a stated scientific weakness and is written here, dated, before it is run.

### 9. What the source-set direction contains — exploratory decomposition

**Status: protocol written and committed on 2026-10-02, before any covariate of the photographs was computed.** Only
existing data are used; no new human data, no generation, no VLM request.

**Why.** Section 8 showed that the embedding contrast is replicated across training seeds while the visual
characteristics read by the VLM are not. The replicated contrast is a robust "signature" of the corpus; what it is made of
is not known. *Note on the comparison with the human preference:* the share transferred is of the same order for the two
(embedding direction: 0.043 of 0.235, 18%; mean rating: +0.105 of the 1.08 points that separate the two photograph sets,
10%); what differs is the certainty (the first is replicated on three training seeds, the second rests on one pair of
runs). **Question:** how much of the AESTHETIC − CONTROL separation along the direction is accounted for by photography,
framing and scene, architectural content, and catalogue metadata?

**9.1 Covariates, in four blocks fixed in advance** (computed on the same 480-px thumbnails used for the embeddings):
* *photography* (10, `image_scripts/image_metrics.py`): brightness, contrast, saturation, colourfulness, warmth,
  sharpness, detail, sky brightness, share of near-black and of near-white pixels;
* *framing and scene* (11, CLIP ViT-L/14 zero-shot, the descriptions of `image_scripts/content.py`): whole building,
  low angle, sunny, vintage photograph, real photograph, people, cars, greenery, water, urban, interior;
* *architectural content* (10, same CLIP zero-shot): iconic design, complex form, curved/organic, monumental,
  contemporary, glass, concrete, wood, white, colourful;
* *catalogue metadata* (photographs only): building type (the 10 most frequent + other), style (the 8 most frequent +
  other), period, continent, industrial, year.
The CLIP attribute "elegant building vs ugly building" is a judgement of beauty, not a description: it is left out of the
blocks and reported on its own.

**9.2 Target.** The relative source-set affinity of each photograph (section 6, A4; leave-one-out for the members of the
two sets), i.e. its position along the frozen direction.

**9.3 Analyses.**
1. *How much of the direction each block describes:* cross-validated R² (ridge, 10-fold) of the affinity on each block
   alone, on all blocks, and the unique part of each block (all blocks minus all the others), over the 600 photographs;
   bootstrap intervals.
2. *How much of the separation between the two sets each block accounts for:* the covariate model is fitted on the 422
   photographs that belong to neither set and applied to the 178 of the two sets; accounted share = predicted
   AESTHETIC − CONTROL gap / observed gap, for each block and for all (bootstrap intervals). The model never sees the two
   sets, so it cannot learn the difference itself.
3. *Residual direction:* the part of the embeddings that is linearly predictable from the image-derived blocks
   (photography, framing and scene, architectural content; fitted on the 422) is removed from every embedding,
   photographs and generated images. The direction is recomputed on the residuals. Reported: its length against the
   original gap; whether it still follows the ratings of the 422 photographs; whether the AESTHETIC − CONTROL contrast of
   the generated images along it is still positive for the three training seeds of section 8.
4. *Same variables in the generated images:* for each covariate, the AESTHETIC − CONTROL difference between the two
   photograph sets and between the generated images (three paired training seeds), in units of the standard deviation of
   the corpus; correlation between the two profiles across covariates.

**9.4 Reading, fixed in advance.** If photography plus framing and scene account for at least half of the separation
(analysis 2), the direction is mainly a prior of photographic representation. If all blocks together account for less
than a quarter, the direction is not reducible to the variables measured here. In between: mixed. **The two outcomes are
not symmetric:** a large accounted share is a positive finding; a small one only says that these variables do not capture
the direction, not that the remainder is an architectural structure. The metadata block is expected to account for little
by construction, because CONTROL was matched to AESTHETIC on building type and style. "Accounted for" is linear and
correlational: it does not say what caused the consensus.

**9.5 The captions (added on 2026-10-02, before any caption was analysed and before the results of 9.3 were seen).** A
LoRA is trained on image–caption pairs, so part of what is transferred could pass through the text. Verified first: the
replication runs used the same 89 + 89 captions as the original runs (caption snapshots stored in the frozen datasets,
identical word for word in the exports before and after the replication; no empty caption; 35 words on average in both
sets). *Question:* do the captions of the two sets differ? *Analyses:* (a) words (lower case, words present in at least
five captions): share of captions containing each word in the two sets, Fisher exact test, Benjamini–Hochberg correction;
(b) can the set be told from the caption alone? Ridge classifier on word presence, 10-fold cross-validated AUC, permutation
p (1,000 permutations of the labels); (c) which of the unbalanced words also occur in the 48 generation prompts.
*Reading:* an AUC compatible with 0.5 means no evidence that the text carries the difference between the sets; an AUC
clearly above 0.5 does not invalidate anything, but the text is then a second channel to be declared, next to the images.
Captions describe the photographs, so words can differ simply because the photographs differ: this analysis cannot
separate the two.

*Result of 9.5 (2026-10-02; `decomposition_report/report_captions.md`):* the captions tell the two sets apart, AUC 0.85
(95% CI 0.79 / 0.90; permutation p = .001), more than the images do (0.78). Four words differ after correction: `green`
(40% of the AESTHETIC captions against 7%), `sky` (89% against 61%), `river` (12% against 0%) more frequent in AESTHETIC;
`perspective` (in "eye-level perspective": 31% against 63%) more frequent in CONTROL. All the unbalanced words also occur
in the generation prompts. The text is therefore a second channel, to be declared: what differs in words is vegetation,
sky, water and viewpoint — scene and framing, not architecture.

*Result of 9.3 (2026-10-02; `decomposition_report/report_direction_decomposition.md`):* reading by the rule of 9.4:
**mixed**. Share of the AESTHETIC − CONTROL separation accounted for (model fitted on the other 422 photographs):
photography 8%, framing and scene 33%, architectural content 57%, metadata 21%; photography + framing and scene 35%
(below the half fixed for "mainly a photographic prior"); all four blocks 65% (interval 50% / 85%). The direction is thus
described mostly by attributes of what is depicted (more iconic/sculptural, curved, contemporary, monumental, complex,
glass; less concrete) and of how it is framed (low angle, no cars, less urban, less "vintage"), hardly by exposure and
colour. After removing what the 31 image-derived covariates predict, 71% of the length of the direction remains and it
still follows the ratings of the 422 photographs (ρ +0.24 against +0.39), but the contrast of the generated images along
it is positive with an interval above zero for two training seeds of three only (A, C; not B). The profile of the 31
covariates in the generated images follows that of the photographs (r +0.71; same sign for 27 of 31; for most of the
large differences the sign is the same for the three training seeds), at about two thirds of the size. These attributes
come from CLIP zero-shot descriptions that were not validated against human annotation, and 31 covariates were looked at:
the profile is a description, not a set of confirmed effects.

## Verification

`verification/` runs the platform's analysis code on an export and compares every number with an independent
implementation (see its README). Last run on the export of 2026-09-28: 0 discrepancies.
