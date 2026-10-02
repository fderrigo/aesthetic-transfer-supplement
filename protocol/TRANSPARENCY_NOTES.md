# Transparency notes

What a reader should know before relying on the results. Every statement that refers to a date can be checked in
`timeline_from_audit_log.csv` (times in UTC), in `ANALYSIS_PLAN.md` or in the data. The limits listed here are those of
section 5.5 of the manuscript, with their evidence.

## 1. What was fixed in advance and what was not

| Element | Status |
|---|---|
| Design of phase 1 (scale, question, target of ratings per photograph, balanced incomplete assignment) | fixed when the protocol was frozen, before the first rating (2026-09-25 14:20) |
| **Preference rule of the AESTHETIC set** (at least 10 ratings, mean at least 5 on the 1–7 scale, top 35%, quality exclusions applied; no threshold on dispersion, no cap on the number of images) | **defined after phase 1 and frozen before the datasets and the training; not preregistered.** The audit trail records a preliminary threshold of 33% on 2026-09-27 12:49, while phase 1 was still open, and the final rule at 13:35, frozen at 13:36, after the collection was closed, when the ratings were observable and no generated image existed. **The percentile is not the binding criterion:** the photographs are selected by the mean threshold alone, and the selected set is the same for any percentile between 20% and 50% (`analysis/human_report/report_selection_rule.md`) |
| CONTROL set | 89 photographs not in AESTHETIC, matched 1:1 on building type and style, drawn with the fixed random seed 345145722 |
| Dataset versions | v1 and v2 have identical membership; v2 harmonises the captions |
| Training settings | chosen in technical pilots on 2026-09-27 and 09-28, before phase 2 (noise offset; timesteps below 250 excluded). The same settings for both corpora |
| Training seeds of the two final adapters | 1254 (AESTHETIC) and 9865 (CONTROL): chosen deliberately different and fixed before phase 2 (the records show the seeds and when they were set; the intention is the authors' statement). The two adapters are two independent realisations of the fine-tuning workflow, **one per condition**: the human endpoint compares two workflows, not the pure effect of the corpus at a fixed seed |
| Equivalence intervals (±0.3 points; OR 0.8–1.25) | computed during the analysis with a region that was not preregistered; listed in the reports of `analysis/human_report/`, **not used in the manuscript**, which reports effect sizes and confidence intervals |
| Statistical power | no prospective calculation; a minimum detectable effect is reported a posteriori |
| Research question and hypothesis text stored by the platform (`data/export/experiment.json`) | rewritten on 2026-10-01, after the data collection, when the study was reframed as a pilot; it still carries the working title |
| Embeddings, covariates, VLM, captions | designed after the human data collection: **exploratory** |
| Sensitivity with three training seeds applied to both corpora | post hoc with respect to the human experiment; its protocol, criterion, third seed, order of the trainings and VLM subset were written before the additional trainings (plan, section 8). It does not replicate the human endpoint and does not replace a prospective factorial replication |

## 2. The quality screening that excluded 33 triplets

Before phase 2 the 192 triplets were screened for visible defects (mostly perspective and deformation). If one condition
of a triplet had a recorded defect, the whole triplet was excluded to keep matched cells: 33 triplets (99 images, 17.2%
of the pool). Images directly marked: 21 BASE, 14 AESTHETIC, 13 CONTROL.
The interface showed the three images of a triplet **without the condition label and in random order**. The screening
was done by **one researcher**, who knew the study and had already seen images of the models during the technical
tuning: the blinding concerns the act of marking, not a complete naivety about the systems. The computational analyses
are repeated on the 192 triplets (`report_dino.md` B6; `report_vlm.md` 4).

## 3. Human evaluation

- The contrast AESTHETIC − CONTROL is positive and small (d_z = +0.22); the pairwise choice is close to
  parity. AESTHETIC does not exceed BASE, while BASE exceeds CONTROL.
- **Human data exist only for the two final adapters.** No participant saw the images of the adapters trained for the
  paired-seed sensitivity.
- The curators are a mixed sample by role and expertise and are not geographically representative; 32 participants of
  phase 2 also have a valid phase-1 session. With the ratings valid in this export, two photographs of the frozen
  AESTHETIC set have a mean just below 5 and six photographs outside it reach exactly 5: the selection was made on the
  ratings valid on 2026-09-27, and exclusions decided afterwards moved a few means across the threshold
  (`report_selection_rule.md`). None of those six is in CONTROL.
- Exclusions of answers and participants are flagged in the data (`excluded_from_analysis`, `quality_flags.csv`); their
  sensitivity is in the human report.

## 4. Model-dependent instruments

- DINOv2 and CLIP are probes trained on web-scale data; they do not separate architecture, photography and the bias of
  their own pre-training. Fitting the covariate models on the 422 photographs outside the two sets reduces direct
  circularity, not the bias shared between models.
- The 21 CLIP attributes are zero-shot contrasts between two descriptions written by the authors
  (`analysis/image_scripts/covariates.py`); they were not validated against human annotation.
- `sharpness` and `detail` depend on resolution and compression; `sky_brightness` is the mean luminance of the upper third
  of the image, not a segmentation of the sky.
- VLM: blind to condition, prompt and seed; good test–retest stability within model, low absolute agreement between
  models. In the first run one provider stopped for a spending limit after 471 of 691 evaluations; the key was opened at
  that point and the remaining requests were made later with the same script and manifest (dated notes in the plan,
  section 7).

## 5. Results that qualify the reading of the manuscript

The manuscript states them; the evidence is here.

- The VLM evaluation repeated on the paired training seeds does not reproduce the contrasts of the original pair
  (`report_replication_vlm.md`): the VLM results describe the two final adapters, not a stable property of the corpora.
- In the same batch the original CONTROL adapter scores lower than two other CONTROL adapters on representation
  quality: part of the lower preference for CONTROL may be specific to that run. This evidence does not concern human
  judgements and does not identify the cause.
- The captions of the two training sets can be told apart (`report_captions.md`): the text may be a second channel of
  transfer, not separated here from the visual information.

## 6. Participation and ethics

Unpaid adult volunteers were recruited through social media, messaging broadcast lists, web pages and online forums.
Participation was anonymous and voluntary: no IP address and no identifying or sensitive personal data were stored; all
participants received an information text (`consent_text.md`), gave informed consent and could stop at any time. No
formal review by an ethics committee was sought or obtained. The age is self-declared (lowest range offered: 18–24). The
consent lists age range, professional role and self-assessed expertise as profile data; the country was also collected
as an optional field.
