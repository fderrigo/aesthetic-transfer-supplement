# Protocol

| File | Content |
|---|---|
| `ANALYSIS_PLAN.md` | the analysis plan. Sections 1–5: human data. 6: DINOv2 embeddings. 7: blind VLM evaluation. 8: replication across training seeds. 9: decomposition of the direction and captions. Every change, interruption and result is recorded there with its date |
| `TRANSPARENCY_NOTES.md` | what was fixed in advance and what was not; limits of the screening, of the human evaluation and of the instruments |
| `timeline_from_audit_log.csv` | study-level events recorded by the platform (UTC): protocol freeze and changes, phases, selection rule, datasets, prompt sets, generation plans, training runs, cloud jobs, defect screening. No users |
| `consent_text.md` | information and consent shown to participants (English, Italian, Spanish) |
| `vlm_rubric_v1.txt` | the exact text sent to the three vision-language models with each image |
| `generation_prompts.csv` | the 48 prompts (text, negative prompt, building category) |

## Sequence of the study (UTC)

| When | What |
|---|---|
| 2026-09-25 14:20 | protocol frozen; phase 1 opens |
| 2026-09-27 13:35 | phase 1 closed; selection rule created and frozen; AESTHETIC and CONTROL sets generated |
| 2026-09-27 → 09-28 | training runs and test generations (four versions of the adapters) |
| 2026-09-28 15:06–15:14 | blind defect screening of the 192 triplets |
| 2026-09-28 15:31 | phase 2 opens |
| 2026-10-01 12:52 | phase 2 closed |
| 2026-10-01 | embedding analysis; VLM protocol, pilot and full run; baseline frozen; replication protocol |
| 2026-10-02 | replication trainings and generation; replication analyses; decomposition and captions |

## How to read the plan

The plan is a working document kept under version control: notes of the form "*Result of …*", "*State of the run …*",
"*Decision recorded before unblinding …*" were added on the dates they carry. The version-control history that proves
those dates is not part of this anonymous copy and can be shown to the editors.

In the plan, "the platform" is the web application used to collect the data and to run training and generation; file
names such as `vlm/…` or `replication/…` are relative to the `analysis/` folder of this repository.
