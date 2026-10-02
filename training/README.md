# Training and generation

`aar_worker.py` is the script that ran on the rented GPU instance for every training run and every generation. Its
SHA-256 was recorded for each job and was the same for the original and the replication runs.

## Training (one LoRA adapter per run)

| Setting | Value |
|---|---|
| base model | `stabilityai/stable-diffusion-xl-base-1.0`, commit `462165984030d82259a11f4367a4eed129e94a7b` |
| data | 89 photographs with their captions (frozen dataset AESTHETIC-v2 or CONTROL-v2) |
| method | LoRA, rank 16, alpha 16 |
| learning rate, batch, steps | 1e-4, 1, 1,500 |
| resolution | 1024 |
| configuration | `{"noise_offset": 0.0357, "timestep_min": 250}` |
| training seed | sets every random source of the run (order of the images, noise, timesteps) |
| hardware and libraries | NVIDIA L4; torch 2.4.1+cu121, diffusers 0.30.3, peft 0.12.0, transformers 4.44.2, accelerate 0.34.2 |

All runs and their seeds are in `data/export/training_runs.csv`. Training is not reproducible bit for bit: repeating a
run with identical settings gives different weights (see the determinism control in
`analysis/replication_report/report_replication_dino.md`).

## Generation

| Setting | Value |
|---|---|
| prompts | 48 (`protocol/generation_prompts.csv`), each with its negative prompt |
| generation seeds | 25478, 85, 6127, 7 |
| size | 1216 × 832 |
| sampler, steps, guidance | DPM++ 2M Karras, 40, 5.0 |
| adapter weight | default (no scaling) |
| precision | float16 |

The full parameters of every image, including the SHA-256 of the adapter that produced it, are in the column
`generation_parameters_json` of `data/export/generated_images.csv`. `analysis/replication/check_images.py` verifies that the
replication images were generated with the parameters, prompts and seeds of the original ones.
