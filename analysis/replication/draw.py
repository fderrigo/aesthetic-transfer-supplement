"""
Draws of the replication (ANALYSIS_PLAN.md, section 8), made once, before any replication training:

    python draw.py <export.zip | extracted folder>

Reads from the export only the prompt set of the main generation plans (code and category of the 48 prompts) and the
generation seeds. No rating, pairwise result, embedding or VLM score is read. Seed of the draws: 20261004.

Writes design.json (training seed C, order of the trainings) and vlm_subset.csv (24 prompts × 2 generation seeds,
balanced by building category, for the VLM stage of 8.6).
"""
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261004
SEED_A, SEED_B = 1254, 9865          # training seeds of the original AESTHETIC and CONTROL runs


def main():
    folder = sys.argv[1]
    if zipfile.is_zipfile(folder):
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(folder).extractall(tmp)
        folder = tmp
    plans = pd.read_csv(os.path.join(folder, "generation_plan.csv"))
    prompts = pd.read_csv(os.path.join(folder, "prompt_sets.csv"))
    prompts = prompts[prompts.prompt_set_id.isin(plans.prompt_set_id)].sort_values("prompt_code")
    gen_seeds = sorted(int(s) for row in plans.seeds for s in str(row).split())
    assert len(prompts) == 48 and len(gen_seeds) == 4

    rng = np.random.default_rng([SEED, 1])
    seed_c = SEED_A
    while seed_c in (SEED_A, SEED_B):
        seed_c = int(rng.integers(1, 100000))
    runs = [("REP-AES-S%d" % SEED_B, "AESTHETIC-v2", SEED_B), ("REP-CTL-S%d" % SEED_A, "CONTROL-v2", SEED_A),
            ("REP-AES-S%d" % seed_c, "AESTHETIC-v2", seed_c), ("REP-CTL-S%d" % seed_c, "CONTROL-v2", seed_c)]
    order = [runs[i] for i in rng.permutation(len(runs))]
    json.dump(dict(draw_seed=SEED, training_seeds=dict(A=SEED_A, B=SEED_B, C=seed_c),
                   original_runs=[dict(code="RUN-AESTHETIC-4", dataset="AESTHETIC-v2", seed=SEED_A), dict(code="RUN-CONTROL-4", dataset="CONTROL-v2", seed=SEED_B)],
                   training_order=[dict(position=i + 1, code=c, dataset=d, seed=s) for i, (c, d, s) in enumerate(order)],
                   optional_determinism_run=dict(code="REP-AES-S%d-R" % SEED_A, dataset="AESTHETIC-v2", seed=SEED_A, after_position=len(order)),
                   generation_seeds=gen_seeds), open(os.path.join(HERE, "design.json"), "w", encoding="utf-8"), indent=1)

    # VLM subset: half of the prompts of each category (categories with an odd number alternate between rounding down and up), two of the four generation seeds per prompt
    rng = np.random.default_rng([SEED, 2])
    chosen, up = [], False
    for cat, g in prompts.groupby("category"):
        codes = list(rng.permutation(g.prompt_code.to_numpy()))
        k = len(codes) // 2
        if len(codes) % 2:
            k += int(up); up = not up
        chosen += codes[:k]
    assert len(chosen) == 24, len(chosen)
    rows = []
    for p in sorted(chosen):
        for s in sorted(rng.choice(gen_seeds, 2, replace=False)):
            rows.append(dict(prompt_code=p, category=prompts.set_index("prompt_code").category[p], generation_seed=int(s)))
    sub = pd.DataFrame(rows)
    sub.to_csv(os.path.join(HERE, "vlm_subset.csv"), index=False)
    print("training seed C:", seed_c)
    print("training order:", [c for c, _, _ in order])
    print("VLM subset:", len(sub), "prompt/seed cells;", sub.prompt_code.nunique(), "prompts; per category", sub.drop_duplicates("prompt_code").category.value_counts().sort_index().to_dict())
    print("per generation seed:", sub.generation_seed.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
