"""
Blocking check of the replication images (ANALYSIS_PLAN.md, 8.2–8.3), before any embedding is computed:

    python check_images.py <extracted export folder> <thumbnail folder>

Every new image must have been generated with the parameters of the original images (everything except LoRA, condition
and job), with the same prompt text, negative prompt and generation seed as the original image of its cell, by the LoRA
of its run, and each replication run must cover the 192 cells exactly once. Exits with an error if anything differs.
"""
import json
import os
import sys

import pandas as pd

SKIP = {"condition", "lora_run", "lora_sha256", "cloud_job", "seed", "prompt", "negative_prompt"}


def main():
    folder, thumbs = sys.argv[1], sys.argv[2]
    g = pd.read_csv(os.path.join(folder, "generated_images.csv"))
    runs = pd.read_csv(os.path.join(folder, "training_runs.csv")).set_index("code")
    g["o"] = g.opaque_id.str.replace("-", "").str.lower()
    g["p"] = g.generation_parameters_json.map(json.loads)
    old = g[g.training_run.isin(["RUN-AESTHETIC-4", "RUN-CONTROL-4"]) | (g.condition_code == "BASE")]
    new = g[g.training_run.astype(str).str.startswith("REP-")]
    norm = lambda p: tuple(sorted((k, str(v)) for k, v in p.items() if k not in SKIP))
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("OK   " if cond else "FAIL ") + name + (" — " + detail if detail else ""))
    a, b = set(old.p.map(norm)), set(new.p.map(norm))
    check("generation parameters identical to the original images", a == b and len(a) == 1, json.dumps(dict(next(iter(b)))))
    base = old[old.condition_code == "BASE"].set_index(["prompt_code", "seed"]).p
    same = new.apply(lambda r: (r.p["prompt"], r.p.get("negative_prompt"), r.p["seed"]) == (base[(r.prompt_code, r.seed)]["prompt"], base[(r.prompt_code, r.seed)].get("negative_prompt"), base[(r.prompt_code, r.seed)]["seed"]), axis=1)
    check("prompt text, negative prompt and generation seed equal to the original cell", same.all(), f"{int(same.sum())} of {len(new)}")
    check("LoRA of the image = LoRA of its run", (new.apply(lambda r: r.p["lora_run"] == r.training_run, axis=1)).all())
    for code, d in new.groupby("training_run"):
        cells = d[["prompt_code", "seed"]].drop_duplicates()
        check(f"{code}: 192 cells once, one LoRA file, size 1216x832", len(d) == 192 and len(cells) == 192 and d.p.map(lambda p: p["lora_sha256"]).nunique() == 1 and set(zip(d.width, d.height)) == {(1216, 832)},
              f"corpus {runs.training_dataset[code]}, training seed {runs.seed[code]}, LoRA {d.p.iloc[0]['lora_sha256'][:16]}, status {runs.status[code]}")
    keys = ["lora_rank", "lora_alpha", "learning_rate", "batch_size", "steps", "resolution", "configuration_json", "base_model", "base_model_revision", "method"]
    ref = {"AESTHETIC": runs.loc["RUN-AESTHETIC-4", keys], "CONTROL": runs.loc["RUN-CONTROL-4", keys]}
    for code in sorted(new.training_run.unique()):
        r = ref["AESTHETIC" if str(runs.training_dataset[code]).startswith("AESTHETIC") else "CONTROL"]
        check(f"{code}: training parameters equal to the original run", (runs.loc[code, keys].astype(str) == r.astype(str)).all())
    missing = [o for o in new.o if not os.path.exists(os.path.join(thumbs, o + ".jpg"))]
    check("thumbnail downloaded for every new image", not missing, f"{len(new) - len(missing)} of {len(new)}")
    print("\nALL CHECKS PASSED" if ok else "\nSTOP: at least one check failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
