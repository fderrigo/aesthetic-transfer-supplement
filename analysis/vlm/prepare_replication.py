"""
Blind bundle for the VLM stage of the replication (ANALYSIS_PLAN.md, 8.6):

    python prepare_replication.py <extracted export folder> <folder(s) with the display images, comma separated>

The subset was fixed before any replication image existed: ../replication/vlm_subset.csv (24 prompts × 2 generation
seeds). For each of its 48 cells the six images are taken: the two original LoRA (RUN-AESTHETIC-4, RUN-CONTROL-4) and
the four replication LoRA (the determinism run is not part of this stage). 288 images + 20% repeated.
Reads from the export only the lineage of the images (run, prompt, generation seed, opaque id) and the corpus and seed of
each run; no vote, embedding or score. Same conventions as prepare.py: byte-identical copies under random anonymous
names, a blind manifest, a list of repeats, and a key that is not opened before unblinding.
Order of the requests: cells in random order, the six images of a cell one after the other in random order, repeats at
the end; requests are independent, so the model cannot see the order.
"""
import hashlib
import os
import shutil
import sys

import numpy as np
import pandas as pd

import providers as P

STAGE = "replication"
ORIGINAL = ["RUN-AESTHETIC-4", "RUN-CONTROL-4"]
EXCLUDED = ["REP-AES-S1254-R"]


def main():
    export, folders = sys.argv[1], sys.argv[2].split(",")
    cfg = P.load_config()
    g = pd.read_csv(os.path.join(export, "generated_images.csv"), usecols=["opaque_id", "condition_code", "training_run", "prompt_code", "seed", "generation_plan_id"])
    runs = pd.read_csv(os.path.join(export, "training_runs.csv"), usecols=["code", "training_dataset", "seed"]).set_index("code")
    sub = pd.read_csv(os.path.join(P.HERE, "..", "replication", "vlm_subset.csv"))
    g["opaque_id"] = g.opaque_id.astype(str).str.replace("-", "").str.lower()
    main_plans = set(g[g.training_run.isin(ORIGINAL)].generation_plan_id)
    keep = (g.training_run.isin(ORIGINAL) & g.generation_plan_id.isin(main_plans)) | (g.training_run.astype(str).str.startswith("REP-") & ~g.training_run.isin(EXCLUDED))
    g = g[keep].merge(sub[["prompt_code", "generation_seed"]], left_on=["prompt_code", "seed"], right_on=["prompt_code", "generation_seed"])
    g["cell"] = g.prompt_code + "_" + g.seed.astype(str)
    g["corpus"] = g.training_run.map(lambda r: "AESTHETIC" if str(runs.training_dataset[r]).startswith("AESTHETIC") else "CONTROL")
    g["training_seed"] = g.training_run.map(lambda r: int(runs.seed[r]))
    g = g.sort_values(["cell", "training_run"]).reset_index(drop=True)
    assert len(g) == 288 and (g.groupby("cell").size() == 6).all() and (g.groupby(["corpus", "training_seed"]).size() == 48).all(), "the subset is not 48 cells × 6 LoRA"

    rng = np.random.default_rng([cfg["run"]["selection_seed"], 3])
    rows = []
    for c in rng.permutation(sorted(g.cell.unique())):
        rows += [dict(block="1_cells", src=i, repeat=False) for i in rng.permutation(g.index[g.cell == c].to_numpy())]
    n_rep = round(len(g) * cfg["run"]["retest_share"])
    rows += [dict(block="2_repeats", src=i, repeat=True) for i in rng.permutation(rng.choice(g.index.to_numpy(), n_rep, replace=False))]
    m = pd.DataFrame(rows)
    m["position"] = np.arange(1, len(m) + 1)
    m["anon_id"] = [f"VLM_R{k:05d}" for k in rng.permutation(len(m)) + 1]

    out = os.path.join(P.path(cfg, "blind_bundle"), STAGE)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    sha = []
    for r in m.itertuples():
        src = next((os.path.join(f, g.opaque_id[r.src] + ".jpg") for f in folders if os.path.exists(os.path.join(f, g.opaque_id[r.src] + ".jpg"))), None)
        assert src, f"image {g.opaque_id[r.src]} not found"
        dst = os.path.join(out, r.anon_id + ".jpg")
        shutil.copyfile(src, dst)
        a, b = (hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (src, dst))
        assert a == b
        sha.append(a)
    man = os.path.join(P.HERE, "manifests")
    m[["anon_id", "block", "position"]].to_csv(os.path.join(man, f"{STAGE}_manifest_blind.csv"), index=False)
    orig = m[~m.repeat].set_index("src").anon_id
    m[m.repeat].assign(first_anon_id=lambda d: d.src.map(orig))[["first_anon_id", "anon_id"]].rename(columns={"anon_id": "repeat_anon_id"}).to_csv(os.path.join(man, f"{STAGE}_repeats.csv"), index=False)
    key = m.assign(**{c: g[c][m.src].to_numpy() for c in ["opaque_id", "training_run", "corpus", "training_seed", "prompt_code", "seed", "cell"]}, sha256=sha)
    stem, ext = os.path.splitext(P.path(cfg, "key_file"))
    key[["anon_id", "repeat", "opaque_id", "training_run", "corpus", "training_seed", "prompt_code", "seed", "cell", "sha256"]].to_csv(f"{stem}_{STAGE}{ext or '.csv'}", index=False)
    from PIL import Image
    sizes = {Image.open(os.path.join(out, a + ".jpg")).size for a in m.anon_id}
    print(f"{STAGE}: {len(m)} evaluations ({int((~m.repeat).sum())} images, {int(m.repeat.sum())} repeats), {g.cell.nunique()} cells, {g.prompt_code.nunique()} prompts, image sizes {sizes}")


if __name__ == "__main__":
    main()
