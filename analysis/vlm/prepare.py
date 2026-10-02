"""
Blind bundle for the VLM evaluation (ANALYSIS_PLAN.md, 7.2 and 7.5):

    python prepare.py <export folder or zip> pilot|full

Reads from the export only generated_images.csv and, in it, only: opaque id, condition, prompt, seed, active flag.
No rating, pairwise result or embedding score is read. Nothing is chosen by hand: every draw uses the seed in config.json.

pilot: 15 complete active triplets, 15 different prompts, the four seeds balanced (45 images) + 9 images repeated (20%).
full:  all the generated images + about 20% repeated.

Writes (in the folders of config.json):
  bundle/<stage>/<anon_id>.jpg       byte-identical copies under random anonymous names (repeats are not recognisable)
  manifests/<stage>_manifest_blind.csv   anon_id, block, position — what the runner reads; no condition, prompt or seed
  manifests/<stage>_repeats.csv          pairs of anon ids that are the same image (for test–retest; no condition)
  <key file>_<stage>.csv                 anon_id → image, condition, prompt, seed, active: NOT read before unblinding (7.7)

`block`/`position` give the order of the requests: the three images of a triplet one after the other (in random order),
triplets in random order; in the full run first the triplets shown to the participants, then the repeats, then the
triplets with an image not shown. A provider that stops for lack of credit therefore leaves complete triplets and the
primary set first. Each request is independent, so the order is not visible to the model.
"""
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

import providers as P


def main():
    export, stage = sys.argv[1], sys.argv[2]
    assert stage in ("pilot", "full")
    cfg = P.load_config()
    if zipfile.is_zipfile(export):
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(export).extract("generated_images.csv", tmp)
        export = tmp
    g = pd.read_csv(os.path.join(export, "generated_images.csv"), usecols=["opaque_id", "condition_code", "prompt_code", "seed", "is_active"])
    g["opaque_id"] = g.opaque_id.astype(str).str.replace("-", "").str.lower()
    g["active"] = g.is_active.astype(str).str.lower().eq("true")
    g["triplet"] = g.prompt_code + "_" + g.seed.astype(str)
    g = g.sort_values(["prompt_code", "seed", "condition_code"]).reset_index(drop=True)
    size = g.groupby("triplet").size()
    assert (size == 3).all(), "incomplete triplets in the export"
    shown = g.groupby("triplet").active.all()                     # the three images were shown to the participants
    rng = np.random.default_rng([cfg["run"]["selection_seed"], 1 if stage == "pilot" else 2])
    images = P.path(cfg, "images")

    if stage == "pilot":
        act = g[g.triplet.map(shown)]
        prompts = rng.permutation(sorted(act.prompt_code.unique()))[:15]
        used, chosen = {}, []
        for p in prompts:                                         # for each prompt the seed used least so far (ties at random)
            seeds = rng.permutation(sorted(act[act.prompt_code == p].seed.unique()))
            s = min(seeds, key=lambda x: used.get(x, 0))
            used[s] = used.get(s, 0) + 1
            chosen.append(f"{p}_{s}")
        g = g[g.triplet.isin(chosen)].copy()
        n_rep = 9
    else:
        n_rep = round(len(g) * cfg["run"]["retest_share"])

    # order: triplets at random (shown first), images of a triplet at random; repeats in between the two groups
    rows = []
    trips = list(rng.permutation(sorted(g.triplet.unique())))
    first = [t for t in trips if shown[t]]
    last = [t for t in trips if not shown[t]]
    rep_idx = rng.choice(g.index.to_numpy(), n_rep, replace=False)
    def add(block, idx, repeat):
        for i in idx:
            rows.append(dict(block=block, src=i, repeat=repeat))
    for t in first:
        add("1_shown", rng.permutation(g.index[g.triplet == t].to_numpy()), False)
    add("2_repeats", rng.permutation(rep_idx), True)
    for t in last:
        add("3_not_shown", rng.permutation(g.index[g.triplet == t].to_numpy()), False)
    m = pd.DataFrame(rows)
    m["position"] = np.arange(1, len(m) + 1)
    prefix = "VLM_P" if stage == "pilot" else "VLM_F"
    m["anon_id"] = [f"{prefix}{k:05d}" for k in rng.permutation(len(m)) + 1]      # the name says nothing about order or repeats

    out = os.path.join(P.path(cfg, "blind_bundle"), stage)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    sha = []
    for r in m.itertuples():
        src = os.path.join(images, g.opaque_id[r.src] + ".jpg")
        dst = os.path.join(out, r.anon_id + ".jpg")
        shutil.copyfile(src, dst)
        a, b = (hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (src, dst))
        assert a == b, "copy differs from the original"
        sha.append(a)
    man = os.path.join(P.HERE, "manifests")
    os.makedirs(man, exist_ok=True)
    m[["anon_id", "block", "position"]].to_csv(os.path.join(man, f"{stage}_manifest_blind.csv"), index=False)
    orig = m[~m.repeat].set_index("src").anon_id
    reps = m[m.repeat].assign(first_anon_id=lambda d: d.src.map(orig))[["first_anon_id", "anon_id"]].rename(columns={"anon_id": "repeat_anon_id"})
    reps.to_csv(os.path.join(man, f"{stage}_repeats.csv"), index=False)
    key = m.assign(opaque_id=g.opaque_id[m.src].to_numpy(), condition=g.condition_code[m.src].to_numpy(), prompt_code=g.prompt_code[m.src].to_numpy(),
                   seed=g.seed[m.src].to_numpy(), triplet=g.triplet[m.src].to_numpy(), active=g.active[m.src].to_numpy(), sha256=sha)
    stem, ext = os.path.splitext(P.path(cfg, "key_file"))
    key[["anon_id", "repeat", "opaque_id", "condition", "prompt_code", "seed", "triplet", "active", "sha256"]].to_csv(f"{stem}_{stage}{ext or '.csv'}", index=False)
    print(f"{stage}: {len(m)} evaluations ({int((~m.repeat).sum())} images, {int(m.repeat.sum())} repeats), "
          f"{g.triplet.nunique()} triplets, {g.prompt_code.nunique()} prompts; blocks {m.block.value_counts().sort_index().to_dict()}")
    if stage == "pilot":
        print("seeds:", g.drop_duplicates("triplet").seed.value_counts().to_dict())


if __name__ == "__main__":
    main()
