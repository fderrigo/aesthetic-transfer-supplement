"""
Freeze of the baseline (ANALYSIS_PLAN.md, section 8.1), run once before any replication training:

    python freeze.py <export.zip>

1. direction_dinov2.npz — the source-set contrast direction and everything Part B of the embedding analysis needs from the
   photographs, computed exactly as in embedding_analysis.py and never recomputed afterwards:
   for `cls` and `patch_mean`: v (unit vector CONTROL centroid → AESTHETIC centroid), gap (its length before normalisation),
   centroid_aesthetic, centroid_control, centroid_corpus, u_aesthetic, u_control (unit vectors corpus centroid → each set).
   Self-check: the AESTHETIC − CONTROL shift of the original generated images along v must reproduce the reported value.
2. BASELINE_MANIFEST.csv — SHA-256 and size of every file of the baseline: the analysis folder as tracked by git, and the
   local files that are not in the repository (export, images, key of the VLM evaluation).
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(ANALYSIS)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    export_zip = sys.argv[1]
    folder = tempfile.mkdtemp()
    zipfile.ZipFile(export_zip).extractall(folder)
    rd = lambda f: pd.read_csv(os.path.join(folder, f))
    as_bool = lambda s: s.astype(str).str.lower().eq("true")
    emb = os.path.join(ANALYSIS, "metrics", "embeddings_dinov2_vitb14.npz")
    z = np.load(emb, allow_pickle=False)
    kind, ids = z["kind"], z["id"]
    ratings = rd("posttraining_ratings.csv")
    used = ratings.groupby("condition_code").training_dataset.agg(lambda x: x.dropna().iloc[0] if x.notna().any() else None).to_dict()
    def members(file, cond):
        d = rd(file)
        return set(d[d.dataset_code == used[cond]].image_code)
    aes, ctl = members("aesthetic_dataset.csv", "AESTHETIC"), members("control_dataset.csv", "CONTROL")
    summ = rd("pretraining_summary.csv").set_index("image_code")
    G = rd("generated_images.csv")
    G = G[as_bool(G.is_active)].assign(o=lambda d: d.opaque_id.astype(str).str.replace("-", "").str.lower(), trip=lambda d: d.prompt_code + "_" + d.seed.astype(str))
    out, check = dict(model=z["model"], revision=z["revision"], preprocessing=z["preprocessing"], datasets=np.array(f"{used['AESTHETIC']} / {used['CONTROL']}"),
                      aesthetic_codes=np.array(sorted(aes)), control_codes=np.array(sorted(ctl))), {}
    for which in ("cls", "patch_mean"):
        E = unit(z[which].astype(np.float64))
        photo = {i: E[n] for n, i in enumerate(ids) if kind[n] == "source"}
        gen = {i: E[n] for n, i in enumerate(ids) if kind[n] == "generated"}
        codes = sorted(c for c in photo if c in summ.index)
        X = np.stack([photo[c] for c in codes])
        A, C = X[[c in aes for c in codes]], X[[c in ctl for c in codes]]
        v = A.mean(0) - C.mean(0)
        gap = float(np.linalg.norm(v))
        v = v / gap
        out.update({f"{which}_v": v, f"{which}_gap": np.array(gap), f"{which}_centroid_aesthetic": A.mean(0), f"{which}_centroid_control": C.mean(0),
                    f"{which}_centroid_corpus": X.mean(0), f"{which}_u_aesthetic": unit(A.mean(0) - X.mean(0)), f"{which}_u_control": unit(C.mean(0) - X.mean(0)),
                    f"{which}_n": np.array([len(A), len(C), len(X)])})
        piv = G[G.o.isin(gen)].pivot(index="trip", columns="condition_code", values="o").dropna()
        eA, eB, eC = (np.stack([gen[o] for o in piv[c]]) for c in ("AESTHETIC", "BASE", "CONTROL"))
        check[which] = (len(piv), float(((eA - eB) @ v - (eC - eB) @ v).mean()), gap)
    np.savez(os.path.join(HERE, "direction_dinov2.npz"), **out)
    for which, (n, d, gap) in check.items():
        print(f"{which}: {n} triplets, AESTHETIC − CONTROL along the direction {d:+.4f}, gap {gap:.3f}, transfer ratio {d / gap:.0%}")
    reported = pd.read_csv(os.path.join(ANALYSIS, "embedding_report", "results_dino.csv"))
    r = reported[(reported.analysis == "AESTHETIC − CONTROL shift along the aesthetic direction") & (reported.part == "B")].set_index("embedding").estimate
    assert all(abs(check[w][1] - r[w]) < 1e-9 for w in check), "the frozen direction does not reproduce the reported result"
    print("self-check passed: identical to embedding_report/results_dino.csv")

    rows = []
    tracked = subprocess.run(["git", "ls-files", "analysis", "src/AestheticArchitectureResearch.Application/Cloud/aar_worker.py"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    for f in sorted(x for x in tracked if x and not x.endswith("BASELINE_MANIFEST.csv")):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            rows.append(dict(path=f, in_repository=True, bytes=os.path.getsize(p), sha256=sha(p)))
    rows.append(dict(path="analysis/baseline/direction_dinov2.npz", in_repository=True, bytes=os.path.getsize(os.path.join(HERE, "direction_dinov2.npz")), sha256=sha(os.path.join(HERE, "direction_dinov2.npz"))))
    local = [("export/" + os.path.basename(export_zip), export_zip)]
    for sub in ("images/source", "images/generated", "vlm/images"):
        d = os.path.join(ANALYSIS, sub)
        local += [(f"analysis/{sub}/{n}", os.path.join(d, n)) for n in sorted(os.listdir(d))]
    local.append(("analysis/images/manifest.csv", os.path.join(ANALYSIS, "images", "manifest.csv")))
    local += [(f"analysis/vlm/{n}", os.path.join(ANALYSIS, "vlm", n)) for n in sorted(os.listdir(os.path.join(ANALYSIS, "vlm"))) if "DO_NOT_USE" in n]
    rows += [dict(path=name, in_repository=False, bytes=os.path.getsize(p), sha256=sha(p)) for name, p in local]
    m = pd.DataFrame(rows).drop_duplicates("path")
    m.to_csv(os.path.join(HERE, "BASELINE_MANIFEST.csv"), index=False)
    print(f"manifest: {len(m)} files ({int(m.in_repository.sum())} in the repository, {int((~m.in_repository).sum())} local only), {m.bytes.sum() / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
