"""
ARCH300 — replication across training seeds, embedding analysis (confirmatory; plan: ANALYSIS_PLAN.md, section 8.5).

    python replication_analysis.py <export.zip | extracted folder> <new embeddings .npz> [--out replication_report]

Written and committed before any image of the replication existed. Uses the frozen direction
(baseline/direction_dinov2.npz, never recomputed), the baseline embeddings (metrics/embeddings_dinov2_vitb14.npz) for the
images of BASE and of the two original LoRA, and the new embeddings for the images of the replication LoRA. From the
export: the lineage of every generated image (training run → corpus and training seed), and, for the labelled proxy only,
the phase-1 ratings of the photographs. No phase-2 vote and no VLM score is read.

Unit: the prompt × generation-seed cell. y(corpus, training seed, cell) = projection of the image on the frozen direction.
Intervals: bootstrap over prompts. Criterion (fixed in advance): the three paired contrasts D_s are all positive and each
interval excludes zero. No test between training seeds is made (three seeds).
Writes <out>/report_replication_dino.md, results_replication_dino.csv, cells_replication_dino.csv.
"""
from __future__ import annotations

import argparse
import itertools
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd

from embedding_analysis import Report, ridge_cv, ridge_fit_predict, unit

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N_BOOT = 20261005, 5000
ORIGINAL = {"RUN-AESTHETIC-4": ("AESTHETIC", 1254), "RUN-CONTROL-4": ("CONTROL", 9865)}
DETERMINISM_RUN = "REP-AES-S1254-R"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("embeddings")
    ap.add_argument("--out", default=os.path.join(HERE, "replication_report"))
    args = ap.parse_args()
    folder = args.export
    if zipfile.is_zipfile(folder):
        folder = tempfile.mkdtemp()
        zipfile.ZipFile(args.export).extractall(folder)
    rd = lambda f: pd.read_csv(os.path.join(folder, f))
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    direction = np.load(os.path.join(HERE, "baseline", "direction_dinov2.npz"), allow_pickle=False)
    base_npz = np.load(os.path.join(HERE, "metrics", "embeddings_dinov2_vitb14.npz"), allow_pickle=False)
    new_npz = np.load(args.embeddings, allow_pickle=False)
    assert str(new_npz["model"]) == str(base_npz["model"]) == str(direction["model"]) and str(new_npz["preprocessing"]) == str(base_npz["preprocessing"]), "embedding model or preprocessing differ from the baseline"

    # ------------------------------------------------------------ design: which image belongs to which corpus / training seed
    runs = rd("training_runs.csv").set_index("code")
    G = rd("generated_images.csv")
    G["o"] = G.opaque_id.astype(str).str.replace("-", "").str.lower()
    G["cell"] = G.prompt_code + "_" + G.seed.astype(str)
    main_plans = set(G[G.training_run.isin(ORIGINAL)].generation_plan_id)
    def label(r):
        if r.condition_code == "BASE" and r.generation_plan_id in main_plans:
            return ("BASE", 0)
        if r.training_run in ORIGINAL and r.generation_plan_id in main_plans:
            return ORIGINAL[r.training_run]
        if isinstance(r.training_run, str) and r.training_run.startswith("REP-") and r.training_run in runs.index:
            corpus = "AESTHETIC" if str(runs.training_dataset[r.training_run]).startswith("AESTHETIC") else "CONTROL"
            return (corpus + "-R" if r.training_run == DETERMINISM_RUN else corpus, int(runs.seed[r.training_run]))
        return (None, None)
    lab = G.apply(label, axis=1, result_type="expand")
    G["corpus"], G["tseed"] = lab[0], lab[1]
    G = G[G.corpus.notna()].copy()
    was_shown = G[G.generation_plan_id.isin(main_plans)].groupby("cell").is_active.agg(lambda x: x.astype(str).str.lower().eq("true").all())
    cells = sorted(G[G.corpus == "BASE"].cell.unique())
    prompt = np.array([c.rsplit("_", 1)[0] for c in cells])
    prompts = np.unique(prompt)
    seeds = sorted(int(s) for s in G[G.corpus.isin(["AESTHETIC", "CONTROL"])].tseed.unique())
    name = {s: n for s, n in zip([1254, 9865] + [s for s in seeds if s not in (1254, 9865)], "ABCDEFG")}
    shown = was_shown.reindex(cells).fillna(False).to_numpy(bool)

    rep = Report()
    rep.p("# ARCH300 — replication across training seeds (DINOv2 embeddings)")
    rep.p(f"Confirmatory analysis (ANALYSIS_PLAN.md, section 8). Model `{str(base_npz['model'])}`; frozen direction `baseline/direction_dinov2.npz` (datasets {str(direction['datasets'])}). "
          f"Seed {SEED}; {N_BOOT} bootstrap samples over prompts. {len(cells)} prompt × generation-seed cells ({len(prompts)} prompts), {int(shown.sum())} of them shown to the participants in the original experiment.")

    def boot(x, mask=None):
        """Mean and 95% bootstrap interval over prompts of a per-cell quantity."""
        m = np.ones(len(x), bool) if mask is None else mask
        x, pr = x[m], prompt[m]
        qs = np.unique(pr)
        mean = np.array([x[pr == q].mean() for q in qs]); w = np.array([(pr == q).sum() for q in qs])
        i = rng.integers(0, len(qs), (N_BOOT, len(qs)))
        b = (mean[i] * w[i]).sum(1) / w[i].sum(1)
        return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(m.sum())

    summary = {}
    for which in ("cls", "patch_mean"):
        full = which == "cls"
        v, gap = direction[f"{which}_v"], float(direction[f"{which}_gap"])
        uA, uC = direction[f"{which}_u_aesthetic"], direction[f"{which}_u_control"]
        E = {}
        for z in (base_npz, new_npz):
            e = unit(z[which].astype(np.float64))
            E.update({i: e[n] for n, i in enumerate(z["id"]) if z["kind"][n] == "generated"})
        def matrix(corpus, tseed):
            g = G[(G.corpus == corpus) & (G.tseed == tseed)].drop_duplicates("cell").set_index("cell").reindex(cells)
            assert g.o.notna().all() and g.o.isin(E).all(), f"missing images or embeddings for {corpus} seed {tseed}"
            return np.stack([E[o] for o in g.o])
        eB = matrix("BASE", 0)
        e = {(c, s): matrix(c, s) for c in ("AESTHETIC", "CONTROL") for s in seeds}
        y = {k: m @ v for k, m in e.items()}
        tag = dict(emb=which)

        # ---- primary: paired contrasts
        D = {s: boot(y[("AESTHETIC", s)] - y[("CONTROL", s)]) for s in seeds}
        Dsh = {s: boot(y[("AESTHETIC", s)] - y[("CONTROL", s)], shown) for s in seeds}
        for s in seeds:
            rep.add("primary", f"D: AESTHETIC − CONTROL along the frozen direction, training seed {name[s]} = {s}", *D[s][:3], None, D[s][3], **tag)
            rep.add("primary", f"D, cells shown to the participants only, training seed {name[s]} = {s}", *Dsh[s][:3], None, Dsh[s][3], **tag)
            rep.add("primary", f"embedding transfer ratio D / gap, training seed {name[s]} = {s}", D[s][0] / gap, D[s][1] / gap, D[s][2] / gap, None, D[s][3], **tag)
        vals = np.array([D[s][0] for s in seeds])
        met = bool((vals > 0).all() and all(D[s][1] > 0 for s in seeds))
        rep.add("primary", "criterion met (1 = the three contrasts are positive and each interval excludes zero)", float(met), None, None, None, len(seeds), **tag)
        rep.add("primary", "mean of the paired contrasts", vals.mean(), vals.min(), vals.max(), None, len(seeds), **tag)
        rep.add("primary", "standard deviation of the paired contrasts across training seeds", vals.std(ddof=1), None, None, None, len(seeds), **tag)
        unpaired = boot(y[("AESTHETIC", 1254)] - y[("CONTROL", 9865)])
        rep.add("primary", "original unpaired contrast (AESTHETIC seed A − CONTROL seed B)", *unpaired[:3], None, unpaired[3], **tag)

        # ---- corpus effect against seed effect
        between = {(c, a, b): boot(y[(c, a)] - y[(c, b)]) for c in ("AESTHETIC", "CONTROL") for a, b in itertools.combinations(seeds, 2)}
        for (c, a, b), r in between.items():
            rep.add("seed effect", f"{c}: training seed {name[a]} − {name[b]} along the frozen direction", *r[:3], None, r[3], **tag)
        ratio = np.abs(vals).mean() / np.mean([abs(r[0]) for r in between.values()])
        rep.add("seed effect", "R = mean |corpus contrast| / mean |between-seed difference|", ratio, None, None, None, None, **tag)
        # per-cell version: how far the image moves when only the corpus, or only the training seed, changes
        cell_corpus = np.mean([np.abs(y[("AESTHETIC", s)] - y[("CONTROL", s)]) for s in seeds], 0)
        cell_seed = np.mean([np.abs(y[(c, a)] - y[(c, b)]) for c in ("AESTHETIC", "CONTROL") for a, b in itertools.combinations(seeds, 2)], 0)
        rep.add("seed effect", "per cell: mean |projection difference| when only the corpus changes", *boot(cell_corpus)[:3], None, len(cells), **tag)
        rep.add("seed effect", "per cell: mean |projection difference| when only the training seed changes", *boot(cell_seed)[:3], None, len(cells), **tag)
        dist_corpus = np.mean([np.linalg.norm(e[("AESTHETIC", s)] - e[("CONTROL", s)], axis=1) for s in seeds], 0)
        dist_seed = np.mean([np.linalg.norm(e[(c, a)] - e[(c, b)], axis=1) for c in ("AESTHETIC", "CONTROL") for a, b in itertools.combinations(seeds, 2)], 0)
        rep.add("seed effect", "per cell: embedding distance between images when only the corpus changes", *boot(dist_corpus)[:3], None, len(cells), **tag)
        rep.add("seed effect", "per cell: embedding distance between images when only the training seed changes", *boot(dist_seed)[:3], None, len(cells), **tag)

        # ---- variance components (descriptive): y = corpus + seed + corpus×seed + prompt + cell within prompt + residual
        Y = np.stack([np.stack([y[(c, s)] for s in seeds]) for c in ("AESTHETIC", "CONTROL")])          # corpus × seed × cell
        grand = Y.mean()
        comp = dict(corpus=np.var(Y.mean((1, 2)), ddof=1), training_seed=np.var(Y.mean((0, 2)), ddof=1),
                    corpus_x_seed=np.var((Y.mean(2) - Y.mean((1, 2))[:, None] - Y.mean((0, 2))[None, :] + grand).ravel(), ddof=1),
                    prompt=np.var([Y[:, :, prompt == q].mean() for q in prompts], ddof=1),
                    generation_seed_within_prompt=np.mean([np.var(Y[:, :, prompt == q].mean((0, 1)), ddof=1) for q in prompts if (prompt == q).sum() > 1]),
                    residual=np.var((Y - Y.mean((0, 1))[None, None, :] - Y.mean(2)[:, :, None] + grand).ravel(), ddof=1))
        for k, val in comp.items():
            rep.add("variance", f"standard deviation of the means: {k}", float(np.sqrt(val)), None, None, None, None, **tag)

        # ---- frozen sensitivities: amount against direction, specificity
        amt, cosd, spec = {}, {}, {}
        for s in seeds:
            dA, dC = e[("AESTHETIC", s)] - eB, e[("CONTROL", s)] - eB
            nA, nC = np.linalg.norm(dA, axis=1), np.linalg.norm(dC, axis=1)
            cA, cC = unit(dA) @ v, unit(dC) @ v
            amt[s], cosd[s] = boot(nA - nC), boot(cA - cC)
            part_dir = ((nA + nC) / 2 * (cA - cC)).mean() / (y[("AESTHETIC", s)] - y[("CONTROL", s)]).mean()
            spec[s] = boot((dA @ uA - dA @ uC) - (dC @ uA - dC @ uC))
            rep.add("sensitivity", f"amount of shift ‖Δ‖, AESTHETIC − CONTROL, training seed {name[s]}", *amt[s][:3], None, amt[s][3], **tag)
            rep.add("sensitivity", f"direction of shift (cosine), AESTHETIC − CONTROL, training seed {name[s]}", *cosd[s][:3], None, cosd[s][3], **tag)
            rep.add("sensitivity", f"share of the contrast due to direction, training seed {name[s]}", float(part_dir), None, None, None, len(cells), **tag)
            rep.add("sensitivity", f"specificity, training seed {name[s]}", *spec[s][:3], None, spec[s][3], **tag)

        # ---- determinism control (if the run exists)
        det = None
        if ((G.corpus == "AESTHETIC-R")).any():
            eR = matrix("AESTHETIC-R", 1254)
            same = boot(np.linalg.norm(eR - e[("AESTHETIC", 1254)], axis=1))
            proj = boot(np.abs(eR @ v - y[("AESTHETIC", 1254)]))
            signed = boot(eR @ v - y[("AESTHETIC", 1254)])
            det = (same, proj, signed, float((np.linalg.norm(eR - e[("AESTHETIC", 1254)], axis=1) < 1e-6).mean()))
            rep.add("determinism", "same corpus, same training seed, trained twice: embedding distance between the two images of a cell", *same[:3], None, same[3], **tag)
            rep.add("determinism", "same corpus, same training seed, trained twice: |projection difference| per cell", *proj[:3], None, proj[3], **tag)
            rep.add("determinism", "same corpus, same training seed, trained twice: signed projection difference (repeat − original)", *signed[:3], None, signed[3], **tag)
            rep.add("determinism", "share of cells with identical embedding", det[3], None, None, None, len(cells), **tag)
        summary[which] = dict(D=D, met=met, ratio=ratio, vals=vals)

        if not full:
            continue
        f4 = lambda r: f"{r[0]:+.4f} ({r[1]:+.4f} / {r[2]:+.4f})"
        rep.h("1. Primary: AESTHETIC − CONTROL at equal training seed")
        rep.p(f"Projection on the frozen direction (CLS token; the two training sets are {gap:.3f} apart along it). Baseline value, unpaired, 159 cells: +0.0429.")
        rep.table(["training seed", "AESTHETIC LoRA", "CONTROL LoRA", "D (95% CI), all cells", "transfer ratio", "cells with D > 0", "D (95% CI), cells shown to the participants"], [
            [f"{name[s]} = {s}", G[(G.corpus == 'AESTHETIC') & (G.tseed == s)].training_run.iloc[0], G[(G.corpus == 'CONTROL') & (G.tseed == s)].training_run.iloc[0], f"**{f4(D[s])}**", f"{D[s][0] / gap:.0%}",
             f"{int(((y[('AESTHETIC', s)] - y[('CONTROL', s)]) > 0).sum())} of {len(cells)}", f4(Dsh[s])] for s in seeds])
        rep.p(f"**Criterion fixed in advance** (the three contrasts positive, each interval above zero): **{'MET' if met else 'NOT MET'}**. "
              f"Mean of the three {vals.mean():+.4f}, range {vals.min():+.4f} to {vals.max():+.4f}, standard deviation across training seeds {vals.std(ddof=1):.4f}. "
              "With three training seeds no test between seeds is possible and none is reported.")
        rep.p(f"Original unpaired contrast on the same {len(cells)} cells (AESTHETIC seed A − CONTROL seed B): {f4(unpaired)}.")
        rep.h("2. Corpus effect against training-seed effect")
        rep.table(["same corpus, different training seed", "difference along the direction (95% CI)"], [[f"{c}: {name[a]} − {name[b]}", f4(r)] for (c, a, b), r in between.items()])
        rep.p(f"R = mean |corpus contrast| / mean |between-seed difference| = **{ratio:.1f}** (well above 1 = along the direction the corpus moves the images more than the training seed does).")
        rep.table(["per cell, mean over the comparisons", "only the corpus changes", "only the training seed changes"], [
            ["|difference of the projection on the direction|", f"{boot(cell_corpus)[0]:.4f}", f"{boot(cell_seed)[0]:.4f}"],
            ["distance between the two images in embedding space (any direction)", f"{boot(dist_corpus)[0]:.3f}", f"{boot(dist_seed)[0]:.3f}"]])
        rep.h("3. Variance components (descriptive)")
        rep.table(["source", "standard deviation of the means along the direction"], [[k.replace("_", " "), f"{np.sqrt(val):.4f}"] for k, val in comp.items()])
        rep.p("With three training seeds the components that involve the training seed are imprecise; they are shown to give the order of magnitude, not as estimates.")
        rep.h("4. Sensitivities frozen in advance")
        rep.table(["training seed", "amount ‖Δ_A‖ − ‖Δ_C‖ (95% CI)", "direction, cosine difference (95% CI)", "specificity (95% CI)"],
                  [[f"{name[s]} = {s}", f"{amt[s][0]:+.3f} ({amt[s][1]:+.3f} / {amt[s][2]:+.3f})", f"{cosd[s][0]:+.3f} ({cosd[s][1]:+.3f} / {cosd[s][2]:+.3f})", f4(spec[s])] for s in seeds])
        if det:
            rep.h("5. Determinism control")
            rep.p(f"`{DETERMINISM_RUN}` repeats `RUN-AESTHETIC-4` exactly (same corpus, seed, parameters). Per cell, between the image of the original LoRA and the image of the repeated one: "
                  f"embedding distance {det[0][0]:.3f} ({det[0][1]:.3f} / {det[0][2]:.3f}); |projection difference| {det[1][0]:.4f}; signed difference {f4(det[2])}; identical embedding in {det[3]:.0%} of the cells. "
                  f"For comparison: distance when only the training seed changes {boot(dist_seed)[0]:.3f}, when only the corpus changes {boot(dist_corpus)[0]:.3f}.")
        # ---- proxy
        summ = rd("pretraining_summary.csv").set_index("image_code")
        ph = unit(base_npz[which].astype(np.float64))
        codes = [i for n, i in enumerate(base_npz["id"]) if base_npz["kind"][n] == "source" and i in summ.index]
        X = np.stack([ph[list(base_npz["id"]).index(c)] for c in codes]); yr = summ.loc[codes, "mean"].to_numpy(float)
        _, lam = ridge_cv(X, yr, np.random.default_rng(SEED))
        rows = []
        for s in seeds:
            r = boot(ridge_fit_predict(X, yr, e[("AESTHETIC", s)], lam) - ridge_fit_predict(X, yr, e[("CONTROL", s)], lam))
            rep.add("proxy", f"predicted rating AESTHETIC − CONTROL (photo-trained predictor), training seed {name[s]}", *r[:3], None, r[3], **tag)
            rows.append([f"{name[s]} = {s}", f"{r[0]:+.3f} ({r[1]:+.3f} / {r[2]:+.3f})"])
        rep.h("6. Computational proxy (not human evidence)")
        rep.p("Rating predictor trained on the phase-1 photographs only, applied to the generated images. It is a proxy computed by a model: it does not say what people would prefer, and no participant saw the new images.")
        rep.table(["training seed", "predicted rating, AESTHETIC − CONTROL (95% CI)"], rows)
        # ---- per-cell file
        out = pd.DataFrame(dict(prompt_code=prompt, generation_seed=[c.rsplit("_", 1)[1] for c in cells], shown_to_participants=shown, base=eB @ v))
        for (c, s), val in y.items():
            out[f"{c.lower()}_seed_{name[s]}"] = val
        out.to_csv(os.path.join(args.out, "cells_replication_dino.csv"), index=False)

    c, pm = summary["cls"], summary["patch_mean"]
    rep.h("Sensitivity — mean of the patch tokens instead of the CLS token")
    rep.table(["quantity", "CLS token", "mean of patch tokens"], [
        [f"D, training seed {name[s]}", f"{c['D'][s][0]:+.4f} ({c['D'][s][1]:+.4f} / {c['D'][s][2]:+.4f})", f"{pm['D'][s][0]:+.4f} ({pm['D'][s][1]:+.4f} / {pm['D'][s][2]:+.4f})"] for s in seeds] + [
        ["criterion", "met" if c["met"] else "not met", "met" if pm["met"] else "not met"], ["R (corpus / training seed)", f"{c['ratio']:.1f}", f"{pm['ratio']:.1f}"]])
    open(os.path.join(args.out, "report_replication_dino.md"), "w", encoding="utf-8").write("\n".join(rep.lines) + "\n")
    pd.DataFrame(rep.rows).to_csv(os.path.join(args.out, "results_replication_dino.csv"), index=False)
    print("\n".join(rep.lines))


if __name__ == "__main__":
    main()
