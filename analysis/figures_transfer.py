"""
ARCH300 — what is transferred: the 31 image-derived descriptors against BASE, and the shifts in embedding space.

    python figures_transfer.py <extracted export folder> [--out figures]

Reuses, without change, the pipeline of direction_decomposition.py (section 4 of the plan): the 31 descriptors of
metrics/covariates_images.csv (10 photographic measures, 21 CLIP zero-shot attributes; "elegant" excluded), CLIP
attributes turned into logits, every descriptor standardised on the mean and SD of the 600 photographs, all 192
prompt × generation-seed cells, the three paired training seeds averaged. The BASE scores come from the same run of
image_scripts/covariates.py as the AESTHETIC and CONTROL ones.

X (identical in every figure) = AESTHETIC − CONTROL between the two sets of photographs.
Y = AESTHETIC − CONTROL, AESTHETIC − BASE or CONTROL − BASE in the generated images, paired by cell (and by training seed
for the two adapters), in SD of the corpus. Intervals: bootstrap over prompts (per descriptor) and over the 31
descriptors (for r), 5,000 samples, as in the original analysis. No significance test is made or claimed.

Also: fig_embedding_shifts.png — the DINOv2 embeddings projected on the frozen source-set direction (horizontal) and on
the main orthogonal direction of change of the fine-tuned adapters (vertical): photograph centroids, BASE images, and the
centroids of each adapter with the arrow from BASE.
Writes fig_source_vs_aesthetic_minus_control.png, fig_source_vs_aesthetic_from_base.png, fig_source_vs_control_from_base.png,
fig_embedding_shifts.png, feature_base_contrasts.csv, feature_base_analysis.md.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from direction_decomposition import CONTENT, EXCLUDED_RUNS, FRAMING, ORIGINAL, PHOTO
from embedding_analysis import unit

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N_BOOT = 20261007, 5000
BLOCK = {**{c: "photography" for c in PHOTO}, **{c: "framing and scene" for c in FRAMING}, **{c: "architectural content (CLIP)" for c in CONTENT}}
COLOUR = {"photography": "#5B7DB1", "framing and scene": "#C9873A", "architectural content (CLIP)": "#6E8B5E"}
NAME = {1254: "A", 9865: "B", 42160: "C"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    folder = args.export
    if zipfile.is_zipfile(folder):
        folder = tempfile.mkdtemp()
        zipfile.ZipFile(args.export).extractall(folder)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rd = lambda f: pd.read_csv(os.path.join(folder, f))

    # ------------------------------------------------------------ descriptors, exactly as in direction_decomposition.py
    cov = pd.read_csv(os.path.join(HERE, "metrics", "covariates_images.csv")).set_index("id")
    aff = pd.read_csv(os.path.join(HERE, "embedding_report", "source_affinity_dino.csv")).set_index("image_code")
    codes = list(aff.index)
    cols = PHOTO + FRAMING + CONTENT
    def transform(df):
        z = df[cols + ["elegant"]].astype(float).copy()
        for c in FRAMING + CONTENT + ["elegant"]:
            p = z[c].clip(1e-4, 1 - 1e-4)
            z[c] = np.log(p / (1 - p))
        return z
    Zp_raw = transform(cov.loc[codes])
    mu, sd = Zp_raw.mean(), Zp_raw.std().replace(0, 1)
    Zp = (Zp_raw - mu) / sd
    is_a, is_c = (aff.source_set == "AESTHETIC").to_numpy(), (aff.source_set == "CONTROL").to_numpy()
    X = (Zp.loc[is_a, cols].mean() - Zp.loc[is_c, cols].mean())

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
        if isinstance(r.training_run, str) and r.training_run.startswith("REP-") and r.training_run not in EXCLUDED_RUNS:
            return ("AESTHETIC" if str(runs.training_dataset[r.training_run]).startswith("AESTHETIC") else "CONTROL", int(runs.seed[r.training_run]))
        return (None, None)
    lab = G.apply(label, axis=1, result_type="expand")
    G["corpus"], G["tseed"] = lab[0], lab[1]
    G = G[G.corpus.notna()]
    seeds = sorted(int(s) for s in G[G.corpus != "BASE"].tseed.unique())
    cells = sorted(G[G.corpus == "BASE"].cell.unique())
    prompt = np.array([c.rsplit("_", 1)[0] for c in cells])
    def Z(corpus, s):
        o = G[(G.corpus == corpus) & (G.tseed == s)].drop_duplicates("cell").set_index("cell").o.reindex(cells)
        assert o.notna().all() and o.isin(cov.index).all(), (corpus, s)
        return ((transform(cov.loc[o.to_numpy()]) - mu) / sd)[cols].to_numpy()
    ZB = Z("BASE", 0)
    ZA = {s: Z("AESTHETIC", s) for s in seeds}
    ZC = {s: Z("CONTROL", s) for s in seeds}
    # per cell, mean over the three paired seeds
    dAC = np.mean([ZA[s] - ZC[s] for s in seeds], 0)
    dAB = np.mean([ZA[s] - ZB for s in seeds], 0)
    dCB = np.mean([ZC[s] - ZB for s in seeds], 0)

    qs = np.unique(prompt)
    W = np.array([(prompt == q).sum() for q in qs])
    def boot_cells(M):
        """Mean over cells and bootstrap interval over prompts, for every column of M (cells × descriptors)."""
        means = np.stack([M[prompt == q].mean(0) for q in qs])              # prompts × descriptors
        i = rng.integers(0, len(qs), (N_BOOT, len(qs)))
        b = (means[i] * W[i][:, :, None]).sum(1) / W[i].sum(1)[:, None]
        return M.mean(0), np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
    def boot_r(x, y):
        r = np.corrcoef(x, y)[0, 1]
        b = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            b.append(np.corrcoef(x[i], y[i])[0, 1])
        return r, float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))

    est = {k: boot_cells(M) for k, M in (("A_minus_C", dAC), ("A_minus_BASE", dAB), ("CONTROL_minus_BASE", dCB))}
    T = pd.DataFrame({"feature": cols, "block": [BLOCK[c] for c in cols], "training_A_minus_C": X.to_numpy()})
    for k, (m, lo, hi) in est.items():
        T[f"generated_{k}"], T[f"generated_{k}_ci_low"], T[f"generated_{k}_ci_high"] = m, lo, hi
    T["sign_training"] = np.sign(T.training_A_minus_C).astype(int)
    T["sign_A_minus_C"] = np.sign(T.generated_A_minus_C).astype(int)
    T["sign_A_minus_BASE"] = np.sign(T.generated_A_minus_BASE).astype(int)
    T["sign_CONTROL_minus_BASE"] = np.sign(T.generated_CONTROL_minus_BASE).astype(int)
    T["algebra_abs_error"] = np.abs((T.generated_A_minus_BASE - T.generated_CONTROL_minus_BASE) - T.generated_A_minus_C)
    T.to_csv(os.path.join(args.out, "feature_base_contrasts.csv"), index=False)
    max_err = float(T.algebra_abs_error.max())
    rs = {k: boot_r(T.training_A_minus_C.to_numpy(), T[f"generated_{k}"].to_numpy()) for k in est}
    same = {k: int((T.sign_training == T[f"sign_{k}"]).sum()) for k in est}
    zeros = int((T.sign_training == 0).sum() + sum((T[f"sign_{k}"] == 0).sum() for k in est))

    # ------------------------------------------------------------ figures: same frame for the three scatter plots
    allY = np.concatenate([T.generated_A_minus_C, T.generated_A_minus_BASE, T.generated_CONTROL_minus_BASE])
    lim = float(np.ceil(max(np.abs(T.training_A_minus_C).max(), np.abs(allY).max()) * 10 + 0.5) / 10)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def scatter(ycol, ylabel, title, file, r_, n_same):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=200)
        ax.axhspan(0, lim, 0.5, 1, color="#f1f1ee", zorder=0); ax.axhspan(-lim, 0, 0, 0.5, color="#f1f1ee", zorder=0)
        ax.axhline(0, color="#777", lw=0.8); ax.axvline(0, color="#777", lw=0.8)
        ax.plot([-lim, lim], [-lim, lim], ls=(0, (4, 4)), color="#aaa", lw=0.9, label="identity (same size in the images as in the photographs)")
        for blk, colr in COLOUR.items():
            d = T[T.block == blk]
            ax.errorbar(d.training_A_minus_C, d[ycol], yerr=[d[ycol] - d[ycol + "_ci_low"], d[ycol + "_ci_high"] - d[ycol]], fmt="o", ms=6, color=colr,
                        ecolor=colr, elinewidth=0.7, alpha=0.95, capsize=0, label=blk, zorder=3)
        # labels pushed apart vertically where they would overlap
        order = T.sort_values(ycol).reset_index(drop=True)
        ys = order[ycol].to_numpy().copy()
        step = lim / 22
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < step:
                ys[i] = ys[i - 1] + step
        ys -= (ys.mean() - order[ycol].mean())
        for (x, y0, name_), yl in zip(zip(order.training_A_minus_C, order[ycol], order.feature), ys):
            dx = 0.03 if x >= 0 else -0.03
            ax.annotate(name_, (x, y0), (x + dx, yl), fontsize=7.2, color="#333", ha="left" if x >= 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5, shrinkA=0, shrinkB=2))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("AESTHETIC − CONTROL between the two sets of photographs (SD of the corpus)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.text(0.02, 0.98, f"Pearson r = {r_[0]:+.2f} (95% CI {r_[1]:+.2f} / {r_[2]:+.2f})\nsame sign as the photographs: {n_same} of {len(T)}",
                transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        fig.text(0.01, 0.005, "31 image-derived descriptors; 192 prompt × seed cells; AESTHETIC and CONTROL = mean of three paired training seeds; bars = 95% bootstrap over prompts. "
                 "Descriptors are visual proxies (photographic measures and CLIP zero-shot attributes), not architectural quality.", fontsize=6.5, color="#555", wrap=True)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(args.out, file)); plt.close(fig)

    scatter("generated_A_minus_C", "AESTHETIC − CONTROL in the generated images (SD of the corpus)", "Training-set differences and AESTHETIC − CONTROL in the generated images",
            "fig_source_vs_aesthetic_minus_control.png", rs["A_minus_C"], same["A_minus_C"])
    scatter("generated_A_minus_BASE", "AESTHETIC − BASE in the generated images (SD of the corpus)", "Training-set differences and AESTHETIC shifts from BASE",
            "fig_source_vs_aesthetic_from_base.png", rs["A_minus_BASE"], same["A_minus_BASE"])
    scatter("generated_CONTROL_minus_BASE", "CONTROL − BASE in the generated images (SD of the corpus)", "Training-set differences and CONTROL shifts from BASE",
            "fig_source_vs_control_from_base.png", rs["CONTROL_minus_BASE"], same["CONTROL_minus_BASE"])

    # ------------------------------------------------------------ the three contrasts side by side, labels for the main descriptors only
    panels = (("generated_A_minus_C", "AESTHETIC − CONTROL", "a. between the two adapters", rs["A_minus_C"], same["A_minus_C"]),
              ("generated_A_minus_BASE", "AESTHETIC − BASE", "b. AESTHETIC adapter from BASE", rs["A_minus_BASE"], same["A_minus_BASE"]),
              ("generated_CONTROL_minus_BASE", "CONTROL − BASE", "c. CONTROL adapter from BASE", rs["CONTROL_minus_BASE"], same["CONTROL_minus_BASE"]))
    main_lab = set(T.reindex(T.training_A_minus_C.abs().sort_values(ascending=False).index).feature[:12])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), dpi=200, sharey=True)
    for ax, (ycol, ylab, title, r_, n_same) in zip(axes, panels):
        ax.axhspan(0, lim, 0.5, 1, color="#f1f1ee", zorder=0); ax.axhspan(-lim, 0, 0, 0.5, color="#f1f1ee", zorder=0)
        ax.axhline(0, color="#777", lw=0.7); ax.axvline(0, color="#777", lw=0.7)
        ax.plot([-lim, lim], [-lim, lim], ls=(0, (4, 4)), color="#aaa", lw=0.8)
        for blk, colr in COLOUR.items():
            d = T[T.block == blk]
            ax.errorbar(d.training_A_minus_C, d[ycol], yerr=[d[ycol] - d[ycol + "_ci_low"], d[ycol + "_ci_high"] - d[ycol]], fmt="o", ms=4.5, color=colr, ecolor=colr,
                        elinewidth=0.6, alpha=0.95, capsize=0, label=blk, zorder=3)
        d = T[T.feature.isin(main_lab)].sort_values(ycol).reset_index(drop=True)
        ys = d[ycol].to_numpy().copy(); step = lim / 13
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < step:
                ys[i] = ys[i - 1] + step
        ys -= ys.mean() - d[ycol].mean()
        for (x, y0, nm), yl in zip(zip(d.training_A_minus_C, d[ycol], d.feature), ys):
            right = x >= 0
            ax.annotate(nm, (x, y0), (lim * 0.98 if right else -lim * 0.98, yl), fontsize=6.5, color="#333", ha="right" if right else "left", va="center",
                        arrowprops=dict(arrowstyle="-", color="#c8c8c8", lw=0.5, shrinkA=0, shrinkB=2))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(title + chr(10) + ylab + " in the generated images", loc="left", fontsize=9.5, pad=8)
        ax.set_xlabel("AESTHETIC − CONTROL, photographs (SD)")
        ax.text(0.03, 0.97, f"r = {r_[0]:+.2f} ({r_[1]:+.2f} / {r_[2]:+.2f})" + chr(10) + f"same sign: {n_same} of 31", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
    axes[0].set_ylabel("difference in the generated images (SD of the corpus)")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.text(0.01, 0.01, "31 image-derived descriptors; 192 prompt × seed cells; adapters = mean of three paired training seeds; bars = 95% bootstrap over prompts; labels for the twelve descriptors "
             "with the largest difference between the photograph sets. Dashed line: same size in the images as in the photographs. Descriptors are visual proxies, not architectural quality.", fontsize=6.5, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(args.out, "fig_source_contrasts_three_panels.png")); plt.close(fig)

    # ------------------------------------------------------------ embedding shifts
    direction = np.load(os.path.join(HERE, "baseline", "direction_dinov2.npz"), allow_pickle=False)
    v, gap = direction["cls_v"], float(direction["cls_gap"])
    E = {}
    for f in ("embeddings_dinov2_vitb14.npz", "embeddings_replication_dinov2_vitb14.npz"):
        z = np.load(os.path.join(HERE, "metrics", f), allow_pickle=False)
        e = unit(z["cls"].astype(np.float64))
        E.update({i: e[n] for n, i in enumerate(z["id"])})
    def emb(corpus, s):
        o = G[(G.corpus == corpus) & (G.tseed == s)].drop_duplicates("cell").set_index("cell").o.reindex(cells)
        return np.stack([E[i] for i in o])
    eB = emb("BASE", 0); eA = {s: emb("AESTHETIC", s) for s in seeds}; eC = {s: emb("CONTROL", s) for s in seeds}
    photos = np.stack([E[c] for c in codes])
    cA, cC, cX = photos[is_a].mean(0), photos[is_c].mean(0), photos.mean(0)
    # vertical axis: the common direction of change of the fine-tuned adapters, orthogonal to v
    shift = np.mean([eA[s].mean(0) - eB.mean(0) for s in seeds] + [eC[s].mean(0) - eB.mean(0) for s in seeds], 0)
    w = shift - (shift @ v) * v
    w = w / np.linalg.norm(w)
    P = lambda M: np.column_stack([M @ v, M @ w])
    cent = {("BASE", 0): P(eB.mean(0)[None, :])[0]}
    for s in seeds:
        cent[("AESTHETIC", s)] = P(eA[s].mean(0)[None, :])[0]; cent[("CONTROL", s)] = P(eC[s].mean(0)[None, :])[0]
    pA, pC, pX = (P(c_[None, :])[0] for c_ in (cA, cC, cX))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=200, gridspec_kw=dict(width_ratios=[1.25, 1]))
    pb, pp = P(eB), P(photos)
    ax.scatter(pp[:, 0], pp[:, 1], s=5, color="#d9c9b0", alpha=0.5, label="photographs of the corpus (600)", zorder=0)
    ax.scatter(pb[:, 0], pb[:, 1], s=7, color="#b5b5b5", alpha=0.7, label="BASE images (192)", zorder=1)
    for q, m_, lab_, dy, ha in ((pC, "s", "CONTROL photographs", 10, "right"), (pX, "^", "corpus", -14, "center"), (pA, "D", "AESTHETIC photographs", 10, "left")):
        ax.scatter(*q, s=80, marker=m_, color="#333", edgecolor="white", zorder=5)
        ax.annotate(lab_ + ", centroid", q, (-6 if ha == "right" else 6 if ha == "left" else 0, dy), textcoords="offset points", fontsize=7, color="#333", ha=ha)
    b0 = cent[("BASE", 0)]
    ax.scatter(*b0, s=100, color="#666", edgecolor="white", zorder=6); ax.annotate("BASE, centroid", b0, (0, -13), textcoords="offset points", fontsize=7, ha="center")
    for s in seeds:
        for nm, colr in (("AESTHETIC", "#6E8B5E"), ("CONTROL", "#C9873A")):
            ax.annotate("", cent[(nm, s)], b0, arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.4, alpha=0.9), zorder=7)
    ax.set_xlabel(f"along the source-set direction CONTROL → AESTHETIC (photographs {gap:.3f} apart)")
    ax.set_ylabel("shared direction of change of the adapters (orthogonal)")
    ax.set_title("A. Overview: photographs, BASE images and the adapters' centroids", loc="left", fontsize=10, pad=10)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    # zoom
    xs_ = [q[0] for q in cent.values()] + [pC[0]]; ys_ = [q[1] for q in cent.values()]
    x0, x1 = min(xs_) - 0.03, max(xs_) + 0.07; y0, y1 = min(ys_) - 0.03, max(ys_) + 0.03
    ax2.axvline(pC[0], color="#333", lw=0.7, ls=":"); ax2.annotate("CONTROL photographs (centroid)", (pC[0], y1), (4, -4), textcoords="offset points", fontsize=7, va="top", color="#333")
    ax2.axvline(b0[0], color="#999", lw=0.5, ls=":")
    ax2.scatter(*b0, s=140, color="#666", edgecolor="white", zorder=6); ax2.annotate("BASE", b0, (0, -14), textcoords="offset points", fontsize=8, ha="center")
    for k, s in enumerate(seeds):
        for nm, colr, side in (("AESTHETIC", "#6E8B5E", 1), ("CONTROL", "#C9873A", -1)):
            q = cent[(nm, s)]
            ax2.annotate("", q, b0, arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.8, alpha=0.9), zorder=7)
            ax2.scatter(*q, s=70, color=colr, edgecolor="white", zorder=8)
            ax2.annotate(f"{nm} · seed {NAME[s]}", q, (12 if side > 0 else -12, 12 - 12 * k), textcoords="offset points", fontsize=7.5, color=colr, ha="left" if side > 0 else "right")
    ax2.annotate("", (pA[0], y0 + 0.004), (b0[0], y0 + 0.004), arrowprops=dict(arrowstyle="<->", color="#333", lw=0.8))
    ax2.annotate(f"gap between the photograph sets: {gap:.3f}  ·  AESTHETIC adapters move about 18% of it, CONTROL about 0%", ((b0[0] + pA[0]) / 2, y0 + 0.006), fontsize=7, ha="center", va="bottom", color="#333")
    ax2.set_xlim(x0, max(x1, pA[0] + 0.02)); ax2.set_ylim(y0, y1)
    ax2.set_xlabel("along the source-set direction")
    ax2.set_title("B. Zoom: every adapter moves up by the same amount; only the AESTHETIC adapters also move towards the AESTHETIC photographs", wrap=True, loc="left", fontsize=10, pad=10)
    fig.text(0.01, 0.005, "Unit-normalised DINOv2 CLS embeddings; centroids over the 192 prompt × seed cells per adapter; arrows from the BASE centroid. Horizontal axis: the frozen direction of the analysis. "
             "Vertical axis: the component of change shared by AESTHETIC and CONTROL; it carries no information about the corpus.", fontsize=6.5, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(args.out, "fig_embedding_shifts.png")); plt.close(fig)
    proj = {("AESTHETIC", s): float(((eA[s] - eB) @ v).mean()) for s in seeds} | {("CONTROL", s): float(((eC[s] - eB) @ v).mean()) for s in seeds}
    along_w = {("AESTHETIC", s): float(((eA[s] - eB) @ w).mean()) for s in seeds} | {("CONTROL", s): float(((eC[s] - eB) @ w).mean()) for s in seeds}

    # ------------------------------------------------------------ report
    L = ["# What is transferred: the 31 descriptors against BASE, and the shifts in embedding space", "",
         "## What was reused", "",
         "- Descriptors: `metrics/covariates_images.csv`, produced by `image_scripts/covariates.py` in one run for all images (photographs, BASE, CONTROL, AESTHETIC, replication adapters). "
         "The BASE scores were already there: nothing was recomputed.",
         "- Pipeline: that of `direction_decomposition.py`, section 4 (the original figure): CLIP attributes as logits, every descriptor standardised on the mean and SD of the 600 photographs, "
         "\"elegant\" excluded, 31 descriptors.", f"- Sample: all {len(cells)} prompt × generation-seed cells ({len(qs)} prompts), as in the original figure, with BASE (192 images), "
         f"AESTHETIC and CONTROL as the mean of the three paired training seeds ({', '.join(f'{NAME[s]} = {s}' for s in seeds)}; {len(seeds) * 192} images per corpus).",
         "- X: AESTHETIC − CONTROL between the two sets of photographs (89 + 89), identical in the three figures. Y: paired difference per cell, averaged over cells (and seeds).",
         "- Intervals: bootstrap over prompts for each descriptor; bootstrap over the 31 descriptors for r; 5,000 samples. No significance test.", "",
         "## Numbers", "",
         "| figure | Y | Pearson r (95% CI) | same sign as the photographs |", "|---|---|---|---|",
         f"| original | AESTHETIC − CONTROL | {rs['A_minus_C'][0]:+.2f} ({rs['A_minus_C'][1]:+.2f} / {rs['A_minus_C'][2]:+.2f}) | {same['A_minus_C']} of 31 |",
         f"| 1 | AESTHETIC − BASE | {rs['A_minus_BASE'][0]:+.2f} ({rs['A_minus_BASE'][1]:+.2f} / {rs['A_minus_BASE'][2]:+.2f}) | {same['A_minus_BASE']} of 31 |",
         f"| 2 | CONTROL − BASE | {rs['CONTROL_minus_BASE'][0]:+.2f} ({rs['CONTROL_minus_BASE'][1]:+.2f} / {rs['CONTROL_minus_BASE'][2]:+.2f}) | {same['CONTROL_minus_BASE']} of 31 |", "",
         f"Zeros in the signs: {zeros}. Algebraic check (A − BASE) − (CONTROL − BASE) = A − CONTROL, descriptor by descriptor: maximum absolute error {max_err:.2e} "
         "(BASE is shared by the two contrasts within each cell, so the identity holds up to floating-point rounding).", "",
         "Mean absolute size of the differences (SD of the corpus): photographs "
         f"{T.training_A_minus_C.abs().mean():.2f}; AESTHETIC − CONTROL {T.generated_A_minus_C.abs().mean():.2f}; AESTHETIC − BASE {T.generated_A_minus_BASE.abs().mean():.2f}; CONTROL − BASE {T.generated_CONTROL_minus_BASE.abs().mean():.2f}.", "",
         "## The four groups of descriptors", ""]
    sT, sA, sC = T.sign_training, T.sign_A_minus_BASE, T.sign_CONTROL_minus_BASE
    groups = [("AESTHETIC moves from BASE in the direction of the training difference, CONTROL does not (or moves the other way)", T[(sA == sT) & (sC != sT)]),
              ("both adapters move from BASE in the direction of the training difference", T[(sA == sT) & (sC == sT)]),
              ("CONTROL moves in the direction of the training difference, AESTHETIC does not", T[(sA != sT) & (sC == sT)]),
              ("neither adapter moves in the direction of the training difference", T[(sA != sT) & (sC != sT)])]
    for title, d in groups:
        L += [f"**{title}** — {len(d)} descriptors: " + (", ".join(f"{r.feature} (photos {r.training_A_minus_C:+.2f}; A−B {r.generated_A_minus_BASE:+.2f}; C−B {r.generated_CONTROL_minus_BASE:+.2f})" for r in d.sort_values('training_A_minus_C', key=abs, ascending=False).itertuples()) or "none"), ""]
    same_dir = T[np.sign(T.generated_A_minus_BASE) == np.sign(T.generated_CONTROL_minus_BASE)]
    L += [f"Descriptors where the two adapters move from BASE in the **same** direction (whatever the training difference): {len(same_dir)} of 31; correlation between the two shifts across descriptors: "
          f"{np.corrcoef(T.generated_A_minus_BASE, T.generated_CONTROL_minus_BASE)[0, 1]:+.2f}.", "",
          "## Embedding space (fig_embedding_shifts.png)", "",
          "| adapter | along the source-set direction (→ AESTHETIC photographs) | along the shared direction of change |", "|---|---|---|"]
    for s in seeds:
        for c_ in ("AESTHETIC", "CONTROL"):
            L.append(f"| {c_} seed {NAME[s]} | {proj[(c_, s)]:+.4f} | {along_w[(c_, s)]:+.4f} |")
    L += ["", f"The gap between the photograph centroids along the direction is {gap:.3f}. The vertical axis is the component of change that AESTHETIC and CONTROL share: it is large for both and says "
          "nothing about the corpus; the difference between the adapters is almost entirely horizontal.", "",
          "## Reading", "", "The descriptors are visual proxies (photographic measures and CLIP zero-shot attributes), not measures of architectural quality or beauty. "
          "The groups above separate what the AESTHETIC adapter does that CONTROL does not from what both fine-tunings do. Interpretation is left to the text of the paper.", ""]
    open(os.path.join(args.out, "feature_base_analysis.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
