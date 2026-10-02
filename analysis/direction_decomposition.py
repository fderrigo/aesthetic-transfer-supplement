"""
ARCH300 — what the source-set direction contains (exploratory; plan: ANALYSIS_PLAN.md, section 9).

    python direction_decomposition.py <export.zip | extracted folder> [--out decomposition_report]

Uses only existing data: the frozen direction and the embeddings (baseline and replication), the relative source-set
affinity of the photographs (embedding_report/source_affinity_dino.csv), the image-derived covariates
(metrics/covariates_images.csv, from image_scripts/covariates.py), the catalogue metadata and the phase-1 mean ratings.
No phase-2 vote and no VLM score is read.

Blocks fixed in advance: photography, framing and scene, architectural content (image-derived), catalogue metadata
(photographs only). Covariate models are fitted on the 422 photographs that belong to neither training set and applied to
the others, so that they cannot learn the AESTHETIC − CONTROL difference itself.
Writes <out>/report_direction_decomposition.md, results_direction_decomposition.csv,
covariate_profile_direction_decomposition.csv.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from embedding_analysis import ridge_cv, ridge_fit_predict, unit

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N_BOOT = 20261007, 5000
PHOTO = ["brightness", "contrast", "saturation", "colorfulness", "warmth", "sharpness", "detail", "sky_brightness", "dark_share", "bright_share"]
FRAMING = ["whole_building", "low_angle", "sunny", "vintage_photo", "real_photo", "people", "cars", "greenery", "water", "urban", "interior"]
CONTENT = ["iconic_design", "complex_form", "curved_organic", "monumental", "contemporary", "glass", "concrete", "wood", "white", "colourful"]
ORIGINAL = {"RUN-AESTHETIC-4": ("AESTHETIC", 1254), "RUN-CONTROL-4": ("CONTROL", 9865)}
EXCLUDED_RUNS = ["REP-AES-S1254-R"]
LABEL = {"photo": "photography", "framing": "framing and scene", "content": "architectural content", "meta": "catalogue metadata"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--out", default=os.path.join(HERE, "decomposition_report"))
    args = ap.parse_args()
    folder = args.export
    if zipfile.is_zipfile(folder):
        folder = tempfile.mkdtemp()
        zipfile.ZipFile(args.export).extractall(folder)
    rd = lambda f: pd.read_csv(os.path.join(folder, f))
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, L = [], []
    add = lambda analysis, quantity, estimate, lo=None, hi=None, n=None: rows.append(dict(analysis=analysis, quantity=quantity, estimate=estimate, ci_low=lo, ci_high=hi, n=n))
    def table(header, body):
        L.extend(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in body] + [""])

    # ------------------------------------------------------------ photographs
    cov = pd.read_csv(os.path.join(HERE, "metrics", "covariates_images.csv")).set_index("id")
    aff = pd.read_csv(os.path.join(HERE, "embedding_report", "source_affinity_dino.csv")).set_index("image_code")
    meta = rd("image_metadata.csv").set_index("image_code").reindex(aff.index)
    codes = list(aff.index)
    assert all(c in cov.index for c in codes), "covariates missing for some photographs"
    img_cols = PHOTO + FRAMING + CONTENT
    def transform(df):
        z = df[PHOTO + FRAMING + CONTENT + ["elegant"]].astype(float).copy()
        for c in FRAMING + CONTENT + ["elegant"]:                      # probabilities → logits
            p = z[c].clip(1e-4, 1 - 1e-4)
            z[c] = np.log(p / (1 - p))
        return z
    Zp_raw = transform(cov.loc[codes])
    mu, sd = Zp_raw.mean(), Zp_raw.std().replace(0, 1)
    Zp = (Zp_raw - mu) / sd                                                # standardised on the 600 photographs
    def top(series, k):
        keep = series.value_counts().index[:k]
        return series.where(series.isin(keep), "other")
    M = pd.concat([pd.get_dummies(top(meta.building_type, 10), prefix="type"), pd.get_dummies(top(meta.architectural_style, 8), prefix="style"),
                   pd.get_dummies(meta.period, prefix="period"), pd.get_dummies(meta.continent, prefix="continent"),
                   (meta.industrial_or_nonindustrial == "industrial").rename("industrial"), meta.year_completed.fillna(meta.year_completed.median()).rename("year")], axis=1).astype(float)
    M = (M - M.mean()) / M.std().replace(0, 1)
    blocks = {"photo": Zp[PHOTO].to_numpy(), "framing": Zp[FRAMING].to_numpy(), "content": Zp[CONTENT].to_numpy(), "meta": M.to_numpy()}
    y = aff.relative_source_set_affinity.to_numpy(float)
    rating = aff.mean_rating.to_numpy(float)
    is_a, is_c = (aff.source_set == "AESTHETIC").to_numpy(), (aff.source_set == "CONTROL").to_numpy()
    other = ~is_a & ~is_c
    X = lambda names: np.column_stack([blocks[n] for n in names])
    sets = {"photography": ["photo"], "framing and scene": ["framing"], "architectural content": ["content"], "catalogue metadata": ["meta"],
            "photography + framing and scene": ["photo", "framing"], "image-derived (three blocks)": ["photo", "framing", "content"], "all four blocks": ["photo", "framing", "content", "meta"]}

    L += ["# ARCH300 — what the source-set direction contains (decomposition)", "",
          f"Exploratory analysis (ANALYSIS_PLAN.md, section 9). {len(codes)} photographs: AESTHETIC {int(is_a.sum())}, CONTROL {int(is_c.sum())}, neither {int(other.sum())}. "
          f"Covariates: {len(PHOTO)} photographic measures, {len(FRAMING)} framing/scene and {len(CONTENT)} architectural-content attributes (CLIP ViT-L/14 zero-shot), {M.shape[1]} catalogue-metadata indicators. "
          f"Seed {SEED}; {N_BOOT} bootstrap samples. \"Accounted for\" is linear and correlational.", ""]

    # ------------------------------------------------------------ 1. how much of the direction each block describes
    def cv_r2(Xb):
        pred, _ = ridge_cv(Xb, y, np.random.default_rng(SEED))
        i = rng.integers(0, len(y), (N_BOOT, len(y)))
        r2 = lambda p, t: 1 - ((p - t) ** 2).sum(-1) / ((t - t.mean(-1, keepdims=True)) ** 2).sum(-1)
        b = r2(pred[i], y[i])
        return float(r2(pred, y)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), pred
    r2 = {k: cv_r2(X(v)) for k, v in sets.items()}
    full = ["photo", "framing", "content", "meta"]
    uniq = {b: r2["all four blocks"][0] - cv_r2(X([x for x in full if x != b]))[0] for b in full}
    L += ["## 1. How much of the direction each block describes", "",
          "Position of each photograph along the direction (relative source-set affinity) predicted from the covariates; cross-validated R² over the 600 photographs (0 = nothing, 1 = everything).", ""]
    body = []
    for k, v in sets.items():
        add("1. cross-validated R² of the affinity", k, r2[k][0], r2[k][1], r2[k][2], len(y))
        u = uniq[v[0]] if len(v) == 1 else None
        if u is not None:
            add("1. unique part (all four blocks minus the three others)", k, u, None, None, len(y))
        body.append([k, f"**{r2[k][0]:.2f}** ({r2[k][1]:.2f} / {r2[k][2]:.2f})", f"{u:+.2f}" if u is not None else "–"])
    table(["covariates", "R² (95% CI)", "unique part"], body)
    eleg = Zp[["elegant"]].to_numpy()
    r_e = cv_r2(eleg)
    add("1. cross-validated R² of the affinity", "CLIP 'elegant vs ugly' alone (a judgement, not in the blocks)", r_e[0], r_e[1], r_e[2], len(y))
    L += [f"CLIP attribute \"elegant building vs ugly building\" alone (a judgement of beauty, kept out of the blocks): R² {r_e[0]:.2f} ({r_e[1]:.2f} / {r_e[2]:.2f}).", ""]

    # ------------------------------------------------------------ 2. how much of the separation each block accounts for
    obs_gap = y[is_a].mean() - y[is_c].mean()
    L += ["## 2. How much of the AESTHETIC − CONTROL separation each block accounts for", "",
          f"The covariate model is fitted on the {int(other.sum())} photographs of neither set and applied to the two sets. Observed gap along the direction: {obs_gap:+.4f}. "
          "Accounted share = gap predicted from the covariates / observed gap.", ""]
    body, share = [], {}
    ia, ic = np.where(is_a)[0], np.where(is_c)[0]
    for k, v in list(sets.items()) + [("CLIP 'elegant vs ugly' alone (not in the blocks)", None)]:
        Xb = eleg if v is None else X(v)
        _, lam = ridge_cv(Xb[other], y[other], np.random.default_rng(SEED))
        pred = ridge_fit_predict(Xb[other], y[other], Xb, lam)
        ba, bc = rng.integers(0, len(ia), (N_BOOT, len(ia))), rng.integers(0, len(ic), (N_BOOT, len(ic)))
        b = (pred[ia][ba].mean(1) - pred[ic][bc].mean(1)) / (y[ia][ba].mean(1) - y[ic][bc].mean(1))
        s_ = (pred[is_a].mean() - pred[is_c].mean()) / obs_gap
        share[k] = s_
        add("2. share of the AESTHETIC − CONTROL gap accounted for (model fitted on the other photographs)", k, s_, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(is_a.sum() + is_c.sum()))
        body.append([k, f"**{s_:.0%}** ({np.percentile(b, 2.5):.0%} / {np.percentile(b, 97.5):.0%})"])
    table(["covariates", "share of the gap accounted for (95% CI)"], body)
    pf, al = share["photography + framing and scene"], share["all four blocks"]
    verdict = ("photography plus framing and scene account for at least half of the separation: the direction is mainly a prior of photographic representation" if pf >= 0.5 else
               "all blocks together account for less than a quarter of the separation: the direction is not reducible to the variables measured here" if al < 0.25 else
               "mixed: the measured variables account for part of the separation, neither most of it nor almost none")
    L += [f"**Reading fixed in advance (9.4):** {verdict} (photography + framing and scene {pf:.0%}; all four blocks {al:.0%}). "
          "A small share only says that these variables do not capture the direction, not that the remainder is an architectural structure.", ""]

    # ------------------------------------------------------------ embeddings and design of the generated images
    direction = np.load(os.path.join(HERE, "baseline", "direction_dinov2.npz"), allow_pickle=False)
    v, gap = direction["cls_v"], float(direction["cls_gap"])
    E = {}
    for f in ("embeddings_dinov2_vitb14.npz", "embeddings_replication_dinov2_vitb14.npz"):
        z = np.load(os.path.join(HERE, "metrics", f), allow_pickle=False)
        e = unit(z["cls"].astype(np.float64))
        E.update({i: e[n] for n, i in enumerate(z["id"])})
    Ep = np.stack([E[c] for c in codes])
    runs = rd("training_runs.csv").set_index("code")
    G = rd("generated_images.csv")
    G["o"] = G.opaque_id.astype(str).str.replace("-", "").str.lower()
    G["cell"] = G.prompt_code + "_" + G.seed.astype(str)
    main_plans = set(G[G.training_run.isin(ORIGINAL)].generation_plan_id)
    def label(r):
        if r.training_run in ORIGINAL and r.generation_plan_id in main_plans:
            return ORIGINAL[r.training_run]
        if isinstance(r.training_run, str) and r.training_run.startswith("REP-") and r.training_run not in EXCLUDED_RUNS:
            return ("AESTHETIC" if str(runs.training_dataset[r.training_run]).startswith("AESTHETIC") else "CONTROL", int(runs.seed[r.training_run]))
        return (None, None)
    lab = G.apply(label, axis=1, result_type="expand")
    G["corpus"], G["tseed"] = lab[0], lab[1]
    G = G[G.corpus.notna()]
    seeds = sorted(int(s) for s in G.tseed.unique())
    name = {s: n for s, n in zip([1254, 9865] + [s for s in seeds if s not in (1254, 9865)], "ABC")}
    cells = sorted(G.cell.unique())
    prompt = np.array([c.rsplit("_", 1)[0] for c in cells])
    ids = {(c, s): G[(G.corpus == c) & (G.tseed == s)].drop_duplicates("cell").set_index("cell").o.reindex(cells) for c in ("AESTHETIC", "CONTROL") for s in seeds}
    assert all(o.notna().all() and o.isin(cov.index).all() for o in ids.values()), "generated images or covariates missing"
    Zg = {k: ((transform(cov.loc[o.to_numpy()]) - mu) / sd) for k, o in ids.items()}
    Eg = {k: np.stack([E[i] for i in o]) for k, o in ids.items()}

    def boot_cells(x):
        qs = np.unique(prompt)
        mean = np.array([x[prompt == q].mean() for q in qs]); w = np.array([(prompt == q).sum() for q in qs])
        i = rng.integers(0, len(qs), (N_BOOT, len(qs)))
        b = (mean[i] * w[i]).sum(1) / w[i].sum(1)
        return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    # ------------------------------------------------------------ 3. residual direction
    Zi = Zp[img_cols].to_numpy()
    def multi_ridge(Ztr, Etr, lam):
        zm, em = Ztr.mean(0), Etr.mean(0)
        A = Ztr - zm
        B = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Etr - em))
        return lambda Z: (Z - zm) @ B                              # predictable part, around the mean
    idx = rng.permutation(np.where(other)[0])
    errs = {}
    for lam in (0.01, 0.1, 1.0, 10.0):
        e_ = 0.0
        for k in range(10):
            te = idx[k::10]; tr = np.setdiff1d(idx, te)
            f = multi_ridge(Zi[tr], Ep[tr], lam)
            e_ += (((Ep[te] - Ep[tr].mean(0)) - f(Zi[te])) ** 2).sum()
        errs[lam] = e_
    lam = min(errs, key=errs.get)
    f = multi_ridge(Zi[other], Ep[other], lam)
    tot = ((Ep[other] - Ep[other].mean(0)) ** 2).sum()
    explained_var = 1 - errs[lam] / tot
    Ep_res = Ep - f(Zi)
    v_res = Ep_res[is_a].mean(0) - Ep_res[is_c].mean(0)
    gap_res = float(np.linalg.norm(v_res))
    u_res = v_res / gap_res
    rho0, rho1 = stats.spearmanr(Ep[other] @ v, rating[other]), stats.spearmanr(Ep_res[other] @ u_res, rating[other])
    add("3. residual direction", "share of the variance of the embeddings predictable from the image-derived covariates (cross-validated, photographs of neither set)", explained_var, None, None, int(other.sum()))
    add("3. residual direction", "length of the direction: original", gap, None, None, None)
    add("3. residual direction", "length of the direction: after removing the predictable part", gap_res, None, None, None)
    add("3. residual direction", "cosine between original and residual direction", float(v @ u_res), None, None, None)
    add("3. residual direction", "direction vs mean rating, photographs of neither set (Spearman): original", rho0.statistic, None, None, int(other.sum()))
    add("3. residual direction", "direction vs mean rating, photographs of neither set (Spearman): residual", rho1.statistic, None, None, int(other.sum()))
    L += ["## 3. The direction after removing what the image-derived covariates predict", "",
          f"The part of every embedding that is linearly predictable from the {len(img_cols)} image-derived covariates (model fitted on the {int(other.sum())} photographs of neither set; "
          f"it predicts {explained_var:.0%} of the variance of the embeddings out of sample) is removed from photographs and generated images, and the direction is computed again.", ""]
    table(["quantity", "original direction", "residual direction"], [
        ["distance between the AESTHETIC and CONTROL centroids", f"{gap:.3f}", f"{gap_res:.3f} ({gap_res / gap:.0%} of the original; cosine with the original {float(v @ u_res):+.2f})"],
        [f"follows the mean rating of the {int(other.sum())} photographs of neither set (Spearman ρ)", f"{rho0.statistic:+.2f}", f"{rho1.statistic:+.2f}"]])
    body = []
    for s in seeds:
        d0 = boot_cells(Eg[("AESTHETIC", s)] @ v - Eg[("CONTROL", s)] @ v)
        ra, rc = Eg[("AESTHETIC", s)] - f(Zg[("AESTHETIC", s)][img_cols].to_numpy()), Eg[("CONTROL", s)] - f(Zg[("CONTROL", s)][img_cols].to_numpy())
        d1 = boot_cells(ra @ u_res - rc @ u_res)
        dp = boot_cells(f(Zg[("AESTHETIC", s)][img_cols].to_numpy()) @ v - f(Zg[("CONTROL", s)][img_cols].to_numpy()) @ v)
        add("3. generated images, AESTHETIC − CONTROL along the original direction", f"training seed {name[s]}", *d0, len(cells))
        add("3. generated images, AESTHETIC − CONTROL along the residual direction", f"training seed {name[s]}", *d1, len(cells))
        add("3. generated images, part of the contrast along the original direction carried by the covariate-predictable part", f"training seed {name[s]}", *dp, len(cells))
        body.append([f"{name[s]} = {s}", f"{d0[0]:+.4f} ({d0[1]:+.4f} / {d0[2]:+.4f}); {d0[0] / gap:.0%} of the gap", f"{dp[0]:+.4f} ({dp[1]:+.4f} / {dp[2]:+.4f}); {dp[0] / d0[0]:.0%} of the contrast",
                     f"**{d1[0]:+.4f}** ({d1[1]:+.4f} / {d1[2]:+.4f}); {d1[0] / gap_res:.0%} of the residual gap"])
    L += ["Generated images, AESTHETIC − CONTROL at equal training seed (192 cells, bootstrap over prompts):", ""]
    table(["training seed", "along the original direction", "of which carried by the covariate-predictable part", "along the residual direction"], body)

    # ------------------------------------------------------------ 4. same variables in the generated images
    prof = []
    for c in img_cols + ["elegant"]:
        src = float(Zp.loc[is_a, c].mean() - Zp.loc[is_c, c].mean())
        per = {s: boot_cells(Zg[("AESTHETIC", s)][c].to_numpy() - Zg[("CONTROL", s)][c].to_numpy()) for s in seeds}
        pooled = boot_cells(np.mean([Zg[("AESTHETIC", s)][c].to_numpy() - Zg[("CONTROL", s)][c].to_numpy() for s in seeds], 0))
        r_dir = float(np.corrcoef(Zp[c], y)[0, 1])
        prof.append(dict(covariate=c, block="judgement (not in the blocks)" if c == "elegant" else LABEL["photo" if c in PHOTO else "framing" if c in FRAMING else "content"],
                         photographs_aesthetic_minus_control=src, correlation_with_direction=r_dir,
                         **{f"generated_seed_{name[s]}": per[s][0] for s in seeds}, generated_mean=pooled[0], generated_ci_low=pooled[1], generated_ci_high=pooled[2],
                         generated_same_sign_three_seeds=len({np.sign(per[s][0]) for s in seeds}) == 1))
    P = pd.DataFrame(prof)
    P.to_csv(os.path.join(args.out, "covariate_profile_direction_decomposition.csv"), index=False)
    Pb = P[P.covariate != "elegant"]
    r_prof, rho_prof = np.corrcoef(Pb.photographs_aesthetic_minus_control, Pb.generated_mean)[0, 1], stats.spearmanr(Pb.photographs_aesthetic_minus_control, Pb.generated_mean).statistic
    b = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(Pb), len(Pb))
        b.append(np.corrcoef(Pb.photographs_aesthetic_minus_control.to_numpy()[i], Pb.generated_mean.to_numpy()[i])[0, 1])
    add("4. profile of the covariates: photographs vs generated images (Pearson r across covariates)", f"{len(Pb)} image-derived covariates", r_prof, float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5)), len(Pb))
    add("4. profile of the covariates: photographs vs generated images (Spearman ρ across covariates)", f"{len(Pb)} image-derived covariates", rho_prof, None, None, len(Pb))
    L += ["## 4. The same variables in the generated images", "",
          "For each covariate: AESTHETIC − CONTROL between the two sets of photographs, and between the generated images at equal training seed (mean of the three seeds; 95% CI over prompts), "
          "in standard deviations of the 600 photographs. Sorted by the size of the difference between the photograph sets.", ""]
    show = P.reindex(P.photographs_aesthetic_minus_control.abs().sort_values(ascending=False).index)
    table(["covariate", "block", "photographs", "generated (95% CI)", "same sign for the three seeds", "same sign as the photographs"],
          [[r.covariate, r.block, f"{r.photographs_aesthetic_minus_control:+.2f}", f"{r.generated_mean:+.2f} ({r.generated_ci_low:+.2f} / {r.generated_ci_high:+.2f})", "yes" if r.generated_same_sign_three_seeds else "no",
            "yes" if np.sign(r.generated_mean) == np.sign(r.photographs_aesthetic_minus_control) else "no"] for r in show.itertuples()])
    agree = int((np.sign(Pb.generated_mean) == np.sign(Pb.photographs_aesthetic_minus_control)).sum())
    L += [f"Across the {len(Pb)} image-derived covariates, the profile of the generated images follows that of the photographs with Pearson r **{r_prof:+.2f}** "
          f"({np.nanpercentile(b, 2.5):+.2f} / {np.nanpercentile(b, 97.5):+.2f}; Spearman ρ {rho_prof:+.2f}); same sign for {agree} of {len(Pb)}. "
          f"Mean size of the differences: photographs {Pb.photographs_aesthetic_minus_control.abs().mean():.2f}, generated images {Pb.generated_mean.abs().mean():.2f} standard deviations.", ""]
    for r in P.itertuples():
        add("4. covariate difference AESTHETIC − CONTROL, photographs (SD of the corpus)", r.covariate, r.photographs_aesthetic_minus_control, None, None, int(is_a.sum() + is_c.sum()))
        add("4. covariate difference AESTHETIC − CONTROL, generated images, mean of the three seeds (SD of the corpus)", r.covariate, r.generated_mean, r.generated_ci_low, r.generated_ci_high, len(cells))

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "results_direction_decomposition.csv"), index=False)
    open(os.path.join(args.out, "report_direction_decomposition.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
