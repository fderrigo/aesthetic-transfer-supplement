"""
ARCH300 — computational analysis of the images with DINOv2 embeddings (exploratory; plan: ANALYSIS_PLAN.md, section 6).

    python embedding_analysis.py <export.zip | extracted folder> <embeddings .npz> [--out embedding_report]

Part A (photographs): what distinguishes, in embedding space, the set selected by the aesthetic consensus (AESTHETIC)
from the matched control set (CONTROL) and from the corpus: centroids, dispersion, distances, bootstrap, permutation tests,
and how well the embeddings predict the phase-1 ratings.
Part B (generated images): for every prompt/seed, the shifts BASE → AESTHETIC and BASE → CONTROL; whether the AESTHETIC
shift goes in the direction observed between the training sets; how that alignment relates to the human votes.

Embeddings: CLS token, L2-normalised (the mean of the patch tokens is used as a sensitivity check). Distances are
Euclidean distances between unit vectors. Everything is computed from the export and the embedding file; fixed seed.
Two named measures:
  relative source-set affinity of an image = mean cosine similarity with the AESTHETIC photographs minus mean cosine
    similarity with the CONTROL photographs (leave-one-out for the photographs that belong to one of the two sets);
  representation transfer score of a triplet = affinity of the AESTHETIC image minus affinity of the CONTROL image
    (same prompt and seed; the BASE image cancels out).
Primary analysis of the generated images: the active ones (seen by the participants); sensitivity: all the generated ones.
Writes <out>/report_dino.md, <out>/results_dino.csv, <out>/source_affinity_dino.csv (one row per photograph) and
<out>/triplet_transfer_dino.csv (one row per prompt/seed).
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261001
N_PERM, N_BOOT = 10000, 2000


def fmt_p(p: float) -> str:
    return "–" if p is None or not math.isfinite(p) else ("< .001" if p < 0.001 else f"{p:.3f}".replace("0.", "."))


class Report:
    def __init__(self):
        self.lines, self.rows = [], []

    def h(self, t, level=2): self.lines += ["", "#" * level + " " + t, ""]
    def p(self, t=""): self.lines.append(t)

    def table(self, header, rows):
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join("---" for _ in header) + "|")
        self.lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]

    def add(self, part, analysis, estimate, lo=None, hi=None, p=None, n=None, emb="cls"):
        self.rows.append(dict(part=part, analysis=analysis, embedding=emb, estimate=estimate, ci_low=lo, ci_high=hi, p=p, n=n))


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(0) - b.mean(0)))


def dispersion(a: np.ndarray) -> float:
    """Mean Euclidean distance of the images from the centroid of their group."""
    return float(np.linalg.norm(a - a.mean(0), axis=1).mean())


def auc(score: np.ndarray, label: np.ndarray) -> float:
    r = stats.rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- ridge regression (dual form, centred)
def ridge_fit_predict(Xtr, ytr, Xte, lam):
    mu, my = Xtr.mean(0), ytr.mean()
    A, B = Xtr - mu, Xte - mu
    K = A @ A.T
    alpha = np.linalg.solve(K + lam * np.trace(K) / len(K) * np.eye(len(K)), ytr - my)
    return B @ (A.T @ alpha) + my


def ridge_cv(X, y, rng, lams=(1e-3, 1e-2, 1e-1, 1.0), folds=10, lam=None):
    """Out-of-fold predictions; the penalty is chosen inside each training fold (inner 5-fold) unless given."""
    idx = rng.permutation(len(y))
    pred = np.empty(len(y))
    chosen = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        best = lam
        if best is None:
            inner = rng.permutation(tr)
            score = []
            for l in lams:
                p = np.empty(len(tr))
                pos = {v: i for i, v in enumerate(tr)}
                for g in range(5):
                    ite = inner[g::5]
                    itr = np.setdiff1d(inner, ite)
                    p[[pos[v] for v in ite]] = ridge_fit_predict(X[itr], y[itr], X[ite], l)
                score.append(-np.mean((p - y[tr]) ** 2))
            best = lams[int(np.argmax(score))]
        chosen.append(best)
        pred[te] = ridge_fit_predict(X[tr], y[tr], X[te], best)
    return pred, float(stats.mode(chosen, keepdims=False).mode)


# ---------------------------------------------------------------- data
def load_export(path: str) -> str:
    if zipfile.is_zipfile(path):
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(path).extractall(tmp)
        return tmp
    return path


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def analyse(folder: str, emb_path: str, which: str, rep: Report, full: bool, out: str | None = None):
    rng = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED + 1)      # additions (A4, B5, B6): a separate stream, so the earlier results do not change
    z = np.load(emb_path, allow_pickle=False)
    E = unit(z[which].astype(np.float64))
    kind, ids = z["kind"], z["id"]
    photo = {i: E[n] for n, i in enumerate(ids) if kind[n] == "source"}
    gen = {i: E[n] for n, i in enumerate(ids) if kind[n] == "generated"}
    rd = lambda f: pd.read_csv(os.path.join(folder, f))

    # ================================================================ Part A — photographs
    ratings = rd("posttraining_ratings.csv")
    used = ratings.groupby("condition_code").training_dataset.agg(lambda x: x.dropna().iloc[0] if x.notna().any() else None).to_dict()
    def members(file, cond):
        d = rd(file)
        code = used.get(cond)
        if code is not None and (d.dataset_code == code).any():
            return set(d[d.dataset_code == code].image_code)
        d = d[as_bool(d.is_frozen)]
        return set(d[d.version == d.version.max()].image_code)
    aes, ctl = members("aesthetic_dataset.csv", "AESTHETIC"), members("control_dataset.csv", "CONTROL")
    summ = rd("pretraining_summary.csv").set_index("image_code")
    meta = rd("image_metadata.csv").set_index("image_code")
    codes = sorted(c for c in photo if c in summ.index)
    X = np.stack([photo[c] for c in codes])
    y = summ.loc[codes, "mean"].to_numpy(float)
    is_a, is_c = np.array([c in aes for c in codes]), np.array([c in ctl for c in codes])
    A, C, rest_a = X[is_a], X[is_c], X[~is_a]
    na, nc = int(is_a.sum()), int(is_c.sum())

    d_ac, d_a_rest, d_c_other = dist(A, C), dist(A, rest_a), dist(C, X[~is_a & ~is_c])
    disp = {"AESTHETIC": dispersion(A), "CONTROL": dispersion(C), "corpus": dispersion(X)}
    pooled = math.sqrt((disp["AESTHETIC"] ** 2 + disp["CONTROL"] ** 2) / 2)

    # bootstrap (images resampled within each group)
    bd, bdd = [], []
    for _ in range(N_BOOT):
        a, c = A[rng.integers(0, na, na)], C[rng.integers(0, nc, nc)]
        bd.append(dist(a, c)); bdd.append(dispersion(a) - dispersion(c))
    ci_d, ci_dd = np.percentile(bd, [2.5, 97.5]), np.percentile(bdd, [2.5, 97.5])

    # permutation 1: AESTHETIC against random subsets of the corpus of the same size
    obs = d_a_rest
    null = np.empty(N_PERM)
    n = len(X)
    for i in range(N_PERM):
        m = np.zeros(n, bool); m[rng.choice(n, na, replace=False)] = True
        null[i] = dist(X[m], X[~m])
    p_subset = (1 + (null >= obs).sum()) / (1 + N_PERM)
    # permutation 2: AESTHETIC / CONTROL labels exchanged (free, and within building type — CONTROL is matched on type and style)
    sub = np.where(is_a | is_c)[0]
    lab = is_a[sub]
    types = meta.reindex([codes[i] for i in sub]).building_type.fillna("?").to_numpy()
    obs_dd = disp["AESTHETIC"] - disp["CONTROL"]
    nfree, nstrat, ndd = np.empty(N_PERM), np.empty(N_PERM), np.empty(N_PERM)
    strata = [np.where(types == t)[0] for t in np.unique(types)]
    mixed = sum(1 for s in strata if 0 < lab[s].sum() < len(s))
    for i in range(N_PERM):
        l = rng.permutation(lab)
        nfree[i] = dist(X[sub][l], X[sub][~l]); ndd[i] = dispersion(X[sub][l]) - dispersion(X[sub][~l])
        l2 = lab.copy()
        for s in strata:
            l2[s] = rng.permutation(lab[s])
        nstrat[i] = dist(X[sub][l2], X[sub][~l2])
    p_free = (1 + (nfree >= d_ac).sum()) / (1 + N_PERM)
    p_strat = (1 + (nstrat >= d_ac).sum()) / (1 + N_PERM)
    p_disp = (1 + (np.abs(ndd) >= abs(obs_dd)).sum()) / (1 + N_PERM)

    # the "aesthetic direction": from the CONTROL centroid to the AESTHETIC centroid
    v = A.mean(0) - C.mean(0)
    gap = float(np.linalg.norm(v))
    v = v / gap
    other = ~is_a & ~is_c
    rho_dir = stats.spearmanr(X[other] @ v, y[other])
    # do the embeddings predict the ratings? (ridge, nested 10-fold CV) and do they separate the two sets?
    pred, lam = ridge_cv(X, y, rng)
    r_cv = float(np.corrcoef(pred, y)[0, 1]); r2_cv = 1 - float(((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    perm_r = [np.corrcoef(ridge_cv(X, rng.permutation(y), rng, lam=lam)[0], y)[0, 1] for _ in range(200)] if full else []
    yl = np.where(lab, 1.0, -1.0)
    predl, laml = ridge_cv(X[sub], yl, rng)
    auc_cv = auc(predl, lab)
    perm_auc = []
    if full:
        for _ in range(200):
            l = rng.permutation(lab)
            perm_auc.append(auc(ridge_cv(X[sub], np.where(l, 1.0, -1.0), rng, lam=laml)[0], l))

    # ---- relative source-set affinity (leave-one-out for the members of the two sets)
    def loo_affinity(Xs, l):
        n1, n0 = int(l.sum()), int((~l).sum())
        c1, c0 = Xs @ Xs[l].mean(0), Xs @ Xs[~l].mean(0)
        c1[l] = (n1 * c1[l] - 1) / (n1 - 1); c0[~l] = (n0 * c0[~l] - 1) / (n0 - 1)      # unit vectors: x·x = 1
        return c1, c0
    simA, simC = X @ A.mean(0), X @ C.mean(0)
    s1, s0 = loo_affinity(X[sub], lab)
    simA[sub], simC[sub] = s1, s0
    aff = simA - simC
    aff_gap = float(aff[is_a].mean() - aff[is_c].mean())
    b = []
    for _ in range(N_BOOT):
        b.append(aff[is_a][rng2.integers(0, na, na)].mean() - aff[is_c][rng2.integers(0, nc, nc)].mean())
    ci_aff = np.percentile(b, [2.5, 97.5])
    null_aff = np.empty(N_PERM)
    for i in range(N_PERM):
        l = rng2.permutation(lab)
        q1, q0 = loo_affinity(X[sub], l)
        null_aff[i] = (q1 - q0)[l].mean() - (q1 - q0)[~l].mean()
    p_aff = (1 + (null_aff >= aff_gap).sum()) / (1 + N_PERM)
    auc_aff = auc(aff[sub], lab)
    rho_aff = stats.spearmanr(aff, y)
    rep.add("A", "relative source-set affinity (leave-one-out), mean AESTHETIC − mean CONTROL", aff_gap, ci_aff[0], ci_aff[1], p_aff, na + nc, which)
    rep.add("A", "relative source-set affinity (leave-one-out): AUC AESTHETIC vs CONTROL", auc_aff, None, None, None, na + nc, which)
    rep.add("A", "relative source-set affinity vs mean rating, all photographs (Spearman)", rho_aff.statistic, None, None, rho_aff.pvalue, n, which)
    for k_, m_ in (("AESTHETIC", is_a), ("CONTROL", is_c), ("neither set", other)):
        rep.add("A", f"relative source-set affinity, mean of the photographs: {k_}", aff[m_].mean(), None, None, None, int(m_.sum()), which)
    if full and out:
        pd.DataFrame(dict(image_code=codes, source_set=np.where(is_a, "AESTHETIC", np.where(is_c, "CONTROL", "neither")),
                          building_type=meta.reindex(codes).building_type.to_numpy(), mean_rating=y,
                          similarity_to_aesthetic=simA, similarity_to_control=simC, relative_source_set_affinity=aff)).to_csv(os.path.join(out, "source_affinity_dino.csv"), index=False)

    tag = "" if which == "cls" else f" [{which}]"
    rep.add("A", "centroid distance AESTHETIC–CONTROL", d_ac, ci_d[0], ci_d[1], p_free, na + nc, which)
    rep.add("A", "centroid distance AESTHETIC–CONTROL, permutation within building type", d_ac, None, None, p_strat, na + nc, which)
    rep.add("A", "centroid distance AESTHETIC–rest of corpus (vs random subsets)", d_a_rest, None, None, p_subset, n, which)
    rep.add("A", "standardised distance AESTHETIC–CONTROL (distance / pooled dispersion)", d_ac / pooled, None, None, None, na + nc, which)
    rep.add("A", "dispersion AESTHETIC − CONTROL", obs_dd, ci_dd[0], ci_dd[1], p_disp, na + nc, which)
    p_r = (1 + sum(x >= r_cv for x in perm_r)) / (1 + len(perm_r)) if perm_r else None
    p_auc = (1 + sum(x >= auc_cv for x in perm_auc)) / (1 + len(perm_auc)) if perm_auc else None
    rep.add("A", "ratings predicted from embeddings, cross-validated r", r_cv, None, None, p_r, n, which)
    rep.add("A", "AESTHETIC vs CONTROL separable from embeddings, cross-validated AUC", auc_cv, None, None, p_auc, na + nc, which)
    rep.add("A", "direction CONTROL→AESTHETIC vs rating, photos in neither set (Spearman)", rho_dir.statistic, None, None, rho_dir.pvalue, int(other.sum()), which)

    if full:
        rep.h("Part A — What distinguishes the photographs selected by the consensus")
        rep.p(f"{n} corpus photographs; AESTHETIC {na}, CONTROL {nc} (the sets the models were trained on). Embedding: DINOv2 "
              f"({str(z['model'])}), {which.upper()} token, L2-normalised, dimension {X.shape[1]}. Distances are Euclidean between unit vectors "
              f"(two unrelated photographs are typically ≈ {np.linalg.norm(X[rng.integers(0, n, 2000)] - X[rng.integers(0, n, 2000)], axis=1).mean():.2f} apart).")
        rep.p("**A1. Centroids and distances.**")
        rep.table(["comparison", "distance between centroids", "reference", "p"], [
            ["AESTHETIC – CONTROL", f"**{d_ac:.3f}** (bootstrap 95% CI {ci_d[0]:.3f} / {ci_d[1]:.3f})", f"labels exchanged at random: {nfree.mean():.3f} (95th pct {np.percentile(nfree, 95):.3f})", fmt_p(p_free)],
            ["AESTHETIC – CONTROL, labels exchanged within building type", f"{d_ac:.3f}", f"{nstrat.mean():.3f} (95th pct {np.percentile(nstrat, 95):.3f}); {mixed} types contain both sets", fmt_p(p_strat)],
            ["AESTHETIC – rest of the corpus", f"**{d_a_rest:.3f}**", f"random subsets of {na} photographs: {null.mean():.3f} (95th pct {np.percentile(null, 95):.3f})", fmt_p(p_subset)],
            ["CONTROL – photographs in neither set", f"{d_c_other:.3f}", "descriptive", "–"]])
        rep.p(f"Standardised distance AESTHETIC – CONTROL = distance / pooled dispersion = **{d_ac / pooled:.2f}** "
              "(how far apart the two centroids are compared with how spread each set is; the two clouds overlap almost entirely when this is well below 1).")
        rep.p("**A2. Dispersion** (mean distance of the photographs from the centroid of their set):")
        rep.table(["set", "photographs", "dispersion"], [[k, {"AESTHETIC": na, "CONTROL": nc, "corpus": n}[k], f"{v_:.3f}"] for k, v_ in disp.items()])
        rep.p(f"AESTHETIC − CONTROL = {obs_dd:+.3f} (bootstrap 95% CI {ci_dd[0]:+.3f} / {ci_dd[1]:+.3f}; permutation p = {fmt_p(p_disp)}). "
              "Negative = the selected photographs are more alike than the control ones.")
        rep.p("**A3. Is the difference usable?**")
        rep.table(["question", "result", "p (permutation)"], [
            ["Can the two sets be told apart from the embedding alone? (ridge classifier, 10-fold CV)", f"AUC **{auc_cv:.2f}** (0.5 = chance, 1 = perfect)", fmt_p(p_auc)],
            [f"Do the embeddings predict the mean rating of a photograph? (ridge, nested 10-fold CV, {n} photographs)", f"r **{r_cv:.2f}**, R² {r2_cv:.2f}", fmt_p(p_r)],
            [f"Does the direction CONTROL → AESTHETIC track the ratings of the {int(other.sum())} photographs in neither set?", f"Spearman ρ **{rho_dir.statistic:+.2f}**", fmt_p(rho_dir.pvalue)]])
        rep.p(f"The direction from the CONTROL centroid to the AESTHETIC centroid (length {gap:.3f}) is the \"aesthetic direction\" used in Part B. "
              "It is defined on the photographs only and never sees the generated images or the phase-2 votes.")
        rep.p("**A4. Relative source-set affinity.** Definition: for an image, the mean cosine similarity with the AESTHETIC photographs minus the mean cosine "
              "similarity with the CONTROL photographs. Positive = the image resembles the selected set more than the control set. For a photograph that belongs "
              "to one of the two sets the photograph itself is left out of its own set (leave-one-out), otherwise it would resemble its set by construction. "
              f"With unit vectors the affinity equals the position along the aesthetic direction multiplied by its length ({gap:.3f}), so it is the same measure used in Part B.")
        rep.table(["photographs", "n", "mean affinity", "share with positive affinity"], [
            [k_, int(m_.sum()), f"{aff[m_].mean():+.4f}", f"{(aff[m_] > 0).mean():.0%}"] for k_, m_ in (("AESTHETIC", is_a), ("CONTROL", is_c), ("neither set", other))])
        rep.p(f"AESTHETIC − CONTROL = **{aff_gap:+.4f}** (bootstrap 95% CI {ci_aff[0]:+.4f} / {ci_aff[1]:+.4f}; labels exchanged, leave-one-out repeated each time: "
              f"mean {null_aff.mean():+.4f}, p {fmt_p(p_aff)}). The affinity alone tells the two sets apart with AUC **{auc_aff:.2f}** (leave-one-out) and follows "
              f"the mean rating of the {n} photographs with Spearman ρ **{rho_aff.statistic:+.2f}** (p {fmt_p(rho_aff.pvalue)}). One value per photograph: `source_affinity_dino.csv`.")

    # ================================================================ Part B — generated images
    G = rd("generated_images.csv")
    G = G[as_bool(G.is_active)].assign(o=lambda d: d.opaque_id.astype(str).str.replace("-", "").str.lower())
    G = G[G.o.isin(gen)]
    G["trip"] = G.prompt_code + "_" + G.seed.astype(str)
    piv = G.pivot(index="trip", columns="condition_code", values="o").dropna()
    prompt = piv.index.str.rsplit("_", n=1).str[0].to_numpy()
    eA, eB, eC = (np.stack([gen[o] for o in piv[c]]) for c in ("AESTHETIC", "BASE", "CONTROL"))
    dA, dC = eA - eB, eC - eB
    pa, pc = dA @ v, dC @ v
    diff = pa - pc
    cosA = (unit(dA) @ v); cosC = (unit(dC) @ v)
    prompts = np.unique(prompt)
    by_prompt = lambda x: np.array([x[prompt == q].mean() for q in prompts])

    def cluster_ci(x):
        m = by_prompt(x); w = np.array([(prompt == q).sum() for q in prompts])
        b = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(prompts), len(prompts))
            b.append((m[i] * w[i]).sum() / w[i].sum())
        return np.percentile(b, [2.5, 97.5])

    def sign_flip_p(x):
        m = by_prompt(x); obs_ = abs(m.mean())
        s = rng.choice([-1.0, 1.0], size=(N_PERM, len(m)))
        return (1 + (np.abs((s * m).mean(1)) >= obs_).sum()) / (1 + N_PERM)

    ci_a, ci_c, ci_diff = cluster_ci(pa), cluster_ci(pc), cluster_ci(diff)
    p_diff, p_pa, p_pc = sign_flip_p(diff), sign_flip_p(pa), sign_flip_p(pc)
    rep.add("B", "shift BASE→AESTHETIC along the aesthetic direction", pa.mean(), ci_a[0], ci_a[1], p_pa, len(pa), which)
    rep.add("B", "shift BASE→CONTROL along the aesthetic direction", pc.mean(), ci_c[0], ci_c[1], p_pc, len(pc), which)
    rep.add("B", "AESTHETIC − CONTROL shift along the aesthetic direction", diff.mean(), ci_diff[0], ci_diff[1], p_diff, len(diff), which)
    rep.add("B", "embedding transfer ratio (AESTHETIC − CONTROL shift / gap between the training sets)", diff.mean() / gap, ci_diff[0] / gap, ci_diff[1] / gap, None, len(diff), which)

    # each fine-tuned model towards its own training set (from the corpus centroid)
    uA, uC = unit(A.mean(0) - X.mean(0)), unit(C.mean(0) - X.mean(0))
    own = {("AESTHETIC", "AESTHETIC"): (dA @ uA), ("AESTHETIC", "CONTROL"): (dA @ uC), ("CONTROL", "AESTHETIC"): (dC @ uA), ("CONTROL", "CONTROL"): (dC @ uC)}
    spec = (own[("AESTHETIC", "AESTHETIC")] - own[("AESTHETIC", "CONTROL")]) - (own[("CONTROL", "AESTHETIC")] - own[("CONTROL", "CONTROL")])
    ci_spec, p_spec = cluster_ci(spec), sign_flip_p(spec)
    rep.add("B", "specificity: each model moves towards its own training set more than towards the other", spec.mean(), ci_spec[0], ci_spec[1], p_spec, len(spec), which)
    nA, nC = np.linalg.norm(dA, axis=1), np.linalg.norm(dC, axis=1)
    cos_ac = (unit(dA) * unit(dC)).sum(1)

    # ---- human votes (exclusions applied; ratings centred on each participant's mean)
    R = ratings[~as_bool(ratings.excluded_from_analysis)].copy()
    R["c"] = R.score - R.groupby("participant_id").score.transform("mean")
    R["trip"] = R.prompt_code + "_" + R.seed.astype(str)
    rt = R.groupby(["trip", "condition_code"]).c.mean().unstack().reindex(piv.index)
    vote_diff = (rt["AESTHETIC"] - rt["CONTROL"]).to_numpy()
    T = rd("pairwise_trials.csv"); T = T[~as_bool(T.excluded_from_analysis) & T.selected_side.isin(["Left", "Right"])]
    T = T[T.left_condition.isin(["AESTHETIC", "CONTROL"]) & T.right_condition.isin(["AESTHETIC", "CONTROL"])]
    T["trip"] = T.prompt_code + "_" + T.seed.astype(str)
    share = (T.winner_condition == "AESTHETIC").groupby(T.trip).mean().reindex(piv.index).to_numpy()

    def spearman_ci(x, yv):
        ok = np.isfinite(x) & np.isfinite(yv)
        x, yv, pr = x[ok], yv[ok], prompt[ok]
        r = stats.spearmanr(x, yv)
        b = []
        for _ in range(N_BOOT):
            q = prompts[rng.integers(0, len(prompts), len(prompts))]
            i = np.concatenate([np.where(pr == k)[0] for k in q])
            if len(i) > 3: b.append(stats.spearmanr(x[i], yv[i]).statistic)
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        return r.statistic, lo, hi, r.pvalue, int(ok.sum())

    s_vote, s_share = spearman_ci(diff, vote_diff), spearman_ci(diff, share)
    rep.add("B", "alignment difference vs rating difference AESTHETIC − CONTROL, per triplet (Spearman)", *s_vote, emb=which)
    rep.add("B", "alignment difference vs share of AESTHETIC choices, per triplet (Spearman)", *s_share, emb=which)
    # per image, within the triplet: position along the direction against the rating
    proj = np.stack([eA @ v, eB @ v, eC @ v], axis=1)
    rate = rt[["AESTHETIC", "BASE", "CONTROL"]].to_numpy()
    ok = np.isfinite(rate).all(1)
    wp, wr = (proj - proj.mean(1, keepdims=True))[ok].ravel(), (rate - rate.mean(1, keepdims=True))[ok].ravel()
    pr3 = np.repeat(prompt[ok], 3)
    r_within = float(np.corrcoef(wp, wr)[0, 1])
    b = []
    for _ in range(N_BOOT):
        q = prompts[rng.integers(0, len(prompts), len(prompts))]
        i = np.concatenate([np.where(pr3 == k)[0] for k in q])
        b.append(np.corrcoef(wp[i], wr[i])[0, 1])
    ci_w = np.percentile(b, [2.5, 97.5])
    rep.add("B", "position along the aesthetic direction vs rating, within triplet (Pearson)", r_within, ci_w[0], ci_w[1], None, int(ok.sum()) * 3, which)
    # the rating predictor learned on the photographs, applied to the generated images
    all_gen = np.concatenate([eA, eB, eC])
    pg = ridge_fit_predict(X, y, all_gen, lam).reshape(3, -1).T
    hum = rate
    okg = np.isfinite(hum).all(1)
    r_pred = stats.spearmanr(pg[okg].ravel(), hum[okg].ravel())
    wpg = (pg - pg.mean(1, keepdims=True))[okg].ravel()
    r_pred_w = float(np.corrcoef(wpg, wr)[0, 1])
    rep.add("B", "rating predicted by the photo-trained model vs human rating of the generated images (Spearman)", r_pred.statistic, None, None, r_pred.pvalue, int(okg.sum()) * 3, which)
    rep.add("B", "same, within triplet (Pearson)", r_pred_w, None, None, None, int(okg.sum()) * 3, which)
    pred_gain = (pg[:, 0] - pg[:, 2])
    ci_pg, p_pg = cluster_ci(pred_gain), sign_flip_p(pred_gain)
    rep.add("B", "predicted rating AESTHETIC − CONTROL (photo-trained model)", pred_gain.mean(), ci_pg[0], ci_pg[1], p_pg, len(pred_gain), which)

    # ---- representation transfer score per triplet, amount separated from direction
    score = gap * diff                                   # affinity(AESTHETIC image) − affinity(CONTROL image)
    ci_score = cluster_ci(score)
    rep.add("B", "representation transfer score per triplet (affinity AESTHETIC image − affinity CONTROL image), mean", score.mean(), ci_score[0], ci_score[1], p_diff, len(score), which)
    mbar = (nA + nC) / 2
    part_dir, part_amt = mbar * (cosA - cosC), (nA - nC) * (cosA + cosC) / 2      # exact: part_dir + part_amt = pa − pc
    mag, cdf = nA - nC, cosA - cosC
    ci_mag, p_mag, ci_cos, p_cos = cluster_ci(mag), sign_flip_p(mag), cluster_ci(cdf), sign_flip_p(cdf)
    ci_pd, p_pd, ci_pm, p_pm = cluster_ci(part_dir), sign_flip_p(part_dir), cluster_ci(part_amt), sign_flip_p(part_amt)
    s_mag_v, s_cos_v = spearman_ci(mag, vote_diff), spearman_ci(cdf, vote_diff)
    s_mag_s, s_cos_s = spearman_ci(mag, share), spearman_ci(cdf, share)
    rep.add("B", "amount of shift ‖Δ‖, AESTHETIC − CONTROL", mag.mean(), ci_mag[0], ci_mag[1], p_mag, len(mag), which)
    rep.add("B", "direction of shift (cosine with the aesthetic direction), AESTHETIC − CONTROL", cdf.mean(), ci_cos[0], ci_cos[1], p_cos, len(cdf), which)
    rep.add("B", "AESTHETIC − CONTROL shift: part due to direction (equal amount)", part_dir.mean(), ci_pd[0], ci_pd[1], p_pd, len(diff), which)
    rep.add("B", "AESTHETIC − CONTROL shift: part due to amount (equal direction)", part_amt.mean(), ci_pm[0], ci_pm[1], p_pm, len(diff), which)
    rep.add("B", "amount difference vs rating difference AESTHETIC − CONTROL, per triplet (Spearman)", *s_mag_v, emb=which)
    rep.add("B", "direction difference vs rating difference AESTHETIC − CONTROL, per triplet (Spearman)", *s_cos_v, emb=which)
    rep.add("B", "amount difference vs share of AESTHETIC choices, per triplet (Spearman)", *s_mag_s, emb=which)
    rep.add("B", "direction difference vs share of AESTHETIC choices, per triplet (Spearman)", *s_cos_s, emb=which)

    # ---- sensitivity: all the generated images, active or not (no human votes for the inactive ones)
    G0 = rd("generated_images.csv").assign(o=lambda d: d.opaque_id.astype(str).str.replace("-", "").str.lower())
    G0 = G0[G0.o.isin(gen)]
    G0["trip"] = G0.prompt_code + "_" + G0.seed.astype(str)
    pivF = G0.pivot(index="trip", columns="condition_code", values="o").dropna()
    actF = as_bool(G0.is_active).groupby(G0.trip).all().reindex(pivF.index).to_numpy()
    promptF = pivF.index.str.rsplit("_", n=1).str[0].to_numpy()
    fA, fB, fC = (np.stack([gen[o] for o in pivF[c]]) for c in ("AESTHETIC", "BASE", "CONTROL"))
    gA, gC = fA - fB, fC - fB

    def summary(mask):
        pr = promptF[mask]; qs = np.unique(pr)
        def ci_p(x):
            m = np.array([x[pr == q].mean() for q in qs]); w = np.array([(pr == q).sum() for q in qs])
            i = rng2.integers(0, len(qs), (N_BOOT, len(qs)))
            lo, hi = np.percentile((m[i] * w[i]).sum(1) / w[i].sum(1), [2.5, 97.5])
            sg = rng2.choice([-1.0, 1.0], size=(N_PERM, len(m)))
            return x.mean(), lo, hi, (1 + (np.abs((sg * m).mean(1)) >= abs(m.mean())).sum()) / (1 + N_PERM)
        a, c = gA[mask], gC[mask]
        sp = (a @ uA - a @ uC) - (c @ uA - c @ uC)
        return dict(n=int(mask.sum()), prompts=len(qs), pa=ci_p(a @ v), pc=ci_p(c @ v), diff=ci_p(a @ v - c @ v), spec=ci_p(sp), pos=int(((a @ v - c @ v) > 0).sum()),
                    nA=float(np.linalg.norm(a, axis=1).mean()), nC=float(np.linalg.norm(c, axis=1).mean()),
                    cosA=float((unit(a) @ v).mean()), cosC=float((unit(c) @ v).mean()), cos_ac=float((unit(a) * unit(c)).sum(1).mean()))
    S_all, S_in = summary(np.ones(len(pivF), bool)), (summary(~actF) if (~actF).sum() >= 5 else None)
    n_img_all, n_img_act = len(G0), int(as_bool(G0.is_active).sum())
    for lbl, S in (("all generated images", S_all), ("triplets with an image not shown to the participants", S_in)):
        if S is None: continue
        rep.add("B-sens", f"{lbl}: shift BASE→AESTHETIC along the aesthetic direction", *S["pa"], S["n"], which)
        rep.add("B-sens", f"{lbl}: shift BASE→CONTROL along the aesthetic direction", *S["pc"], S["n"], which)
        rep.add("B-sens", f"{lbl}: AESTHETIC − CONTROL shift along the aesthetic direction", *S["diff"], S["n"], which)
        rep.add("B-sens", f"{lbl}: embedding transfer ratio", S["diff"][0] / gap, S["diff"][1] / gap, S["diff"][2] / gap, None, S["n"], which)
        rep.add("B-sens", f"{lbl}: specificity", *S["spec"], S["n"], which)
        rep.add("B-sens", f"{lbl}: amount of shift ‖Δ_A‖", S["nA"], None, None, None, S["n"], which)
        rep.add("B-sens", f"{lbl}: amount of shift ‖Δ_C‖", S["nC"], None, None, None, S["n"], which)
    if full and out:
        nt = T.groupby("trip").size()
        pd.DataFrame(dict(
            prompt_code=promptF, seed=pivF.index.str.rsplit("_", n=1).str[1], all_three_shown_to_participants=actF,
            affinity_aesthetic=gap * (fA @ v), affinity_base=gap * (fB @ v), affinity_control=gap * (fC @ v),
            representation_transfer_score=gap * (gA @ v - gC @ v), transfer_ratio=(gA @ v - gC @ v) / gap,
            shift_amount_aesthetic=np.linalg.norm(gA, axis=1), shift_amount_control=np.linalg.norm(gC, axis=1),
            shift_cosine_aesthetic=unit(gA) @ v, shift_cosine_control=unit(gC) @ v,
            shift_projection_aesthetic=gA @ v, shift_projection_control=gC @ v, cosine_between_shifts=(unit(gA) * unit(gC)).sum(1),
            rating_difference_aesthetic_minus_control=(rt["AESTHETIC"] - rt["CONTROL"]).reindex(pivF.index).to_numpy(),
            share_aesthetic_chosen=pd.Series(share, index=piv.index).reindex(pivF.index).to_numpy(),
            decisive_comparisons=nt.reindex(pivF.index).to_numpy())).to_csv(os.path.join(out, "triplet_transfer_dino.csv"), index=False)

    if full:
        rep.h("Part B — Shifts of the generated images")
        rep.p(f"{len(piv)} complete triplets ({len(prompts)} prompts) of active generated images. For each prompt/seed: Δ_A = AESTHETIC − BASE and "
              "Δ_C = CONTROL − BASE (embeddings). Intervals: bootstrap over prompts; p values: sign-flip permutation of the prompt means.")
        rep.p("**B1. Is the AESTHETIC shift coherent with the one observed between the training sets?** Projection on the aesthetic direction "
              f"(unit vector; the two training sets are {gap:.3f} apart along it):")
        rep.table(["shift", "mean projection", "95% CI", "p", "mean cosine with the direction"], [
            ["BASE → AESTHETIC", f"{pa.mean():+.4f}", f"{ci_a[0]:+.4f} / {ci_a[1]:+.4f}", fmt_p(p_pa), f"{cosA.mean():+.3f}"],
            ["BASE → CONTROL", f"{pc.mean():+.4f}", f"{ci_c[0]:+.4f} / {ci_c[1]:+.4f}", fmt_p(p_pc), f"{cosC.mean():+.3f}"],
            ["**AESTHETIC − CONTROL**", f"**{diff.mean():+.4f}**", f"{ci_diff[0]:+.4f} / {ci_diff[1]:+.4f}", fmt_p(p_diff), f"{(cosA - cosC).mean():+.3f}"]])
        rep.p(f"Embedding transfer ratio = (AESTHETIC − CONTROL shift) / (gap between the training sets) = **{diff.mean() / gap:.0%}** "
              f"(95% CI {ci_diff[0] / gap:.0%} / {ci_diff[1] / gap:.0%}). {int((diff > 0).sum())} of {len(diff)} triplets have a positive difference.")
        rep.p("**B2. Does each model move towards its own training set?** Mean projection of the shift on the direction from the corpus centroid to each training set:")
        rep.table(["shift", "towards AESTHETIC set", "towards CONTROL set"], [
            ["BASE → AESTHETIC", f"{own[('AESTHETIC', 'AESTHETIC')].mean():+.4f}", f"{own[('AESTHETIC', 'CONTROL')].mean():+.4f}"],
            ["BASE → CONTROL", f"{own[('CONTROL', 'AESTHETIC')].mean():+.4f}", f"{own[('CONTROL', 'CONTROL')].mean():+.4f}"]])
        rep.p(f"Specificity (own set minus other set, AESTHETIC row minus CONTROL row) = {spec.mean():+.4f} (95% CI {ci_spec[0]:+.4f} / {ci_spec[1]:+.4f}), p {fmt_p(p_spec)}. "
              f"Size of the shifts: ‖Δ_A‖ = {nA.mean():.3f}, ‖Δ_C‖ = {nC.mean():.3f}; cosine between Δ_A and Δ_C = {cos_ac.mean():+.2f} "
              "(how much of the change is common to the two fine-tuned models, whatever the training photographs).")
        rep.p("**B3. Alignment and human votes.**")
        rep.table(["question", "result", "95% CI", "p", "n"], [
            ["Triplets where AESTHETIC moves further along the direction than CONTROL: is AESTHETIC also rated higher? (Spearman)", f"ρ **{s_vote[0]:+.2f}**", f"{s_vote[1]:+.2f} / {s_vote[2]:+.2f}", fmt_p(s_vote[3]), s_vote[4]],
            ["… and chosen more often in the pairwise comparisons? (Spearman)", f"ρ **{s_share[0]:+.2f}**", f"{s_share[1]:+.2f} / {s_share[2]:+.2f}", fmt_p(s_share[3]), s_share[4]],
            ["Within a triplet, is the image further along the direction rated higher? (Pearson)", f"r **{r_within:+.2f}**", f"{ci_w[0]:+.2f} / {ci_w[1]:+.2f}", "–", int(ok.sum()) * 3]])
        rep.p("**B4. The rating predictor learned on the photographs, applied to the generated images.**")
        rep.table(["question", "result", "p"], [
            ["Predicted rating vs human rating of the generated images (Spearman, all images)", f"ρ {r_pred.statistic:+.2f}", fmt_p(r_pred.pvalue)],
            ["Same, within triplet (Pearson)", f"r {r_pred_w:+.2f}", "–"],
            ["Predicted rating, AESTHETIC − CONTROL", f"{pred_gain.mean():+.3f} points (95% CI {ci_pg[0]:+.3f} / {ci_pg[1]:+.3f})", fmt_p(p_pg)]])
        rep.p(f"Human votes for comparison (same triplets, exclusions applied): AESTHETIC − CONTROL = {np.nanmean(vote_diff):+.3f} points; "
              f"AESTHETIC chosen in {np.nanmean(share):.1%} of the decisive pairwise comparisons (mean over triplets).")
        rep.p("**B5. Representation transfer score, and amount separated from direction.** Definition: for a triplet (same prompt and seed), the relative source-set "
              "affinity (A4) of the AESTHETIC image minus that of the CONTROL image. The BASE image cancels out, so the score is the AESTHETIC − CONTROL difference of B1 "
              f"multiplied by the length of the aesthetic direction; divided by the squared gap between the training sets it is the transfer ratio of that triplet. "
              f"Mean score **{score.mean():+.4f}** (95% CI {ci_score[0]:+.4f} / {ci_score[1]:+.4f}, p {fmt_p(p_diff)}); median {np.median(score):+.4f}; "
              f"{int((score > 0).sum())} of {len(score)} triplets positive. One row per triplet: `triplet_transfer_dino.csv`.")
        rep.p("A shift along the direction is the product of how far the image moved from BASE (amount, ‖Δ‖) and where it moved (direction, cosine with the aesthetic direction). The two are reported separately:")
        rep.table(["quantity", "BASE → AESTHETIC", "BASE → CONTROL", "difference", "95% CI", "p"], [
            ["amount of shift ‖Δ‖", f"{nA.mean():.3f}", f"{nC.mean():.3f}", f"{mag.mean():+.3f}", f"{ci_mag[0]:+.3f} / {ci_mag[1]:+.3f}", fmt_p(p_mag)],
            ["direction (cosine with the aesthetic direction)", f"{cosA.mean():+.3f}", f"{cosC.mean():+.3f}", f"**{cdf.mean():+.3f}**", f"{ci_cos[0]:+.3f} / {ci_cos[1]:+.3f}", fmt_p(p_cos)]])
        rep.p("Exact split of the AESTHETIC − CONTROL shift of B1 (the two parts add up to it):")
        rep.table(["part", "mean", "95% CI", "p", "share of the total"], [
            ["due to direction (the two models given the same amount of shift)", f"{part_dir.mean():+.4f}", f"{ci_pd[0]:+.4f} / {ci_pd[1]:+.4f}", fmt_p(p_pd), f"{part_dir.mean() / diff.mean():.0%}"],
            ["due to amount (the two models given the same direction)", f"{part_amt.mean():+.4f}", f"{ci_pm[0]:+.4f} / {ci_pm[1]:+.4f}", fmt_p(p_pm), f"{part_amt.mean() / diff.mean():.0%}"]])
        rep.p("Relation with the human votes, per triplet (Spearman, bootstrap over prompts):")
        rep.table(["AESTHETIC − CONTROL difference in", "vs rating difference", "vs share of AESTHETIC choices"], [
            ["amount of shift", f"ρ {s_mag_v[0]:+.2f} ({s_mag_v[1]:+.2f} / {s_mag_v[2]:+.2f}), p {fmt_p(s_mag_v[3])}", f"ρ {s_mag_s[0]:+.2f} ({s_mag_s[1]:+.2f} / {s_mag_s[2]:+.2f}), p {fmt_p(s_mag_s[3])}"],
            ["direction of shift", f"ρ {s_cos_v[0]:+.2f} ({s_cos_v[1]:+.2f} / {s_cos_v[2]:+.2f}), p {fmt_p(s_cos_v[3])}", f"ρ {s_cos_s[0]:+.2f} ({s_cos_s[1]:+.2f} / {s_cos_s[2]:+.2f}), p {fmt_p(s_cos_s[3])}"],
            ["representation transfer score (B3)", f"ρ {s_vote[0]:+.2f} ({s_vote[1]:+.2f} / {s_vote[2]:+.2f}), p {fmt_p(s_vote[3])}", f"ρ {s_share[0]:+.2f} ({s_share[1]:+.2f} / {s_share[2]:+.2f}), p {fmt_p(s_share[3])}"]])
        rep.p(f"**B6. Sensitivity: all the generated images.** Primary analysis above: the {n_img_act} active images shown to the participants ({len(piv)} triplets). "
              f"Here the same embedding quantities on all {n_img_all} generated images ({S_all['n']} triplets, {S_all['prompts']} prompts), including those never shown; "
              "there are no human votes for the added images, so only the embedding results can be compared.")
        f4 = lambda t: f"{t[0]:+.4f} ({t[1]:+.4f} / {t[2]:+.4f}), p {fmt_p(t[3])}"
        cols = [("primary: shown to the participants", dict(n=len(piv), pa=(pa.mean(), *ci_a, p_pa), pc=(pc.mean(), *ci_c, p_pc), diff=(diff.mean(), *ci_diff, p_diff), spec=(spec.mean(), *ci_spec, p_spec),
                                                         pos=int((diff > 0).sum()), nA=nA.mean(), nC=nC.mean(), cosA=cosA.mean(), cosC=cosC.mean(), cos_ac=cos_ac.mean())),
                ("sensitivity: all generated", S_all)] + ([("only the triplets with an image not shown", S_in)] if S_in else [])
        rep.table(["quantity"] + [f"{k} ({S['n']} triplets)" for k, S in cols], [
            ["BASE → AESTHETIC along the direction"] + [f4(S["pa"]) for _, S in cols],
            ["BASE → CONTROL along the direction"] + [f4(S["pc"]) for _, S in cols],
            ["AESTHETIC − CONTROL"] + [f4(S["diff"]) for _, S in cols],
            ["embedding transfer ratio"] + [f"{S['diff'][0] / gap:.0%} ({S['diff'][1] / gap:.0%} / {S['diff'][2] / gap:.0%})" for _, S in cols],
            ["triplets with a positive score"] + [f"{S['pos']} of {S['n']} ({S['pos'] / S['n']:.0%})" for _, S in cols],
            ["specificity"] + [f4(S["spec"]) for _, S in cols],
            ["amount of shift ‖Δ_A‖ / ‖Δ_C‖"] + [f"{S['nA']:.3f} / {S['nC']:.3f}" for _, S in cols],
            ["direction: cosine of Δ_A / Δ_C with the aesthetic direction"] + [f"{S['cosA']:+.3f} / {S['cosC']:+.3f}" for _, S in cols],
            ["cosine between Δ_A and Δ_C"] + [f"{S['cos_ac']:+.2f}" for _, S in cols]])
    return dict(d_ac=d_ac, std=d_ac / pooled, p_free=p_free, p_strat=p_strat, p_subset=p_subset, auc=auc_cv, r_cv=r_cv, diff=diff.mean(), ratio=diff.mean() / gap,
                p_diff=p_diff, rho_vote=s_vote[0], rho_share=s_share[0], r_within=r_within, spec=spec.mean(), p_spec=p_spec,
                auc_aff=auc_aff, p_aff=p_aff, rho_aff=rho_aff.statistic, ratio_all=S_all['diff'][0] / gap, p_all=S_all['diff'][3], n_all=S_all['n'],
                cos_diff=cdf.mean(), p_cos=p_cos, mag_diff=mag.mean(), p_mag=p_mag, dir_share=part_dir.mean() / diff.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("embeddings")
    ap.add_argument("--out", default="embedding_report")
    args = ap.parse_args()
    folder = load_export(args.export)
    os.makedirs(args.out, exist_ok=True)
    rep = Report()
    z = np.load(args.embeddings, allow_pickle=False)
    rep.p("# ARCH300 — computational analysis of the images (DINOv2 embeddings)")
    rep.p(f"Exploratory analysis (ANALYSIS_PLAN.md, section 6). Model `{str(z['model'])}`; preprocessing: {str(z['preprocessing'])}. "
          f"Export: `{os.path.basename(os.path.normpath(args.export))}`. Seed {SEED}; {N_PERM} permutations, {N_BOOT} bootstrap samples.")
    rep.p("The embedding describes what the network sees in an image (content, composition, photographic rendering) without telling them apart: "
          "the results say whether and how much the sets differ for the model, not why.")
    main_res = analyse(folder, args.embeddings, "cls", rep, full=True, out=args.out)
    sens = analyse(folder, args.embeddings, "patch_mean", rep, full=False)
    rep.h("Sensitivity — mean of the patch tokens instead of the CLS token")
    rep.table(["quantity", "CLS token", "mean of patch tokens"], [
        ["A: standardised distance AESTHETIC – CONTROL", f"{main_res['std']:.2f}", f"{sens['std']:.2f}"],
        ["A: p, labels exchanged / within building type / random subsets", " / ".join(fmt_p(main_res[k]) for k in ("p_free", "p_strat", "p_subset")), " / ".join(fmt_p(sens[k]) for k in ("p_free", "p_strat", "p_subset"))],
        ["A: AUC AESTHETIC vs CONTROL", f"{main_res['auc']:.2f}", f"{sens['auc']:.2f}"],
        ["A: ratings predicted from embeddings (r)", f"{main_res['r_cv']:.2f}", f"{sens['r_cv']:.2f}"],
        ["B: embedding transfer ratio", f"{main_res['ratio']:.0%} (p {fmt_p(main_res['p_diff'])})", f"{sens['ratio']:.0%} (p {fmt_p(sens['p_diff'])})"],
        ["B: specificity", f"{main_res['spec']:+.4f} (p {fmt_p(main_res['p_spec'])})", f"{sens['spec']:+.4f} (p {fmt_p(sens['p_spec'])})"],
        ["A: relative source-set affinity, AUC (leave-one-out) / ρ with the ratings", f"{main_res['auc_aff']:.2f} / {main_res['rho_aff']:+.2f}", f"{sens['auc_aff']:.2f} / {sens['rho_aff']:+.2f}"],
        ["B: amount difference ‖Δ_A‖ − ‖Δ_C‖ / direction difference (cosine)", f"{main_res['mag_diff']:+.3f} (p {fmt_p(main_res['p_mag'])}) / {main_res['cos_diff']:+.3f} (p {fmt_p(main_res['p_cos'])})",
         f"{sens['mag_diff']:+.3f} (p {fmt_p(sens['p_mag'])}) / {sens['cos_diff']:+.3f} (p {fmt_p(sens['p_cos'])})"],
        ["B: share of the AESTHETIC − CONTROL shift due to direction", f"{main_res['dir_share']:.0%}", f"{sens['dir_share']:.0%}"],
        [f"B: embedding transfer ratio on all generated images ({main_res['n_all']} triplets)", f"{main_res['ratio_all']:.0%} (p {fmt_p(main_res['p_all'])})", f"{sens['ratio_all']:.0%} (p {fmt_p(sens['p_all'])})"],
        ["B: alignment vs rating difference (ρ) / vs pairwise share (ρ) / within triplet (r)", f"{main_res['rho_vote']:+.2f} / {main_res['rho_share']:+.2f} / {main_res['r_within']:+.2f}",
         f"{sens['rho_vote']:+.2f} / {sens['rho_share']:+.2f} / {sens['r_within']:+.2f}"]])
    open(os.path.join(args.out, "report_dino.md"), "w", encoding="utf-8").write("\n".join(rep.lines) + "\n")
    pd.DataFrame(rep.rows).to_csv(os.path.join(args.out, "results_dino.csv"), index=False)
    print("\n".join(rep.lines))


if __name__ == "__main__":
    main()
