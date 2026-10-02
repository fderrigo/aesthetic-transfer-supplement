"""
ARCH300 — do the captions of the two training sets differ? (exploratory; plan: ANALYSIS_PLAN.md, section 9.5)

    python caption_analysis.py <export.zip | extracted folder> [--out decomposition_report]

Reads from the export the caption snapshots of the two frozen training sets used by the runs (aesthetic_dataset.csv,
control_dataset.csv) and the generation prompts. Nothing else.
(a) words: share of captions containing each word in the two sets, Fisher exact test, Benjamini–Hochberg;
(b) set predicted from the caption alone: ridge classifier on word presence, 10-fold cross-validated AUC, permutation p;
(c) the unbalanced words that also occur in the generation prompts.
Writes <out>/report_captions.md, results_captions.csv, words_captions.csv.
"""
from __future__ import annotations

import argparse
import os
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from embedding_analysis import auc, ridge_cv

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N_PERM, MIN_DOCS = 20261008, 1000, 5
STOP = set("a an the of and with in on at to by for from its it is are as that this over under into between behind beside near through against along".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", str(text).lower()) if w not in STOP and len(w) > 1}


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    q = np.empty(len(p))
    run = 1.0
    for rank, i in zip(range(len(p), 0, -1), order[::-1]):
        run = min(run, p[i] * len(p) / rank)
        q[i] = run
    return q


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
    runs = rd("training_runs.csv").set_index("code")
    da, dc = runs.training_dataset["RUN-AESTHETIC-4"], runs.training_dataset["RUN-CONTROL-4"]
    A = rd("aesthetic_dataset.csv"); A = A[A.dataset_code == da]
    C = rd("control_dataset.csv"); C = C[C.dataset_code == dc]
    caps = list(A.caption_snapshot) + list(C.caption_snapshot)
    lab = np.array([True] * len(A) + [False] * len(C))
    docs = [words(c) for c in caps]
    plans = rd("generation_plan.csv")
    P = rd("prompt_sets.csv")
    P = P[P.prompt_set_id.isin(plans.prompt_set_id)]
    prompt_docs = [words(t) for t in P.text]
    vocab = sorted({w for d in docs for w in d})
    df = {w: sum(w in d for d in docs) for w in vocab}
    keep = [w for w in vocab if df[w] >= MIN_DOCS]
    X = np.array([[w in d for w in keep] for d in docs], float)
    na, nc = int(lab.sum()), int((~lab).sum())
    rows, L = [], []
    add = lambda analysis, quantity, estimate, lo=None, hi=None, p=None, n=None: rows.append(dict(analysis=analysis, quantity=quantity, estimate=estimate, ci_low=lo, ci_high=hi, p=p, n=n))

    # (a) words
    W = []
    for j, w in enumerate(keep):
        a, c = int(X[lab, j].sum()), int(X[~lab, j].sum())
        W.append(dict(word=w, aesthetic_captions=a, control_captions=c, aesthetic_share=a / na, control_share=c / nc, difference=a / na - c / nc,
                      p=stats.fisher_exact([[a, na - a], [c, nc - c]])[1], in_generation_prompts=sum(w in d for d in prompt_docs)))
    W = pd.DataFrame(W)
    W["p_bh"] = bh(W.p.to_numpy())
    W = W.reindex(W.difference.abs().sort_values(ascending=False).index)
    W.to_csv(os.path.join(args.out, "words_captions.csv"), index=False)
    sig = W[W.p_bh < 0.05]

    # (b) the set from the caption alone
    rng = np.random.default_rng(SEED)
    yl = np.where(lab, 1.0, -1.0)
    pred, lam = ridge_cv(X, yl, np.random.default_rng(SEED))
    a_obs = auc(pred, lab)
    null = []
    for _ in range(N_PERM):
        l = rng.permutation(lab)
        null.append(auc(ridge_cv(X, np.where(l, 1.0, -1.0), rng, lam=lam)[0], l))
    null = np.array(null)
    p_auc = (1 + (null >= a_obs).sum()) / (1 + N_PERM)
    b = []
    for _ in range(5000):
        i = rng.integers(0, len(lab), len(lab))
        if lab[i].any() and (~lab[i]).any():
            b.append(auc(pred[i], lab[i]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    add("b. set predicted from the caption alone", "cross-validated AUC (0.5 = chance)", a_obs, lo, hi, p_auc, len(lab))
    add("b. set predicted from the caption alone", "AUC with the labels exchanged: mean", null.mean(), np.percentile(null, 2.5), np.percentile(null, 97.5), None, N_PERM)
    lens = np.array([len(str(c).split()) for c in caps])
    add("a. words", "caption length, AESTHETIC − CONTROL (words)", lens[lab].mean() - lens[~lab].mean(), None, None, stats.mannwhitneyu(lens[lab], lens[~lab]).pvalue, len(lab))
    add("a. words", f"words present in at least {MIN_DOCS} captions", len(keep), None, None, None, None)
    add("a. words", "words with different frequency in the two sets (Benjamini–Hochberg q < .05)", len(sig), None, None, None, len(keep))
    add("a. words", "words expected to reach p < .05 by chance", 0.05 * len(keep), None, None, None, len(keep))
    add("a. words", "words with uncorrected p < .05", int((W.p < 0.05).sum()), None, None, None, len(keep))

    L += ["# ARCH300 — the captions of the two training sets", "",
          f"Exploratory analysis (ANALYSIS_PLAN.md, section 9.5). Caption snapshots of the frozen datasets `{da}` ({na} captions) and `{dc}` ({nc} captions), the same for the original and the replication runs. "
          f"Mean length {lens[lab].mean():.1f} and {lens[~lab].mean():.1f} words (Mann–Whitney p {stats.mannwhitneyu(lens[lab], lens[~lab]).pvalue:.2f}). "
          f"{len(vocab)} different words, {len(keep)} present in at least {MIN_DOCS} captions. Captions describe the photographs: words can differ simply because the photographs differ.", "",
          "## a. Words", "",
          f"Words whose frequency differs between the two sets after correction for the {len(keep)} words tested (Benjamini–Hochberg q < .05): **{len(sig)}**. "
          f"Uncorrected p < .05: {int((W.p < 0.05).sum())} words, against about {0.05 * len(keep):.0f} expected by chance alone.", "",
          "The twenty words with the largest difference in share of captions:", "",
          "| word | AESTHETIC captions | CONTROL captions | difference | p | q (BH) | generation prompts containing it (of 48) |", "|---|---|---|---|---|---|---|"]
    for r in W.head(20).itertuples():
        L.append(f"| {r.word} | {r.aesthetic_captions} ({r.aesthetic_share:.0%}) | {r.control_captions} ({r.control_share:.0%}) | {r.difference:+.0%} | {r.p:.3f} | {r.p_bh:.2f} | {r.in_generation_prompts} |")
    L += ["", "## b. Can the set be told from the caption alone?", "",
          f"Ridge classifier on the presence of the {len(keep)} words, 10-fold cross-validation: AUC **{a_obs:.2f}** (95% CI {lo:.2f} / {hi:.2f}); with the labels exchanged {null.mean():.2f} "
          f"(95% of the permutations between {np.percentile(null, 2.5):.2f} and {np.percentile(null, 97.5):.2f}); permutation p = {p_auc:.3f}. "
          "For comparison, the images alone give AUC 0.78 (DINOv2 embeddings, section 6).", "",
          "**Reading (9.5):** " + ("the captions do not tell the two sets apart beyond chance: no evidence that the text carries the difference between the sets."
                                   if p_auc >= 0.05 else "the captions tell the two sets apart better than chance: the text is a second channel, to be declared next to the images."), "",
          "## c. Unbalanced words and the generation prompts", ""]
    top = W[W.p < 0.05]
    in_p = top[top.in_generation_prompts > 0]
    L += [f"Of the {len(top)} words with uncorrected p < .05, {len(in_p)} also occur in the generation prompts: "
          + (", ".join(f"{r.word} ({r.difference:+.0%}; {r.in_generation_prompts} prompts)" for r in in_p.itertuples()) or "none") + ".", ""]
    add("c. prompts", "words with uncorrected p < .05 that also occur in the generation prompts", len(in_p), None, None, None, len(top))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "results_captions.csv"), index=False)
    open(os.path.join(args.out, "report_captions.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
