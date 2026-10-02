"""
ARCH300 — analysis of the VLM evaluations after unblinding (exploratory; plan: ANALYSIS_PLAN.md, section 7.7).

    python vlm_analysis.py <export.zip | extracted folder> [--out vlm_report] [--dry-run]

To be run only after the raw outputs of the full run have been committed. Joins the long table of the scores
(vlm/outputs/full/vlm_scores_long.csv) with the key and with the human votes, and computes exactly what section 7.7 lists.
--dry-run shuffles the conditions inside each triplet before anything is computed (to test the script without unblinding).

Composites (computed here, not by the models): architectural_appearance = mean of 4 items, representation_quality = mean
of 5 items. "Three models" = mean of the scores standardised within model (z over the images of the primary set).
Primary set: the triplets shown to the participants; sensitivity: all the triplets. Intervals: bootstrap over prompts;
p values: sign-flip permutation of the prompt means. Fixed seed.
Writes <out>/report_vlm.md, results_vlm.csv, triplet_vlm.csv, vlm_scores_unblinded.csv.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vlm"))
import providers as P  # noqa: E402

SEED, N_PERM, N_BOOT = 20261003, 10000, 2000
ARCH, REPR = P.SCORES[:4], P.SCORES[4:]
COMP = ["architectural_appearance", "representation_quality"]
ALL3 = "three models (z)"


def fmt_p(p):
    return "–" if p is None or not math.isfinite(p) else ("< .001" if p < 0.001 else f"{p:.3f}".replace("0.", "."))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


class Report:
    def __init__(self):
        self.lines, self.rows = [], []

    def h(self, t, level=2): self.lines += ["", "#" * level + " " + t, ""]
    def p(self, t=""): self.lines += [t, ""]

    def table(self, header, rows):
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join("---" for _ in header) + "|")
        self.lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]

    def add(self, section, analysis, model, measure, estimate, lo=None, hi=None, p=None, p_holm=None, n=None, scale=""):
        self.rows.append(dict(section=section, analysis=analysis, model=model, measure=measure, scale=scale, estimate=estimate, ci_low=lo, ci_high=hi, p=p, p_holm=p_holm, n=n))


def as_bool(s):
    return s.astype(str).str.lower().eq("true")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--out", default=os.path.join(HERE, "vlm_report"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    cfg = P.load_config()
    folder = args.export
    if zipfile.is_zipfile(folder):
        folder = tempfile.mkdtemp()
        zipfile.ZipFile(args.export).extractall(folder)
    rd = lambda f: pd.read_csv(os.path.join(folder, f))
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------ data: scores + key
    t = pd.read_csv(os.path.join(P.path(cfg, "raw_outputs"), "full", "vlm_scores_long.csv"))
    stem, ext = os.path.splitext(P.path(cfg, "key_file"))
    key = pd.read_csv(f"{stem}_full{ext or '.csv'}")
    if args.dry_run:
        k1 = key[~key.repeat]
        fake = dict(zip(k1.opaque_id, k1.groupby("triplet").condition.transform(lambda x: rng.permutation(x.to_numpy()))))
        key["condition"] = key.opaque_id.map(fake)
    d = t.merge(key, left_on="anon_image_id", right_on="anon_id")
    d = d[~d.repeat].copy()
    d[COMP[0]], d[COMP[1]] = d[ARCH].mean(1), d[REPR].mean(1)
    measures = COMP + P.SCORES
    shown = key[~key.repeat].groupby("triplet").active.all()
    d["shown"] = d.triplet.map(shown)
    # standardised within model on the primary set; "three models" = mean of the three z scores (only where all three exist)
    zs = []
    for prov, g in d.groupby("model_provider"):
        ref = g[g.shown]
        z = g[["opaque_id", "triplet", "condition", "prompt_code", "seed", "shown"]].copy()
        for m in measures:
            sd = ref[m].std()
            z[m] = (g[m] - ref[m].mean()) / sd if sd > 0 else 0.0
        zs.append(z.assign(model_provider=prov))
    z = pd.concat(zs)
    n_models = z.groupby("opaque_id").model_provider.nunique()
    z3 = z[z.opaque_id.map(n_models) == len(P.PROVIDERS)].groupby(["opaque_id", "triplet", "condition", "prompt_code", "seed", "shown"], as_index=False)[measures].mean().assign(model_provider=ALL3)
    models = [ALL3] + [p for p in P.PROVIDERS if p in set(d.model_provider)]
    tables = {(ALL3, "z"): z3, **{(p, "z"): z[z.model_provider == p] for p in models[1:]}, **{(p, "points"): d[d.model_provider == p] for p in models[1:]}}

    def wide(model, scale, measure, only_shown=True):
        """Triplets × conditions for one measure; complete triplets only."""
        g = tables[(model, scale)]
        if only_shown:
            g = g[g.shown]
        return g.pivot(index="triplet", columns="condition", values=measure).dropna()

    def test(x: np.ndarray, prompt: np.ndarray):
        """Mean, cluster bootstrap CI over prompts, sign-flip p on the prompt means."""
        qs = np.unique(prompt)
        m = np.array([x[prompt == q].mean() for q in qs]); w = np.array([(prompt == q).sum() for q in qs])
        i = rng.integers(0, len(qs), (N_BOOT, len(qs)))
        lo, hi = np.percentile((m[i] * w[i]).sum(1) / w[i].sum(1), [2.5, 97.5])
        sg = rng.choice([-1.0, 1.0], size=(N_PERM, len(qs)))
        p = (1 + (np.abs((sg * m).mean(1)) >= abs(m.mean())).sum()) / (1 + N_PERM)
        return float(x.mean()), float(lo), float(hi), float(p), len(x)

    prm = lambda idx: idx.str.rsplit("_", n=1).str[0].to_numpy()

    def contrast(model, scale, measure, a="AESTHETIC", b="CONTROL", only_shown=True):
        w = wide(model, scale, measure, only_shown)
        return test((w[a] - w[b]).to_numpy(), prm(w.index))

    rep = Report()
    rep.lines += ["# ARCH300 — evaluation of the generated images by three vision-language models", ""]
    if args.dry_run:
        rep.p("**DRY RUN: the conditions were shuffled inside each triplet. These numbers mean nothing.**")
    ids = d.groupby("model_provider").model_id.agg(lambda x: ", ".join(sorted(set(x)))).to_dict()
    rep.p(f"Exploratory analysis (ANALYSIS_PLAN.md, section 7). Rubric `{d.rubric_version.iloc[0]}`; models: "
          + "; ".join(f"{k} `{v}`" for k, v in ids.items()) + f". One image per request, blind to condition, prompt and seed. Seed {SEED}; {N_PERM} permutations, {N_BOOT} bootstrap samples.")
    rep.p("The scores are observations of the models, not measures of beauty or of architectural quality. Each model was trained on images from the web and has its own taste. "
          "Composites: architectural appearance = mean of 4 items; representation quality = mean of 5 items. \"Three models\" = mean of the scores standardised within model "
          "(1 = one standard deviation of the images of the primary set); the columns in points are on the 1–7 scale of each model.")

    # ------------------------------------------------------------ 0. data
    rep.h("0. Data")
    n_img = key[~key.repeat].opaque_id.nunique()
    rows = []
    for p_ in models[1:]:
        g = d[d.model_provider == p_]
        rows.append([p_, f"{g.opaque_id.nunique()} / {n_img}", len(wide(p_, 'points', COMP[0])), len(wide(p_, 'points', COMP[0], False)),
                     f"{g[g.shown][COMP[0]].mean():.2f} ({g[g.shown][COMP[0]].std():.2f})", f"{g[g.shown][COMP[1]].mean():.2f} ({g[g.shown][COMP[1]].std():.2f})"])
    rows.append([ALL3, f"{z3.opaque_id.nunique()} / {n_img}", len(wide(ALL3, 'z', COMP[0])), len(wide(ALL3, 'z', COMP[0], False)), "–", "–"])
    rep.table(["model", "images evaluated", "complete triplets, primary set", "complete triplets, all", "architectural appearance, mean (SD)", "representation quality, mean (SD)"], rows)
    short = [p_ for p_ in models[1:] if d[d.model_provider == p_].opaque_id.nunique() < n_img]
    if short:
        rep.p("**Incomplete run:** " + ", ".join(short) + " did not evaluate all the images (the spending limit of the account was reached; the requests stopped in the fixed order: "
              "primary set first, then the repeats, then the triplets not shown). The table above says how many triplets each model has; for such a model, and for \"three models\", "
              "the sensitivity on all the images covers only part of the triplets not shown. The remaining requests follow the same manifest and can be added later.")
    rep.p("Reliability of the instrument (test–retest, agreement between the models, use of the scale), computed without the key: `full_instrument_report.md`.")
    rep.p("Intervals and p values come from two different procedures (bootstrap of the triplets by prompt; sign-flip of the prompt means), so an interval that just "
          "touches zero and a p value just below .05, or the reverse, can occur together: such results are borderline and are to be read as such.")

    # ------------------------------------------------------------ 1. main comparison
    rep.h("1. Main comparison: AESTHETIC − CONTROL, same prompt and seed")
    res = {}
    for m in COMP:
        for mod in models:
            res[(mod, m, "z")] = contrast(mod, "z", m)
            if mod != ALL3:
                res[(mod, m, "points")] = contrast(mod, "points", m)
    hp = holm([res[(ALL3, m, "z")][3] for m in COMP])
    rows = []
    for k, m in enumerate(COMP):
        for mod in models:
            e = res[(mod, m, "z")]
            pts = res.get((mod, m, "points"))
            ph = hp[k] if mod == ALL3 else None
            rep.add("1", "AESTHETIC − CONTROL", mod, m, *e[:4], ph, e[4], "z")
            if pts:
                rep.add("1", "AESTHETIC − CONTROL", mod, m, *pts[:4], None, pts[4], "points")
            rows.append([m if mod == ALL3 else "", f"**{mod}**" if mod == ALL3 else mod, f"{e[0]:+.3f}", f"{e[1]:+.3f} / {e[2]:+.3f}", fmt_p(e[3]) + (f" (Holm {fmt_p(ph)})" if ph is not None else ""),
                         f"{pts[0]:+.3f} ({pts[1]:+.3f} / {pts[2]:+.3f})" if pts else "–", e[4]])
    rep.table(["composite", "model", "difference (z)", "95% CI", "p", "difference in points (95% CI)", "triplets"], rows)
    same = {m: [np.sign(res[(mod, m, "z")][0]) for mod in models[1:]] for m in COMP}
    rep.p("Do the three models point the same way? " + "; ".join(f"{m}: {'yes' if len(set(v)) == 1 else 'no'} ({', '.join('+' if s > 0 else '−' for s in v)})" for m, v in same.items()) + ".")

    rep.h("Which block carries the difference", 3)
    rows = []
    for mod in models:
        a, r = wide(mod, "z", COMP[0]), wide(mod, "z", COMP[1])
        common = a.index.intersection(r.index)
        x = ((a.loc[common, "AESTHETIC"] - a.loc[common, "CONTROL"]) - (r.loc[common, "AESTHETIC"] - r.loc[common, "CONTROL"])).to_numpy()
        e = test(x, prm(common))
        rep.add("1", "AESTHETIC − CONTROL: architectural minus representation", mod, "difference between the composites", *e[:4], None, e[4], "z")
        rows.append([mod, f"{e[0]:+.3f}", f"{e[1]:+.3f} / {e[2]:+.3f}", fmt_p(e[3])])
    rep.table(["model", "architectural − representation (z)", "95% CI", "p"], rows)
    rep.p("Positive = the AESTHETIC − CONTROL difference is larger for the architectural appearance than for the quality of the representation.")

    # ------------------------------------------------------------ 2. secondary
    rep.h("2. Secondary: the nine items")
    it = {(mod, m): contrast(mod, "z", m) for mod in models for m in P.SCORES}
    hp9 = holm([it[(ALL3, m)][3] for m in P.SCORES])
    rows = []
    for k, m in enumerate(P.SCORES):
        e = it[(ALL3, m)]
        rep.add("2", "AESTHETIC − CONTROL", ALL3, m, *e[:4], hp9[k], e[4], "z")
        cells = []
        for mod in models[1:]:
            q = contrast(mod, "points", m)
            rep.add("2", "AESTHETIC − CONTROL", mod, m, *q[:4], None, q[4], "points")
            cells.append(f"{q[0]:+.2f} (p {fmt_p(q[3])})")
        rows.append([("architecture: " if m in ARCH else "representation: ") + m, f"{e[0]:+.3f}", f"{e[1]:+.3f} / {e[2]:+.3f}", fmt_p(e[3]), fmt_p(hp9[k])] + cells)
    rep.table(["item", "three models (z)", "95% CI", "p", "p (Holm, 9)"] + [f"{m}, points" for m in models[1:]], rows)

    rep.h("Other contrasts between conditions", 3)
    rows = []
    for a, b in (("BASE", "CONTROL"), ("AESTHETIC", "BASE")):
        for m in COMP:
            cells = []
            for mod in models:
                e = contrast(mod, "z", m, a, b)
                rep.add("2", f"{a} − {b}", mod, m, *e[:4], None, e[4], "z")
                cells.append(f"{e[0]:+.3f} ({e[1]:+.3f} / {e[2]:+.3f}), p {fmt_p(e[3])}")
            rows.append([f"{a} − {b}", m] + cells)
    rep.table(["contrast", "composite"] + [f"{m} (z)" if m != ALL3 else m for m in models], rows)

    # ------------------------------------------------------------ 3. human votes and embeddings
    rep.h("3. Agreement with the human votes and with the embedding analysis")
    R = rd("posttraining_ratings.csv")
    R = R[~as_bool(R.excluded_from_analysis)].copy()
    R["c"] = R.score - R.groupby("participant_id").score.transform("mean")
    R["triplet"] = R.prompt_code + "_" + R.seed.astype(str)
    hum = R.groupby(["triplet", "condition_code"]).c.mean().unstack()
    T = rd("pairwise_trials.csv")
    T = T[~as_bool(T.excluded_from_analysis) & T.selected_side.isin(["Left", "Right"])]
    T = T[T.left_condition.isin(["AESTHETIC", "CONTROL"]) & T.right_condition.isin(["AESTHETIC", "CONTROL"])]
    share = (T.winner_condition == "AESTHETIC").groupby(T.prompt_code + "_" + T.seed.astype(str)).mean()
    tt_file = os.path.join(HERE, "embedding_report", "triplet_transfer_dino.csv")
    transfer = None
    if os.path.exists(tt_file):
        tt = pd.read_csv(tt_file)
        transfer = pd.Series(tt.representation_transfer_score.to_numpy(), index=tt.prompt_code + "_" + tt.seed.astype(str))

    def corr_ci(x, y, prompt, kind):
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, prompt = x[ok], y[ok], prompt[ok]
        f = (lambda a, b: stats.spearmanr(a, b).statistic) if kind == "spearman" else (lambda a, b: np.corrcoef(a, b)[0, 1])
        r = f(x, y)
        qs = np.unique(prompt); where = {q: np.where(prompt == q)[0] for q in qs}
        b = []
        for _ in range(N_BOOT):
            i = np.concatenate([where[q] for q in qs[rng.integers(0, len(qs), len(qs))]])
            b.append(f(x[i], y[i]))
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        p = stats.spearmanr(x, y).pvalue if kind == "spearman" else stats.pearsonr(x, y).pvalue
        return float(r), float(lo), float(hi), float(p), int(ok.sum())

    cond = ["AESTHETIC", "BASE", "CONTROL"]
    rows_img, rows_trip = [], []
    for m in COMP:
        for mod in models:
            w = wide(mod, "z", m)
            h = hum.reindex(w.index)[cond]
            ok = h.notna().all(1).to_numpy()
            wv, hv = w[cond].to_numpy()[ok], h.to_numpy()[ok]
            pr3 = np.repeat(prm(w.index)[ok], 3)
            e_all = corr_ci(wv.ravel(), hv.ravel(), pr3, "spearman")
            e_in = corr_ci((wv - wv.mean(1, keepdims=True)).ravel(), (hv - hv.mean(1, keepdims=True)).ravel(), pr3, "pearson")
            rep.add("3", "VLM score vs human rating, all images (Spearman)", mod, m, *e_all[:4], None, e_all[4], "z")
            rep.add("3", "VLM score vs human rating, within triplet (Pearson)", mod, m, *e_in[:4], None, e_in[4], "z")
            rows_img.append([m if mod == ALL3 else "", mod, f"ρ {e_all[0]:+.2f} ({e_all[1]:+.2f} / {e_all[2]:+.2f})", f"r {e_in[0]:+.2f} ({e_in[1]:+.2f} / {e_in[2]:+.2f})", e_all[4]])
            dv = (w["AESTHETIC"] - w["CONTROL"])
            cells = []
            for name, other in (("rating difference AESTHETIC − CONTROL", hum["AESTHETIC"] - hum["CONTROL"]), ("share of AESTHETIC choices (pairwise)", share), ("representation transfer score (DINOv2)", transfer)):
                if other is None:
                    cells.append("–"); continue
                e = corr_ci(dv.to_numpy(), other.reindex(dv.index).to_numpy(float), prm(dv.index), "spearman")
                rep.add("3", f"AESTHETIC − CONTROL difference of the VLM score vs {name}, per triplet (Spearman)", mod, m, *e[:4], None, e[4], "z")
                cells.append(f"ρ {e[0]:+.2f} ({e[1]:+.2f} / {e[2]:+.2f}), p {fmt_p(e[3])}")
            rows_trip.append([m if mod == ALL3 else "", mod] + cells)
    rep.p("**Per image.** Does an image with a higher VLM score also have a higher human rating (ratings centred on each participant's mean, exclusions applied)? "
          "\"All images\" compares images of different prompts; \"within triplet\" compares only the three images of the same prompt and seed.")
    rep.table(["composite", "model", "all images (Spearman, 95% CI)", "within triplet (Pearson, 95% CI)", "images"], rows_img)
    rep.p("**Per triplet.** Where the VLM score favours AESTHETIC over CONTROL more, do the participants (and the embedding analysis) do the same?")
    rep.table(["composite", "model", "vs rating difference", "vs share of AESTHETIC choices", "vs representation transfer score"], rows_trip)

    # ------------------------------------------------------------ 4. sensitivity
    rep.h("4. Sensitivity: all the generated images")
    rows = []
    for m in COMP:
        for mod in models:
            a, b = res[(mod, m, "z")], contrast(mod, "z", m, only_shown=False)
            rep.add("4", "AESTHETIC − CONTROL, all generated images", mod, m, *b[:4], None, b[4], "z")
            rows.append([m if mod == ALL3 else "", mod, f"{a[0]:+.3f} ({a[1]:+.3f} / {a[2]:+.3f}), p {fmt_p(a[3])}, n {a[4]}", f"{b[0]:+.3f} ({b[1]:+.3f} / {b[2]:+.3f}), p {fmt_p(b[3])}, n {b[4]}"])
    rep.table(["composite", "model", "primary: triplets shown to the participants (z)", "all triplets (z)"], rows)
    rep.p("The standardisation is the same in the two columns (mean and standard deviation of the primary set).")

    # ------------------------------------------------------------ files
    out_t = []
    for mod in models:
        for m in COMP:
            w = wide(mod, "z", m, False)
            out_t.append(pd.DataFrame(dict(triplet=w.index, model=mod, measure=m, shown_to_participants=shown.reindex(w.index).to_numpy(), aesthetic=w["AESTHETIC"].to_numpy(), base=w["BASE"].to_numpy(),
                                           control=w["CONTROL"].to_numpy(), aesthetic_minus_control=(w["AESTHETIC"] - w["CONTROL"]).to_numpy())))
    suffix = "_DRYRUN" if args.dry_run else ""
    pd.concat(out_t).to_csv(os.path.join(args.out, f"triplet_vlm{suffix}.csv"), index=False)
    d[["opaque_id", "condition", "prompt_code", "seed", "triplet", "active", "shown", "model_provider", "model_id", "rubric_version"] + P.SCORES + COMP + ["timestamp"]].to_csv(
        os.path.join(args.out, f"vlm_scores_unblinded{suffix}.csv"), index=False)
    pd.DataFrame(rep.rows).to_csv(os.path.join(args.out, f"results_vlm{suffix}.csv"), index=False)
    open(os.path.join(args.out, f"report_vlm{suffix}.md"), "w", encoding="utf-8").write("\n".join(rep.lines) + "\n")
    print("\n".join(rep.lines))


if __name__ == "__main__":
    main()
