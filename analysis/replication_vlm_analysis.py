"""
ARCH300 — replication across training seeds, VLM stage after unblinding (secondary; plan: ANALYSIS_PLAN.md, section 8.6).

    python replication_vlm_analysis.py [--out vlm_report] [--dry-run]

To be run only after the raw outputs of the replication run have been committed. Joins the blind long table
(vlm/outputs/replication/vlm_scores_long.csv) with the key. No human vote and no embedding result is read.
--dry-run shuffles the corpus label inside each cell × training seed pair (to test the script without unblinding).

Question fixed in advance: do the measures that emerged in section 7 reappear with the same (positive) sign in the three
paired contrasts AESTHETIC − CONTROL, one per training seed? Measures named in advance: representation_quality,
proportional_coherence, component_coherence, material_rendering, photographic_composition; the architectural composite
is reported next to them. No search for new differences: the other items are listed for completeness only.
Scores are standardised within model over the 288 images; "three models" = mean of the three z scores.
Unit: the prompt × generation-seed cell (48 cells, 24 prompts). Intervals: bootstrap over prompts. Fixed seed.
Writes <out>/report_replication_vlm.md, results_replication_vlm.csv, cells_replication_vlm.csv,
vlm_scores_unblinded_replication.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vlm"))
import providers as P  # noqa: E402

SEED, N_BOOT = 20261006, 5000
ARCH, REPR = P.SCORES[:4], P.SCORES[4:]
COMP = ["architectural_appearance", "representation_quality"]
NAMED = ["representation_quality", "proportional_coherence", "component_coherence", "material_rendering", "photographic_composition"]
ALL3 = "three models (z)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "vlm_report"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    cfg = P.load_config()
    t = pd.read_csv(os.path.join(P.path(cfg, "raw_outputs"), "replication", "vlm_scores_long.csv"))
    stem, ext = os.path.splitext(P.path(cfg, "key_file"))
    key = pd.read_csv(f"{stem}_replication{ext or '.csv'}")
    if args.dry_run:
        k1 = key[~key.repeat]
        fake = dict(zip(k1.opaque_id, k1.groupby(["cell", "training_seed"]).corpus.transform(lambda x: rng.permutation(x.to_numpy()))))
        key["corpus"] = key.opaque_id.map(fake)
    d = t.merge(key, left_on="anon_image_id", right_on="anon_id")
    d = d[~d.repeat].copy()
    d[COMP[0]], d[COMP[1]] = d[ARCH].mean(1), d[REPR].mean(1)
    measures = COMP + P.SCORES
    zs = []
    for prov, g in d.groupby("model_provider"):
        z = g[["opaque_id", "cell", "prompt_code", "corpus", "training_seed"]].copy()
        for m in measures:
            sd = g[m].std()
            z[m] = (g[m] - g[m].mean()) / sd if sd > 0 else 0.0
        zs.append(z.assign(model_provider=prov))
    z = pd.concat(zs)
    n_models = z.groupby("opaque_id").model_provider.nunique()
    z3 = z[z.opaque_id.map(n_models) == d.model_provider.nunique()].groupby(["opaque_id", "cell", "prompt_code", "corpus", "training_seed"], as_index=False)[measures].mean().assign(model_provider=ALL3)
    models = [ALL3] + [p for p in P.PROVIDERS if p in set(d.model_provider)]
    tab = {ALL3: z3, **{p: z[z.model_provider == p] for p in models[1:]}}
    pts = {p: d[d.model_provider == p] for p in models[1:]}
    seeds = sorted(key.training_seed.unique())
    name = {s: n for s, n in zip([1254, 9865] + [s for s in seeds if s not in (1254, 9865)], "ABC")}
    cells = sorted(key.cell.unique())
    prompt = np.array([c.rsplit("_", 1)[0] for c in cells])

    def boot(x):
        ok = np.isfinite(x)
        x, pr = x[ok], prompt[ok]
        qs = np.unique(pr)
        mean = np.array([x[pr == q].mean() for q in qs]); w = np.array([(pr == q).sum() for q in qs])
        i = rng.integers(0, len(qs), (N_BOOT, len(qs)))
        b = (mean[i] * w[i]).sum(1) / w[i].sum(1)
        return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(ok.sum())

    def contrast(table, measure, s):
        w = table[table.training_seed == s].pivot(index="cell", columns="corpus", values=measure).reindex(cells)
        return (w["AESTHETIC"] - w["CONTROL"]).to_numpy(float)

    rows, L, out_cells = [], [], []
    add = lambda **k: rows.append(k)
    f = lambda r, dgt=3: f"{r[0]:+.{dgt}f} ({r[1]:+.{dgt}f} / {r[2]:+.{dgt}f})"
    def table(header, body):
        L.extend(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in body] + [""])

    L += ["# ARCH300 — replication across training seeds: evaluation by three vision-language models", ""]
    if args.dry_run:
        L += ["**DRY RUN: the corpus labels were shuffled inside each cell and training seed. These numbers mean nothing.**", ""]
    ids = d.groupby("model_provider").model_id.agg(lambda x: ", ".join(sorted(set(x)))).to_dict()
    L += [f"Secondary analysis (ANALYSIS_PLAN.md, section 8.6). Rubric `{d.rubric_version.iloc[0]}`; models: " + "; ".join(f"{k} `{v}`" for k, v in ids.items())
          + f". Subset fixed before the replication images existed: {len(cells)} prompt × generation-seed cells ({len(np.unique(prompt))} prompts) × 6 LoRA = {key[~key.repeat].opaque_id.nunique()} images, "
          "all evaluated in the same batch, blind to corpus, training seed, prompt and seed. Scores standardised within model over these images; intervals: bootstrap over prompts "
          f"({N_BOOT} samples, seed {SEED}). The scores are observations of the models, not measures of beauty or of architectural quality.", ""]
    L += ["## 0. Data", ""]
    table(["model", "images evaluated", "architectural appearance, mean (SD)", "representation quality, mean (SD)"],
          [[p, f"{g.opaque_id.nunique()} / {key[~key.repeat].opaque_id.nunique()}", f"{g[COMP[0]].mean():.2f} ({g[COMP[0]].std():.2f})", f"{g[COMP[1]].mean():.2f} ({g[COMP[1]].std():.2f})"] for p, g in pts.items()])
    L += ["Reliability of the instrument on this batch (test–retest, agreement between the models), computed without the key: `replication_instrument_report.md`.", ""]

    L += ["## 1. Measures named in advance: AESTHETIC − CONTROL at equal training seed (three models, z)", "",
          "A measure **reappears** if the contrast is positive for the three training seeds. Section 7 (original experiment, 159 triplets) is shown for reference. "
          "Three positive signs alone are weak evidence: with no real difference they occur by chance about once in eight (checked with shuffled labels before unblinding). "
          "The interval of the mean of the three seeds says how firm each result is.", ""]
    ref7 = {}
    f7 = os.path.join(HERE, "vlm_report", "results_vlm.csv")
    if os.path.exists(f7):
        r7 = pd.read_csv(f7)
        r7 = r7[(r7.analysis == "AESTHETIC − CONTROL") & (r7.model == ALL3) & (r7.scale == "z") & r7.section.astype(str).isin(["1", "2"])]
        ref7 = dict(zip(r7.measure, r7.estimate))
    body, verdict = [], {}
    for m in NAMED + [COMP[0]]:
        per = {s: boot(contrast(tab[ALL3], m, s)) for s in seeds}
        pooled = boot(np.mean([contrast(tab[ALL3], m, s) for s in seeds], 0))
        same = all(per[s][0] > 0 for s in seeds)
        verdict[m] = same
        for s in seeds:
            add(section="1", measure=m, model=ALL3, training_seed=name[s], scale="z", estimate=per[s][0], ci_low=per[s][1], ci_high=per[s][2], n=per[s][3], named_in_advance=m in NAMED)
        add(section="1", measure=m, model=ALL3, training_seed="mean of the three", scale="z", estimate=pooled[0], ci_low=pooled[1], ci_high=pooled[2], n=pooled[3], named_in_advance=m in NAMED)
        body.append([("**" + m + "**") if m in NAMED else m + " (not named, for reference)", f"{ref7[m]:+.3f}" if m in ref7 else "–"] + [f(per[s]) for s in seeds] + [f(pooled), "**yes**" if same else "no"])
    table(["measure", "section 7"] + [f"training seed {name[s]} = {s}" for s in seeds] + ["mean of the three seeds", "positive for the three seeds"], body)
    n_ok = sum(verdict[m] for m in NAMED)
    L += [f"**{n_ok} of the {len(NAMED)} measures named in advance reappear** (positive for the three training seeds): "
          + (", ".join(m for m in NAMED if verdict[m]) or "none") + ". Not reappearing: " + (", ".join(m for m in NAMED if not verdict[m]) or "none") + ".", ""]

    L += ["## 2. The same contrasts, model by model (points on the 1–7 scale of each model)", ""]
    body = []
    for m in NAMED + [COMP[0]]:
        for p in models[1:]:
            per = {s: boot(contrast(pts[p], m, s)) for s in seeds}
            for s in seeds:
                add(section="2", measure=m, model=p, training_seed=name[s], scale="points", estimate=per[s][0], ci_low=per[s][1], ci_high=per[s][2], n=per[s][3], named_in_advance=m in NAMED)
            body.append([m if p == models[1] else "", p] + [f(per[s], 2) for s in seeds] + ["yes" if all(per[s][0] > 0 for s in seeds) else "no"])
    table(["measure", "model"] + [f"training seed {name[s]}" for s in seeds] + ["positive for the three seeds"], body)

    L += ["## 3. All nine items (for completeness; not a search for new differences)", ""]
    body = []
    for m in P.SCORES:
        per = {s: boot(contrast(tab[ALL3], m, s)) for s in seeds}
        if m not in NAMED:
            for s in seeds:
                add(section="3", measure=m, model=ALL3, training_seed=name[s], scale="z", estimate=per[s][0], ci_low=per[s][1], ci_high=per[s][2], n=per[s][3], named_in_advance=False)
        body.append([("architecture: " if m in ARCH else "representation: ") + m + (" *" if m in NAMED else "")] + [f(per[s]) for s in seeds] + ["yes" if all(per[s][0] > 0 for s in seeds) else "no"])
    table(["item (* = named in advance)"] + [f"training seed {name[s]}" for s in seeds] + ["positive for the three seeds"], body)

    L += ["## 4. Corpus effect against training-seed effect (three models, z)", "",
          "Same corpus, different training seed: if these differences are as large as the contrasts of section 1, the VLM reading depends on the training run as much as on the corpus.", ""]
    body = []
    import itertools
    for m in COMP:
        corpus_eff = np.mean([abs(boot(contrast(tab[ALL3], m, s))[0]) for s in seeds])
        seed_effs = []
        for c in ("AESTHETIC", "CONTROL"):
            w = tab[ALL3][tab[ALL3].corpus == c].pivot(index="cell", columns="training_seed", values=m).reindex(cells)
            for a, b in itertools.combinations(seeds, 2):
                r = boot((w[a] - w[b]).to_numpy(float))
                seed_effs.append(abs(r[0]))
                add(section="4", measure=m, model=ALL3, training_seed=f"{c}: {name[a]} − {name[b]}", scale="z", estimate=r[0], ci_low=r[1], ci_high=r[2], n=r[3], named_in_advance=m in NAMED)
                body.append([m, f"{c}: {name[a]} − {name[b]}", f(r)])
        add(section="4", measure=m, model=ALL3, training_seed="R = mean |corpus contrast| / mean |between-seed difference|", scale="ratio", estimate=corpus_eff / np.mean(seed_effs), ci_low=None, ci_high=None, n=None, named_in_advance=m in NAMED)
        body.append([m, "**R = mean |corpus contrast| / mean |between-seed difference|**", f"**{corpus_eff / np.mean(seed_effs):.1f}**"])
    table(["composite", "comparison", "difference (95% CI)"], body)

    # ---- added after unblinding (2026-10-02), labelled as such: is this batch consistent with section 7?
    L += ["## 5. Consistency with the original experiment (check added after unblinding)", "",
          "Section 7 compared the two original LoRA, which had different training seeds: AESTHETIC with seed A against CONTROL with seed B. The same unpaired contrast, "
          "computed on this batch (the 48 cells of the subset, images evaluated again), and the two CONTROL runs it depends on:", ""]
    body = []
    for m in COMP + [x for x in NAMED if x not in COMP]:
        g = tab[ALL3]
        y = {(c, sd): g[(g.corpus == c) & (g.training_seed == sd)].set_index("cell")[m].reindex(cells).to_numpy(float) for c in ("AESTHETIC", "CONTROL") for sd in seeds}
        orig = boot(y[("AESTHETIC", 1254)] - y[("CONTROL", 9865)])
        ctl = boot(y[("CONTROL", 9865)] - np.mean([y[("CONTROL", sd)] for sd in seeds if sd != 9865], 0))
        add(section="5", measure=m, model=ALL3, training_seed="original unpaired contrast: AESTHETIC seed A − CONTROL seed B", scale="z", estimate=orig[0], ci_low=orig[1], ci_high=orig[2], n=orig[3], named_in_advance=False)
        add(section="5", measure=m, model=ALL3, training_seed="original CONTROL LoRA (seed B) − mean of the two other CONTROL LoRA", scale="z", estimate=ctl[0], ci_low=ctl[1], ci_high=ctl[2], n=ctl[3], named_in_advance=False)
        body.append([m, f"{ref7[m]:+.3f}" if m in ref7 else "–", f(orig), f(ctl)])
    table(["measure", "section 7 (159 triplets)", "same unpaired contrast, this batch (48 cells)", "original CONTROL LoRA − the two other CONTROL LoRA"], body)
    old_file = os.path.join(HERE, "vlm_report", "vlm_scores_unblinded.csv")
    if os.path.exists(old_file):
        old = pd.read_csv(old_file)
        mrg = d.merge(old, on=["opaque_id", "model_provider"], suffixes=("", "_old"))
        rr = [[p, int((mrg.model_provider == p).sum())] + [f"{np.corrcoef(mrg[mrg.model_provider == p][c], mrg[mrg.model_provider == p][c + '_old'])[0, 1]:+.2f}" for c in COMP] for p in models[1:]]
        for r_ in rr:
            for c, val in zip(COMP, r_[2:]):
                add(section="5", measure=c, model=r_[0], training_seed="same image, score of section 7 vs score of this batch (Pearson r)", scale="r", estimate=float(val), ci_low=None, ci_high=None, n=r_[1], named_in_advance=False)
        L += ["The images of the two original LoRA in the subset were evaluated in section 7 and again here. Correlation between the two scores of the same image:", ""]
        table(["model", "images", "architectural appearance (r)", "representation quality (r)"], rr)

    for mod in models:
        for m in COMP:
            g = tab[mod]
            w = g.assign(col=g.corpus.str.lower() + "_seed_" + g.training_seed.map(name)).pivot(index="cell", columns="col", values=m).reindex(cells)
            out_cells.append(w.reset_index().assign(model=mod, measure=m))
    suffix = "_DRYRUN" if args.dry_run else ""
    os.makedirs(args.out, exist_ok=True)
    pd.concat(out_cells).to_csv(os.path.join(args.out, f"cells_replication_vlm{suffix}.csv"), index=False)
    d[["opaque_id", "training_run", "corpus", "training_seed", "prompt_code", "seed", "cell", "model_provider", "model_id", "rubric_version"] + P.SCORES + COMP + ["timestamp"]].to_csv(
        os.path.join(args.out, f"vlm_scores_unblinded_replication{suffix}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"results_replication_vlm{suffix}.csv"), index=False)
    open(os.path.join(args.out, f"report_replication_vlm{suffix}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
