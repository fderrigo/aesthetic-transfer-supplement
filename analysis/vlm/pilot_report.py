"""
Report of the VLM pilot (ANALYSIS_PLAN.md, 7.5), read WITHOUT the key: it does not know conditions, prompts or seeds.

    python pilot_report.py [pilot|full]

Checks: completeness, use of the scale, test–retest on the repeated images, agreement between the three models, the
notes (pilot only) and the tokens used. Writes ../vlm_report/<stage>_instrument_report.md. With `full` the same checks
are made on the full run (still without the key).
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import providers as P

ARCH, REPR = P.SCORES[:4], P.SCORES[4:]


def icc21(m: np.ndarray) -> float:
    """ICC(2,1), absolute agreement, two-way random: rows = images, columns = raters/occasions."""
    m = m[~np.isnan(m).any(1)]
    n, k = m.shape
    if n < 3:
        return float("nan")
    g = m.mean()
    msr = k * ((m.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - g) ** 2).sum() / (k - 1)
    mse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + g) ** 2).sum() / ((n - 1) * (k - 1))
    d = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / d) if d > 0 else float("nan")


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    cfg = P.load_config()
    out_dir = os.path.join(P.path(cfg, "raw_outputs"), stage)
    t = pd.read_csv(os.path.join(out_dir, "vlm_scores_long.csv"))
    man = pd.read_csv(os.path.join(P.HERE, "manifests", f"{stage}_manifest_blind.csv"))
    reps = pd.read_csv(os.path.join(P.HERE, "manifests", f"{stage}_repeats.csv"))
    t["architectural_appearance"], t["representation_quality"] = t[ARCH].mean(1), t[REPR].mean(1)
    items = P.SCORES + ["architectural_appearance", "representation_quality"]
    first = t[~t.anon_image_id.isin(reps.repeat_anon_id)]
    L = [f"# VLM evaluation — instrument report ({stage})", "",
         f"Read without the key: no condition, prompt or seed is known here. Rubric `{t.rubric_version.iloc[0]}`. "
         f"{len(man)} evaluations requested per model ({len(man) - len(reps)} images, {len(reps)} repeated).", "",
         "## 1. Completeness", ""]
    rows = []
    for prov in P.PROVIDERS:
        att = [json.loads(l) for l in open(os.path.join(out_dir, f"{prov}.jsonl"), encoding="utf-8") if l.strip()]
        ok = sum(a["outcome"] == "ok" for a in att)
        use = [a["usage"] for a in att if a["outcome"] == "ok"]
        tin = np.mean([u.get("input_tokens", u.get("promptTokenCount", 0)) for u in use])
        tout = np.mean([u.get("output_tokens", (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))) for u in use])
        rows.append([prov, ", ".join(sorted({str(a["model_returned"]) for a in att if a["outcome"] == "ok"})), f"{t[t.model_provider == prov].anon_image_id.nunique()} / {len(man)}",
                     len(att) - ok, f"{np.median([a['duration_s'] for a in att if a['outcome'] == 'ok']):.1f} s", f"{tin:.0f}", f"{tout:.0f}"])
    L += table(["provider", "model returned", "valid evaluations", "failed attempts", "median time", "input tokens per request", "output tokens per request (reasoning included)"], rows)

    L += ["## 2. Use of the scale", "", "Mean (standard deviation), range, and share of scores 6–7, on the images evaluated once (repeats left out). "
          "An item where almost every image is at 6–7 for every model is not informative.", ""]
    rows = []
    for it in P.SCORES:
        r = [it]
        for prov in P.PROVIDERS:
            x = first[first.model_provider == prov][it]
            r.append(f"{x.mean():.2f} ({x.std():.2f}), {x.min()}–{x.max()}, {((x >= 6).mean()):.0%}")
        x = first[it]
        r.append(f"{(x >= 6).mean():.0%}")
        rows.append(r)
    L += table(["item"] + P.PROVIDERS + ["6–7, all models"], rows)
    rows = [[c] + [f"{first[first.model_provider == p][c].mean():.2f} ({first[first.model_provider == p][c].std():.2f})" for p in P.PROVIDERS] for c in items[-2:]]
    L += table(["composite"] + P.PROVIDERS, rows)
    w = first.pivot(index="anon_image_id", columns="model_provider", values=items)
    rr = [[p, f"{stats.spearmanr(first[first.model_provider == p].architectural_appearance, first[first.model_provider == p].representation_quality).statistic:+.2f}"] for p in P.PROVIDERS]
    L += ["Correlation between the two composites within each model (if close to 1 the model does not separate architecture from representation):", ""] + table(["provider", "Spearman ρ"], rr)

    L += ["## 3. Test–retest", "", f"The same image sent twice in two independent requests ({len(reps)} images per model).", ""]
    rows = []
    for prov in P.PROVIDERS:
        s = t[t.model_provider == prov].set_index("anon_image_id")
        ok = reps[reps.first_anon_id.isin(s.index) & reps.repeat_anon_id.isin(s.index)]
        a, b = s.loc[ok.first_anon_id], s.loc[ok.repeat_anon_id]
        d = np.abs(a[P.SCORES].to_numpy() - b[P.SCORES].to_numpy())
        rows.append([prov, len(ok), f"{(d == 0).mean():.0%}", f"{(d <= 1).mean():.0%}", f"{d.mean():.2f}",
                     f"{icc21(np.column_stack([a.architectural_appearance, b.architectural_appearance])):.2f}",
                     f"{icc21(np.column_stack([a.representation_quality, b.representation_quality])):.2f}",
                     f"{np.abs(a.architectural_appearance.to_numpy() - b.architectural_appearance.to_numpy()).mean():.2f} / {np.abs(a.representation_quality.to_numpy() - b.representation_quality.to_numpy()).mean():.2f}"])
    L += table(["provider", "pairs", "items identical", "items within 1 point", "mean absolute difference (items)", "ICC architectural composite", "ICC representation composite", "mean absolute difference of the composites (arch. / repr.)"], rows)
    if stage == "full":
        rows = []
        for it in P.SCORES:
            r = [it]
            for prov in P.PROVIDERS:
                s = t[t.model_provider == prov].set_index("anon_image_id")
                ok = reps[reps.first_anon_id.isin(s.index) & reps.repeat_anon_id.isin(s.index)]
                r.append(f"{icc21(np.column_stack([s.loc[ok.first_anon_id, it], s.loc[ok.repeat_anon_id, it]])):.2f}")
            rows.append(r)
        L += ["Test–retest ICC per item:", ""] + table(["item"] + P.PROVIDERS, rows)

    L += ["## 4. Agreement between the three models", "", "On the images evaluated once. Spearman ρ between pairs of models and ICC(2,1) (absolute agreement) across the three.", ""]
    rows = []
    for it in items:
        m = w[it].dropna()
        pair = [f"{stats.spearmanr(m[a], m[b]).statistic:+.2f}" for a, b in itertools.combinations(P.PROVIDERS, 2)]
        z = (m - m.mean()) / m.std()
        rows.append([it] + pair + [f"{icc21(m.to_numpy(float)):.2f}", f"{icc21(z.to_numpy(float)):.2f}"])
    L += table(["measure"] + [f"{a} – {b}" for a, b in itertools.combinations(P.PROVIDERS, 2)] + ["ICC (raw scores)", "ICC (scores standardised within model)"], rows)

    notes_file = os.path.join(out_dir, "pilot_notes.csv")
    if stage == "pilot" and os.path.exists(notes_file):
        n = pd.read_csv(notes_file).merge(t[["anon_image_id", "model_provider", "apparent_constructability", "architectural_appearance", "representation_quality"]])
        words = ("beaut", "elegant", "attractive", "pleasing", "aesthetic", "stunning", "striking")
        L += ["## 5. Notes (pilot only)", "",
              "Share of architectural notes that use words of aesthetic appreciation (" + ", ".join(words) + "…): "
              + "; ".join(f"{p} {n[n.model_provider == p].architectural_note.str.lower().str.contains('|'.join(words)).mean():.0%}" for p in P.PROVIDERS) + ".", "",
              "Notes of the images with the lowest and the highest architectural composite, for each model:", ""]
        for p in P.PROVIDERS:
            s = n[n.model_provider == p].sort_values("architectural_appearance")
            for r in list(s.head(3).itertuples()) + list(s.tail(3).itertuples()):
                L.append(f"* **{p}**, architecture {r.architectural_appearance:.2f} (constructability {r.apparent_constructability}), representation {r.representation_quality:.2f} — "
                         f"*architecture:* {r.architectural_note} *representation:* {r.representation_note}")
        L.append("")
    rep_dir = os.path.join(P.HERE, "..", "vlm_report")
    os.makedirs(rep_dir, exist_ok=True)
    open(os.path.join(rep_dir, f"{stage}_instrument_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
