"""Triplet-level analysis (one triplet = the BASE, CONTROL and AESTHETIC images of one prompt and seed) of the ARCH300 export.
Writes triplets.csv, triplet_key.csv and triplet_report.txt into the output folder."""
import sys
import json
import random
import numpy as np
import pandas as pd
from scipy import stats

D, OUT = sys.argv[1], sys.argv[2]
C = ["BASE", "CONTROL", "AESTHETIC"]

def as_bool(s):
    return s.astype(str).str.lower().eq("true")

g = pd.read_csv(f"{D}/generated_images.csv")
plans = pd.read_csv(f"{D}/generation_plan.csv")
main_plans = set(plans.generation_plan_id)
g = g[g.generation_plan_id.isin(main_plans)].copy()
prompts = pd.read_csv(f"{D}/prompt_sets.csv")
set_id = int(plans.prompt_set_id.iloc[0])
prompts = prompts[prompts.prompt_set_id == set_id].set_index("prompt_code")
people = pd.read_csv(f"{D}/participants.csv")
excluded_people = set(people.loc[as_bool(people.excluded_from_analysis), "participant_id"])
r = pd.read_csv(f"{D}/posttraining_ratings.csv")
r = r[~as_bool(r.excluded_from_analysis) & ~r.participant_id.isin(excluded_people)]
p = pd.read_csv(f"{D}/pairwise_trials.csv")
p = p[~as_bool(p.excluded_from_analysis) & ~p.participant_id.isin(excluded_people)]

g["triplet"] = g.prompt_code + "_" + g.seed.astype(str)
r["triplet"] = r.prompt_code + "_" + r.seed.astype(str)
p["triplet"] = p.prompt_code + "_" + p.seed.astype(str)

# per-participant centring: removes "strict" vs "generous" raters (the largest source of variance in the mixed model)
r["centred"] = r.score - r.groupby("participant_id").score.transform("mean")

rows, key = [], []
rnd = random.Random(20260928)
for t, grp in g.groupby("triplet"):
    grp = grp.set_index("condition_code")
    code, seed = grp.prompt_code.iloc[0], int(grp.seed.iloc[0])
    row = {"triplet": t, "prompt_code": code, "seed": seed, "category": prompts.loc[code, "category"] if code in prompts.index else "",
           "prompt": prompts.loc[code, "text"] if code in prompts.index else "",
           "active": bool(as_bool(grp.is_active).all()),
           "excluded_by_review_rule": bool(as_bool(grp.excluded_by_review_rule).any()),
           "exclusion_reason": next((x for x in grp.exclusion_reason if isinstance(x, str) and x), "")}
    tr = r[r.triplet == t]
    for c in C:
        row[f"defects_{c}"] = grp.loc[c, "review_defects"] if c in grp.index and isinstance(grp.loc[c, "review_defects"], str) else ""
        s = tr[tr.condition_code == c]
        row[f"n_{c}"] = len(s)
        row[f"mean_{c}"] = round(s.score.mean(), 3) if len(s) else np.nan
        row[f"centred_{c}"] = round(s.centred.mean(), 3) if len(s) else np.nan
        row[f"image_{c}"] = grp.loc[c, "opaque_id"] if c in grp.index else ""
    tp = p[p.triplet == t]
    for a, b in [("AESTHETIC", "CONTROL"), ("AESTHETIC", "BASE"), ("BASE", "CONTROL")]:
        m = tp[{tp_l for tp_l in [a, b]} == {a, b}] if False else tp[((tp.left_condition == a) & (tp.right_condition == b)) | ((tp.left_condition == b) & (tp.right_condition == a))]
        row[f"pair_{a}_vs_{b}_wins_{a}"] = int((m.winner_condition == a).sum())
        row[f"pair_{a}_vs_{b}_wins_{b}"] = int((m.winner_condition == b).sum())
        row[f"pair_{a}_vs_{b}_ties"] = int((m.winner_condition == "TIE").sum())
    rated = {c: row[f"centred_{c}"] for c in C if not np.isnan(row[f"centred_{c}"])}
    row["best_by_rating"] = max(rated, key=rated.get) if len(rated) == 3 else ""
    rows.append(row)
    # blind sheet order (A, B, C) for the image files, with the key kept apart
    order = C[:]
    rnd.shuffle(order)
    key.append({"triplet": t, "A": order[0], "B": order[1], "C": order[2],
                "image_A": row[f"image_{order[0]}"], "image_B": row[f"image_{order[1]}"], "image_C": row[f"image_{order[2]}"]})

T = pd.DataFrame(rows).sort_values(["prompt_code", "seed"])
T.to_csv(f"{OUT}/triplets.csv", index=False)
pd.DataFrame(key).sort_values("triplet").to_csv(f"{OUT}/triplet_key.csv", index=False)

# ---------------------------------------------------------------- report
L = []
say = L.append
act = T[T.active]
say("TRIPLET ANALYSIS — ARCH300 (one triplet = BASE, CONTROL and AESTHETIC generated from the same prompt and seed)")
say(f"Triplets in the main plans: {len(T)}; active (shown to participants): {len(act)}; excluded by the blind review rule: {int(T.excluded_by_review_rule.sum())}")
say(f"Ratings used: {len(r)} from {r.participant_id.nunique()} participants; pairwise trials: {len(p)} from {p.participant_id.nunique()} participants")
say("Note: each image has only 1–4 ratings, so single triplets are very noisy; conclusions come from the counts over all triplets.\n")

say("1. BLIND REVIEW (defects marked before the collection)")
for c in C:
    d = T[f"defects_{c}"].astype(str).str.len().gt(0).sum()
    say(f"   {c:>9}: {d} images with at least one defect")
say("   defect categories: " + ", ".join(f"{k}={v}" for k, v in pd.Series(",".join(T[[f"defects_{c}" for c in C]].fillna("").astype(str).values.ravel()).split(",")).str.strip().replace("", np.nan).dropna().value_counts().items()))

full = act.dropna(subset=[f"centred_{c}" for c in C])
say(f"\n2. RATINGS PER TRIPLET (participant-centred scores; {len(full)} active triplets rated in all three conditions)")
say("   Which condition got the highest score in each triplet:")
for c, n in full.best_by_rating.value_counts().reindex(C, fill_value=0).items():
    say(f"   {c:>9} best in {n} triplets ({n/len(full):.0%})  — 33% expected if there is no difference")
for a, b in [("AESTHETIC", "CONTROL"), ("AESTHETIC", "BASE"), ("BASE", "CONTROL")]:
    d = full[f"centred_{a}"] - full[f"centred_{b}"]
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    sign = stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else float("nan")
    wil = stats.wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 5 else float("nan")
    say(f"   {a} vs {b}: {a} higher in {wins}, lower in {losses}, equal in {ties} triplets; mean difference {d.mean():+.3f} points "
        f"(median {d.median():+.3f}); sign test p = {sign:.3f}; Wilcoxon p = {wil:.3f}")

say("\n3. PAIRWISE CHOICES WITHIN EACH TRIPLET (same prompt and seed, two conditions side by side)")
for a, b in [("AESTHETIC", "CONTROL"), ("AESTHETIC", "BASE"), ("BASE", "CONTROL")]:
    wa, wb, ti = (act[f"pair_{a}_vs_{b}_wins_{a}"].sum(), act[f"pair_{a}_vs_{b}_wins_{b}"].sum(), act[f"pair_{a}_vs_{b}_ties"].sum())
    net = act[f"pair_{a}_vs_{b}_wins_{a}"] - act[f"pair_{a}_vs_{b}_wins_{b}"]
    tw, tl = int((net > 0).sum()), int((net < 0).sum())
    pv = stats.binomtest(int(wa), int(wa + wb), 0.5).pvalue if wa + wb else float("nan")
    say(f"   {a} vs {b}: {a} chosen {wa} times, {b} {wb}, no preference {ti} → {a} {wa/(wa+wb):.1%} of decisive choices (binomial p = {pv:.3f}, not adjusted for repeated participants); "
        f"{a} ahead in {tw} triplets, behind in {tl}")

say("\n4. BY BUILDING CATEGORY (participant-centred AESTHETIC − CONTROL, active triplets rated in both)")
both = act.dropna(subset=["centred_AESTHETIC", "centred_CONTROL"]).copy()
both["d"] = both.centred_AESTHETIC - both.centred_CONTROL
cat = both.groupby("category").d.agg(["count", "mean"]).sort_values("mean")
for k, v in cat.iterrows():
    say(f"   {k:>12}: {v['mean']:+.2f} over {int(v['count'])} triplets")
say("   (2–24 triplets per category: descriptive only, no test)")

say("\n5. TRIPLETS WITH THE LARGEST DIFFERENCES (participant-centred, AESTHETIC − CONTROL; few ratings each: look at the images, do not over-interpret)")
top = both.sort_values("d")
for label, part in [("AESTHETIC much lower", top.head(8)), ("AESTHETIC much higher", top.tail(8).iloc[::-1])]:
    say(f"   {label}:")
    for _, x in part.iterrows():
        say(f"     {x.triplet:<12} {x.category:<11} Δ {x.d:+.2f}  (A {x.mean_AESTHETIC:.2f}/n{x.n_AESTHETIC}, C {x.mean_CONTROL:.2f}/n{x.n_CONTROL}, B {x.mean_BASE:.2f}/n{x.n_BASE})")
open(f"{OUT}/triplet_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
