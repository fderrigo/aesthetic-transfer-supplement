"""
ARCH300 — the participant-level human estimates reported in the manuscript, recomputed from the export.

    python paper_human_estimates.py <extracted export folder> [--out human_report]

The manuscript reports the estimates computed by the data-collection platform: means of within-participant differences
with one-sample t intervals, Bradley–Terry odds ratios with standard errors clustered by participant, intraclass
correlations on participant-centred scores, the leave-one-rater-out preference signal. This script is an independent
pandas / scipy / statsmodels implementation of the same definitions (it was used to check the platform: no discrepancy
on the values it covers). `arch300_analysis.py` analyses the same data with mixed models and gives close, not identical,
values.
Exclusions are applied through the `excluded_from_analysis` flags of the export.
Writes <out>/report_paper_participant_level.md and results_paper_participant_level.csv.
The file also lists quantities that the manuscript does not use (equivalence intervals, Davidson model, sample sizes).
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_report"))
args = ap.parse_args()
exp = args.export
as_bool = lambda s: s.astype(str).str.lower().eq("true")
R = pd.read_csv(f"{exp}/posttraining_ratings.csv"); T = pd.read_csv(f"{exp}/pairwise_trials.csv")
PR = pd.read_csv(f"{exp}/pretraining_ratings.csv"); P = pd.read_csv(f"{exp}/participants.csv", keep_default_na=False)
SI = pd.read_csv(f"{exp}/source_images.csv"); PS = pd.read_csv(f"{exp}/posttraining_sessions.csv")
R = R[~as_bool(R.excluded_from_analysis)].copy(); T = T[~as_bool(T.excluded_from_analysis)].copy(); PR = PR[~as_bool(PR.excluded_from_analysis)].copy()
T = T[T.selected_side.isin(["Left", "Right", "Tie"])].copy()          # answered comparisons only

LABEL = {
    "diff_participant": "AESTHETIC − CONTROL, ratings, mean of the within-participant differences (points)",
    "dz": "AESTHETIC − CONTROL, standardised effect d_z (mean difference / SD of the differences)",
    "tost_ratings": "AESTHETIC − CONTROL, ratings, 90% interval for the equivalence test (region ±0.3)",
    "mde_points": "minimum detectable effect at 80% power, ratings (points)",
    "n_margin": "participants needed to detect 0.3 points", "n_observed": "participants needed to detect the observed effect",
    "wilcoxon p": "AESTHETIC − CONTROL, ratings, Wilcoxon signed-rank test",
    "inf.f.base_control": "BASE − CONTROL, ratings (points)", "inf.f.aesthetic_base": "AESTHETIC − BASE, ratings (points)",
    "diff_triplet": "AESTHETIC − CONTROL, ratings, mean over the triplets (participant-centred scores)",
    "share_participant": "AESTHETIC chosen against CONTROL, share of the decisive comparisons, mean over participants",
    "n_pairs": "participants needed to detect the observed pairwise share",
    "inf.f.pairs_ab": "AESTHETIC chosen against BASE, share, mean over participants", "inf.f.pairs_bc": "BASE chosen against CONTROL, share, mean over participants",
    "pairs_share_half": "AESTHETIC against CONTROL with ties counted as half", "pairs_triplet": "AESTHETIC chosen against CONTROL, share, mean over the triplets",
    "pairs_ties": "share of ties, AESTHETIC vs CONTROL", "left_share": "share of choices of the image on the left",
    "bt_aesthetic": "Bradley–Terry odds ratio AESTHETIC vs CONTROL", "bt_position": "Bradley–Terry odds ratio of the left position",
    "bt_base": "Bradley–Terry odds ratio BASE vs CONTROL", "bt_aesthetic_base": "Bradley–Terry odds ratio AESTHETIC vs BASE",
    "tost_pairs": "Bradley–Terry odds ratio AESTHETIC vs CONTROL, 90% interval for the equivalence test (region 0.8–1.25)",
    "mde_or": "minimum detectable odds ratio at 80% power",
    "group_interaction": "AESTHETIC − CONTROL: participants linked to phase 1 minus participants of phase 2 only (points)",
    "pairs_linked": "pairwise share: linked minus phase 2 only", "order": "AESTHETIC − CONTROL: ratings first minus pairs first", "pairs_order": "pairwise share: ratings first minus pairs first",
    "fatigue": "slope of the ratings along the session (points per 60 items)",
    "pairs_agree_participant": "ratings difference vs pairwise share, across participants (Spearman)", "pairs_agree_triplet": "ratings difference vs pairwise share, across triplets (Spearman)",
    "convergent": "image mean rating vs share of pairwise wins (Spearman)",
    "icc1 phase2": "phase 2: ICC of a single rating", "icck phase2": "phase 2: ICC of the image mean", "icc1 phase1": "phase 1: ICC of a single rating", "icck phase1": "phase 1: ICC of the photograph mean",
    "consensus_agreement": "phase 1: correlation of each participant with the mean of the others", "consensus_signal": "phase 1: AESTHETIC − CONTROL photographs, same ratings used for the selection (points)",
    "consensus_signal_loo": "phase 1: AESTHETIC − CONTROL photographs, leave-one-rater-out (points)", "transfer_ratio": "phase-2 difference / leave-one-rater-out signal",
    "inf.f.dav_aesthetic": "Davidson model with ties: odds ratio AESTHETIC vs CONTROL", "inf.f.dav_base": "Davidson model with ties: odds ratio BASE vs CONTROL", "dav_tie": "Davidson model: probability of a tie at parity",
    "effect_scene": "AESTHETIC − CONTROL in SD of the generated scenes", "pairs_expected": "pairwise preference expected from the ratings",
}
GROUPS = {"inf.g.linked": "participants linked to phase 1", "inf.g.declared": "declared phase 1, not linked", "inf.g.only2": "phase 2 only"}
rows = []
def G(key, arg=None, section=None):
    return key
def chk(name, row, est=None, lo=None, hi=None, p=None, n=None, tol=None):
    lab = LABEL.get(name, name)
    for g, text in GROUPS.items():
        if name.endswith(" " + g):
            kind = name.split(" ")[0]
            lab = {"group_diff": "AESTHETIC − CONTROL, ratings (points)", "group_bt": "Bradley–Terry odds ratio AESTHETIC vs CONTROL"}[kind] + ": " + text
    rows.append(dict(quantity=lab, estimate=est, ci_low=lo, ci_high=hi, p=p, n=n))

def t1(x, mu=0):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / math.sqrt(n); q = stats.t.ppf(.975, n - 1)
    return m, m - q * se, m + q * se, stats.ttest_1samp(x, mu).pvalue, n, se
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); r = stats.ttest_ind(a, b, equal_var=False)
    ci = r.confidence_interval(.95); return a.mean() - b.mean(), ci.low, ci.high, r.pvalue, len(a) + len(b)

# ------------------------------------------------ phase 2: ratings
def pdiff(r, a, b, minn=3):
    c = r[r.condition_code.isin([a, b])].groupby(["participant_id", "condition_code"]).score.agg(["mean", "count"]).unstack()
    ok = (c["count"][a] >= minn) & (c["count"][b] >= minn)
    return (c["mean"][a] - c["mean"][b])[ok].dropna()
dac = pdiff(R, "AESTHETIC", "CONTROL")
m, lo, hi, p, n, se = t1(dac); chk("diff_participant", G("inf.f.diff_participant"), m, lo, hi, p, n)
chk("dz", G("inf.f.dz"), m / dac.std(ddof=1))
q90 = stats.t.ppf(.95, n - 1); chk("tost_ratings", G("inf.f.tost_ratings"), m, m - q90 * se, m + q90 * se)
chk("mde_points", G("inf.f.mde_points"), (stats.t.ppf(.975, n - 1) + stats.t.ppf(.8, n - 1)) * se)
sd = dac.std(ddof=1)
ss = lambda dz: math.ceil(((stats.norm.ppf(.975) + stats.norm.ppf(.8)) / abs(dz)) ** 2 + stats.norm.ppf(.975) ** 2 / 2)
chk("n_margin", G("inf.f.n_margin"), ss(0.3 / sd)); chk("n_observed", G("inf.f.n_observed"), ss(m / sd))
w = stats.wilcoxon(dac, correction=False, method="approx"); chk("wilcoxon p", G("inf.f.wilcoxon"), p=w.pvalue)
for key, (a, b) in {"inf.f.base_control": ("BASE", "CONTROL"), "inf.f.aesthetic_base": ("AESTHETIC", "BASE")}.items():
    m_, lo_, hi_, p_, n_, _ = t1(pdiff(R, a, b)); chk(key, G(key), m_, lo_, hi_, p_, n_)
R["c"] = R.score - R.groupby("participant_id").score.transform("mean")
R["trip"] = R.prompt_code + "_" + R.seed.astype(str)
tt = R[R.condition_code.isin(["AESTHETIC", "CONTROL"])].groupby(["trip", "condition_code"]).c.mean().unstack().dropna()
m_, lo_, hi_, p_, n_, _ = t1(tt.AESTHETIC - tt.CONTROL); chk("diff_triplet", G("inf.f.diff_triplet"), m_, lo_, hi_, p_, n_)

# ------------------------------------------------ phase 2: pairwise
T["trip"] = T.prompt_code + "_" + T.seed.astype(str)
D = T[T.selected_side != "Tie"].copy()
def pair(df, a, b): return df[df.left_condition.isin([a, b]) & df.right_condition.isin([a, b])]
def share(df, a, b): x = pair(df, a, b); return (x.winner_condition == a).groupby(x.participant_id).mean()
sac = share(D, "AESTHETIC", "CONTROL")
m, lo, hi, p, n, se_s = t1(sac, .5); chk("share_participant", G("inf.f.share_participant"), m, lo, hi, p, n)
chk("n_pairs", G("inf.f.n_pairs"), ss((m - .5) / sac.std(ddof=1)))
for key, (a, b) in {"inf.f.pairs_ab": ("AESTHETIC", "BASE"), "inf.f.pairs_bc": ("BASE", "CONTROL")}.items():
    m_, lo_, hi_, p_, n_, _ = t1(share(D, a, b), .5); chk(key, G(key), m_, lo_, hi_, p_, n_)
x = pair(T, "AESTHETIC", "CONTROL"); half = x.winner_condition.map({"AESTHETIC": 1.0, "CONTROL": 0.0}).fillna(0.5).groupby(x.participant_id).mean()
m_, lo_, hi_, p_, n_, _ = t1(half, .5); chk("pairs_share_half", G("inf.f.pairs_share_half"), m_, lo_, hi_, p_, n_)
x = pair(D, "AESTHETIC", "CONTROL"); bt_ = (x.winner_condition == "AESTHETIC").groupby(x.trip).mean()
m_, lo_, hi_, p_, n_, _ = t1(bt_, .5); chk("pairs_triplet", G("inf.f.pairs_triplet"), m_, lo_, hi_, p_, n_)
x = pair(T, "AESTHETIC", "CONTROL"); chk("pairs_ties", G("inf.f.pairs_ties"), (x.selected_side == "Tie").mean(), n=len(x))
chk("left_share", G("inf.f.left_share"), (D.selected_side == "Left").mean(), n=len(D))
# Bradley–Terry with cluster-robust errors and t(G-1) inference
def bt(df, ref):
    others = [c for c in ["AESTHETIC", "CONTROL", "BASE"] if c != ref]
    X = pd.DataFrame({"const": 1.0, **{c: (df.left_condition == c).astype(float) - (df.right_condition == c).astype(float) for c in others}})
    r = sm.GLM((df.selected_side == "Left").astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": df.participant_id.values})
    g = df.participant_id.nunique(); q = stats.t.ppf(.975, g - 1)
    return {c: (math.exp(r.params[c]), math.exp(r.params[c] - q * r.bse[c]), math.exp(r.params[c] + q * r.bse[c]), 2 * stats.t.sf(abs(r.params[c] / r.bse[c]), g - 1), r.bse[c], g) for c in X.columns}
b = bt(D, "CONTROL")
chk("bt_aesthetic", G("inf.f.bt_aesthetic"), *b["AESTHETIC"][:4], n=len(D)); chk("bt_position", G("inf.f.bt_position"), *b["const"][:4])
chk("bt_base", G("inf.f.bt_base"), *b["BASE"][:4]); chk("bt_aesthetic_base", G("inf.f.bt_aesthetic_base"), *bt(D, "BASE")["AESTHETIC"][:4])
chk("mde_or", G("inf.f.mde_or"), math.exp((stats.t.ppf(.975, b["AESTHETIC"][5] - 1) + stats.t.ppf(.8, b["AESTHETIC"][5] - 1)) * b["AESTHETIC"][4]))

# ------------------------------------------------ groups, order
P["pid"] = P.participant_id.astype(int)
linked = set(P[P.has_pre_training_session.astype(str).str.lower() == "true"].pid); declared = set(P[(P.prior_participation_self_report == "Yes")].pid) - linked
only2 = set(P.pid) - linked - declared
for lab, s_ in (("inf.g.linked", linked), ("inf.g.declared", declared), ("inf.g.only2", only2)):
    m_, lo_, hi_, p_, n_, _ = t1(dac[dac.index.isin(s_)]); chk(f"group_diff {lab}", G("inf.f.group_diff", lab), m_, lo_, hi_, p_, n_)
    dd = D[D.participant_id.isin(s_)]; chk(f"group_bt {lab}", G("inf.f.group_bt", lab), *bt(dd, "CONTROL")["AESTHETIC"][:4], n=len(dd))
chk("group_interaction", G("inf.f.group_interaction"), *welch(dac[dac.index.isin(linked)], dac[dac.index.isin(only2)]))
chk("pairs_linked", G("inf.f.pairs_linked"), *welch(sac[sac.index.isin(linked)], sac[sac.index.isin(only2)]))
first = PS.sort_values("started_at").groupby("participant_id")["mode"].first(); rf = set(first[first == "PostTrainingRating"].index)
chk("order", G("inf.f.order"), *welch(dac[dac.index.isin(rf)], dac[~dac.index.isin(rf)]))
chk("pairs_order", G("inf.f.pairs_order"), *welch(sac[sac.index.isin(rf)], sac[~sac.index.isin(rf)]))
# fatigue slope per 60 items
sl = [np.polyfit(g.position / 60.0, g.c, 1)[0] for _, g in R.groupby("participant_id") if len(g) >= 10]
m_, lo_, hi_, p_, n_, _ = t1(sl); chk("fatigue", G("inf.f.fatigue"), m_, lo_, hi_, p_, n_)
# agreement of the two tasks
both = pd.concat([dac.rename("d"), sac.rename("s")], axis=1).dropna(); r_ = stats.spearmanr(both.d, both.s)
chk("pairs_agree_participant", G("inf.f.pairs_agree_participant"), r_.statistic, p=r_.pvalue, n=len(both))
bothT = pd.concat([(tt.AESTHETIC - tt.CONTROL).round(9).rename("d"), bt_.rename("s")], axis=1).dropna(); r_ = stats.spearmanr(bothT.d, bothT.s)
chk("pairs_agree_triplet", G("inf.f.pairs_agree_triplet"), r_.statistic, p=r_.pvalue, n=len(bothT))
wins = pd.concat([pd.DataFrame({"img": D.left_generated_image_id, "won": (D.selected_side == "Left") * 1.0}), pd.DataFrame({"img": D.right_generated_image_id, "won": (D.selected_side == "Right") * 1.0})]).groupby("img").won.agg(["mean", "count"])
cv = wins[wins["count"] >= 3].join(R.groupby("generated_image_id").c.mean().rename("rating"), how="inner"); r_ = stats.spearmanr(cv.rating.round(9), cv["mean"].round(9))
chk("convergent", G("inf.f.convergent"), r_.statistic, n=len(cv))

# ------------------------------------------------ reliability (centred scores, df corrected for the centring)
def icc(df, item, rater, score):
    g = df.groupby(item)[score].agg(["mean", "count", "var"]); g = g[g["count"] >= 2]; x = df[df[item].isin(g.index)]
    a, N, grand = len(g), g["count"].sum(), x[score].mean()
    msb = (g["count"] * (g["mean"] - grand) ** 2).sum() / (a - 1)
    msw = (g["var"] * (g["count"] - 1)).sum() / (N - a - (df[rater].nunique() - 1))
    k0 = (N - (g["count"] ** 2).sum() / N) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw), (msb - msw) / msb, a
i1, ik, a = icc(R, "generated_image_id", "participant_id", "c"); chk("icc1 phase2", G("inf.f.icc1", "inf.p.phase2"), i1, n=a); chk("icck phase2", G("inf.f.icck", "inf.p.phase2"), ik)
PR["c"] = PR.score - PR.groupby("participant_id").score.transform("mean")
i1, ik, a = icc(PR, "image_asset_id", "participant_id", "c"); chk("icc1 phase1", G("inf.f.icc1", "inf.p.phase1"), i1, n=a); chk("icck phase1", G("inf.f.icck", "inf.p.phase1"), ik)

# ------------------------------------------------ consensus: agreement, circular and leave-one-rater-out signal
def latest(fn):
    d = pd.read_csv(f"{exp}/{fn}"); d = d[d.is_frozen.astype(str).str.lower() == "true"]; return set(d[d.version == d.version.max()].image_code)
AES, CTL = latest("aesthetic_dataset.csv"), latest("control_dataset.csv")
tot = PR.groupby("image_code").score.agg(["sum", "count"])
x = PR.join(tot, on="image_code"); x = x[x["count"] > 1].assign(loo=lambda z: (z["sum"] - z.score) / (z["count"] - 1))
rs = [g.score.corr(g.loo) for _, g in x.groupby("participant_id") if len(g) >= 10]; rs = [v for v in rs if not math.isnan(v)]
m_, lo_, hi_, _, n_, _ = t1(rs); chk("consensus_agreement", G("inf.f.consensus_agreement"), m_, lo_, hi_, n=n_)
per = []
for _, g in PR.groupby("participant_id"):
    a_, c_ = g[g.image_code.isin(AES)].score, g[g.image_code.isin(CTL)].score
    if len(a_) >= 3 and len(c_) >= 3: per.append(a_.mean() - c_.mean())
m_, lo_, hi_, p_, n_, _ = t1(per); chk("consensus_signal", G("inf.f.consensus_signal"), m_, lo_, hi_, n=n_)
active = set(SI[SI.is_active.astype(str).str.lower() == "true"].image_code); ids = SI.set_index("image_code").image_asset_id
loo = []
for _, g in PR.groupby("participant_id"):
    own = g.set_index("image_code").score
    t_ = tot[tot.index.isin(active)].copy(); mean = t_["sum"] / t_["count"]
    idx = own.index.intersection(t_.index)
    mean.loc[idx] = np.where(t_.loc[idx, "count"] > 1, (t_.loc[idx, "sum"] - own.loc[idx]) / (t_.loc[idx, "count"] - 1).replace(0, np.nan), np.nan)
    order = pd.DataFrame({"m": mean.dropna()}); order["id"] = ids.reindex(order.index).values
    top = set(order.sort_values(["m", "id"], ascending=[False, True]).index[:len(AES)])
    a_, c_ = own[own.index.isin(top)], own[own.index.isin(CTL - top)]
    if len(a_) >= 3 and len(c_) >= 3: loo.append(a_.mean() - c_.mean())
m_, lo_, hi_, p_, n_, _ = t1(loo); chk("consensus_signal_loo", G("inf.f.consensus_signal_loo"), m_, lo_, hi_, n=n_)
chk("transfer_ratio", G("inf.f.transfer_ratio"), dac.mean() / m_)

# ------------------------------------------------ Davidson model (ties as a third outcome): generic optimiser, numerical derivatives
from scipy.optimize import minimize
def davidson(df, ref):
    others = [c for c in ["AESTHETIC", "CONTROL", "BASE"] if c != ref]
    X = np.column_stack([np.ones(len(df))] + [(df.left_condition == c).astype(float) - (df.right_condition == c).astype(float) for c in others])
    y = df.selected_side.map({"Left": 1, "Right": 0, "Tie": 2}).values
    def ll_i(th):
        eta = X @ th[:-1]; w = np.column_stack([np.zeros(len(df)), eta, th[-1] + eta / 2])      # right, left, tie
        return w[np.arange(len(df)), y] - np.log(np.exp(w).sum(1))
    r = minimize(lambda th: -ll_i(th).sum(), np.zeros(X.shape[1] + 1), method="BFGS", options={"gtol": 1e-10})
    th, m, h = r.x, len(r.x), 1e-5
    S = np.column_stack([(ll_i(th + h * np.eye(m)[j]) - ll_i(th - h * np.eye(m)[j])) / (2 * h) for j in range(m)])   # per-observation scores
    H = np.zeros((m, m))
    for j in range(m):
        for k in range(m):
            e1, e2 = h * 10 * np.eye(m)[j], h * 10 * np.eye(m)[k]
            H[j, k] = -(ll_i(th + e1 + e2).sum() - ll_i(th + e1 - e2).sum() - ll_i(th - e1 + e2).sum() + ll_i(th - e1 - e2).sum()) / (4 * (h * 10) ** 2)
    g = pd.Series(df.participant_id.values); Sg = pd.DataFrame(S).groupby(g.values).sum().values; Gn, n = len(Sg), len(df)
    V = np.linalg.inv(H) @ (Sg.T @ Sg) @ np.linalg.inv(H) * Gn / (Gn - 1) * (n - 1) / (n - m)
    se = np.sqrt(np.diag(V)); q = stats.t.ppf(.975, Gn - 1)
    names = ["const"] + others + ["tie"]
    return {nm: (th[i], th[i] - q * se[i], th[i] + q * se[i], 2 * stats.t.sf(abs(th[i] / se[i]), Gn - 1)) for i, nm in enumerate(names)}
dv = davidson(T, "CONTROL")
for key, nm in (("inf.f.dav_aesthetic", "AESTHETIC"), ("inf.f.dav_base", "BASE")):
    e_, lo_, hi_, p_ = dv[nm]; chk(key, G(key), math.exp(e_), math.exp(lo_), math.exp(hi_), p_, n=len(T), tol=1e-3)
tie = lambda d_: math.exp(d_) / (2 + math.exp(d_))
chk("dav_tie", G("inf.f.dav_tie"), tie(dv["tie"][0]), tie(dv["tie"][1]), tie(dv["tie"][2]), tol=1e-3)
_ = ("Davidson (independent): OR AESTHETIC vs CONTROL %.3f (%.3f-%.3f) p %.3f | tie probability at parity %.3f" % (math.exp(dv["AESTHETIC"][0]), math.exp(dv["AESTHETIC"][1]), math.exp(dv["AESTHETIC"][2]), dv["AESTHETIC"][3], tie(dv["tie"][0])))

# ------------------------------------------------ effect on the scale of the generated scenes; pairwise preference expected from the ratings
def vc(df, item, rater, score):
    g = df.groupby(item)[score].agg(["mean", "count", "var"]); g = g[g["count"] >= 2]; x = df[df[item].isin(g.index)]
    a, N, grand = len(g), g["count"].sum(), x[score].mean()
    msb = (g["count"] * (g["mean"] - grand) ** 2).sum() / (a - 1)
    msw = (g["var"] * (g["count"] - 1)).sum() / (N - a - (df[rater].nunique() - 1))
    k0 = (N - (g["count"] ** 2).sum() / N) / (a - 1)
    return max(0.0, (msb - msw) / k0), msw
m_, lo_, hi_, _, _, _ = t1(dac)
sd_scene = math.sqrt(vc(R, "trip", "participant_id", "c")[0]); chk("effect_scene", G("inf.f.effect_scene"), m_ / sd_scene, lo_ / sd_scene, hi_ / sd_scene)
sd_noise = math.sqrt(2 * vc(R, "generated_image_id", "participant_id", "c")[1])
chk("pairs_expected", G("inf.f.pairs_expected"), stats.norm.cdf(m_ / sd_noise), stats.norm.cdf(lo_ / sd_noise), stats.norm.cdf(hi_ / sd_noise))

# additions for the manuscript: interval of d_z, equivalence interval of the odds ratio
m_, lo_, hi_, _, n_, _ = t1(dac); s_ = dac.std(ddof=1)
for r_ in rows:
    if r_["quantity"] == LABEL["dz"]:
        r_["ci_low"], r_["ci_high"], r_["n"] = lo_ / s_, hi_ / s_, n_
g_ = b["AESTHETIC"][5]; q90 = stats.t.ppf(.95, g_ - 1); beta, se_b = math.log(b["AESTHETIC"][0]), b["AESTHETIC"][4]
chk("tost_pairs", None, math.exp(beta), math.exp(beta - q90 * se_b), math.exp(beta + q90 * se_b), None, len(D))

out = pd.DataFrame(rows)
os.makedirs(args.out, exist_ok=True)
out.to_csv(os.path.join(args.out, "results_paper_participant_level.csv"), index=False)
blank = lambda v: v is None or (isinstance(v, float) and math.isnan(v))
f = lambda v: "" if blank(v) else (f"{v:.0f}" if abs(v) >= 5 and float(v).is_integer() else f"{v:+.3f}")
fp = lambda v: "" if blank(v) else ("< .0001" if v < 0.0001 else f"{v:.4f}".replace("0.", "."))
L = ["# ARCH300 — participant-level human estimates (as reported in the manuscript)", "",
     "Independent recomputation from the export of the estimates that the manuscript reports from the data-collection platform. "
     "Definitions in `paper_human_estimates.py`; the mixed-model analysis of the same data is in `report_human_votes.md`.", "",
     f"Phase 2 after exclusions: {len(R)} ratings from {R.participant_id.nunique()} participants; {len(T)} answered comparisons "
     f"({int((T.selected_side == 'Tie').sum())} ties, {len(D)} decisive) from {T.participant_id.nunique()} participants. "
     f"Phase 1 after exclusions: {len(PR)} ratings from {PR.participant_id.nunique()} participants.", "",
     "| quantity | estimate | 95% CI (90% where stated) | p | n |", "|---|---|---|---|---|"]
for r_ in rows:
    ci = "" if blank(r_["ci_low"]) else f"{f(r_['ci_low'])} / {f(r_['ci_high'])}"
    L.append(f"| {r_['quantity']} | {f(r_['estimate'])} | {ci} | {fp(r_['p'])} | {'' if blank(r_['n']) else int(r_['n'])} |")
open(os.path.join(args.out, "report_paper_participant_level.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
