"""
ARCH300 — analysis of the pilot study "Investigating Aesthetic Transfer in Generative Models for Architectural Design:
A Pilot Study on Consensus-Based Fine-Tuning and Human Evaluation" (plan: ANALYSIS_PLAN.md).

Sections: 1 consensus (phase 1) · 2 aesthetic transfer (phase 2) · 2b what is transferred · 3 personal transfer ·
4 human evaluation · 5 towards a full study. Estimates with uncertainty; p values are uncorrected indications.

Usage:
    python arch300_analysis.py <export.zip | extracted folder> [--out report_folder]
                               [--metrics folder with image_metrics.csv / content_metrics.csv / triplet_key.csv]

Writes <out>/report_human_votes.md (readable report) and <out>/results_human_votes.csv (every estimate in long format).
Everything is computed from the exported CSV files; nothing is sent anywhere.
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260930)
MARGIN_POINTS, MARGIN_OR = 0.30, (0.80, 1.25)


# ============================================================================ helpers
def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def fmt_p(p: float) -> str:
    return "–" if p is None or (isinstance(p, float) and math.isnan(p)) else ("< .001" if p < 0.001 else f"{p:.3f}".replace("0.", "."))


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    def h(self, text, level=2):
        self.lines += ["", "#" * level + " " + text, ""]

    def p(self, text=""):
        self.lines.append(text)

    def table(self, header, rows):
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(x) for x in r) + " |")
        self.lines.append("")

    def add(self, section, analysis, estimate, lo=None, hi=None, p=None, n=None, note=""):
        self.rows.append(dict(section=section, analysis=analysis, estimate=estimate, ci_low=lo, ci_high=hi, p=p, n=n, note=note))


# ============================================================================ data
@dataclass
class Data:
    r: pd.DataFrame          # post-training ratings
    p: pd.DataFrame          # pairwise choices (decisive and ties)
    pre: pd.DataFrame        # phase-1 ratings
    people: pd.DataFrame
    aes_codes: set
    ctl_codes: set
    prompts: pd.DataFrame
    sessions: pd.DataFrame


def load(folder: str, apply_exclusions: bool = True) -> Data:
    rd = lambda f: pd.read_csv(os.path.join(folder, f))
    people = rd("participants.csv")
    people["excluded"] = as_bool(people.excluded_from_analysis)
    people["linked"] = as_bool(people.has_pre_training_session) if "has_pre_training_session" in people else False
    people["created"] = pd.to_datetime(people.created_at, utc=True)
    r, p, pre = rd("posttraining_ratings.csv"), rd("pairwise_trials.csv"), rd("pretraining_ratings.csv")
    if apply_exclusions:
        out = set(people.loc[people.excluded, "participant_id"])
        r = r[~as_bool(r.excluded_from_analysis) & ~r.participant_id.isin(out)]
        p = p[~as_bool(p.excluded_from_analysis) & ~p.participant_id.isin(out)]
        pre = pre[~as_bool(pre.excluded_from_analysis) & ~pre.participant_id.isin(out)]
    r = r.copy()
    r["scene"] = r.prompt_code + "·" + r.seed.astype(str)
    r["A"] = (r.condition_code == "AESTHETIC").astype(int)
    r["B"] = (r.condition_code == "BASE").astype(int)
    r["c"] = r.score - r.groupby("participant_id").score.transform("mean")   # participant-centred score
    p = p.copy()
    p["scene"] = p.prompt_code + "·" + p.seed.astype(str)

    used = dict(r.groupby("condition_code").training_dataset.agg(lambda x: x.dropna().iloc[0] if x.notna().any() else None)) if "training_dataset" in r else {}

    def latest(df, condition=None):
        code = used.get(condition)
        if code is not None and (df.dataset_code == code).any():
            return set(df[df.dataset_code == code].image_code)
        v = df[as_bool(df.is_frozen)]
        return set(v[v.version == v.version.max()].image_code) if len(v) else set()

    plan = rd("generation_plan.csv")
    ps = rd("prompt_sets.csv")
    prompts = ps[ps.prompt_set_id.isin(plan.prompt_set_id)][["prompt_code", "category", "text"]].drop_duplicates("prompt_code")
    sessions = rd("posttraining_sessions.csv")
    return Data(r, p, pre, people, latest(rd("aesthetic_dataset.csv"), "AESTHETIC"), latest(rd("control_dataset.csv"), "CONTROL"), prompts, sessions)


def decisive(p: pd.DataFrame) -> pd.DataFrame:
    q = p[p.selected_side.isin(["Left", "Right"])].copy()
    q["left_won"] = (q.selected_side == "Left").astype(int)
    for c in ["AESTHETIC", "BASE"]:
        q[c] = (q.left_condition == c).astype(int) - (q.right_condition == c).astype(int)
    return q


# ============================================================================ models
@dataclass
class Est:
    est: float
    se: float
    p: float

    @property
    def lo(self): return self.est - 1.96 * self.se

    @property
    def hi(self): return self.est + 1.96 * self.se


def lmm(r: pd.DataFrame, slopes=True, extra="", reml=True):
    """Crossed random effects via variance components (one dummy group)."""
    df = r.copy()
    df["one"] = 1
    vc = {"participant": "0 + C(participant_id)", "prompt": "0 + C(prompt_code)", "scene": "0 + C(scene)"}
    if slopes:
        vc.update({"slopeA": "0 + C(participant_id):A", "slopeB": "0 + C(participant_id):B"})
    formula = "score ~ A + B" + extra
    try:
        m = smf.mixedlm(formula, df, groups="one", vc_formula=vc, re_formula="0").fit(reml=reml, method="lbfgs")
        if not np.isfinite(m.llf) and slopes:
            raise ValueError
        return m, slopes
    except Exception:
        if not slopes:
            raise
        return lmm(r, slopes=False, extra=extra, reml=reml)


def coef(m, name) -> Est:
    b, se = float(m.fe_params[name]), float(np.sqrt(m.cov_params().loc[name, name]))
    return Est(b, se, 2 * (1 - stats.norm.cdf(abs(b / se))))


def contrast(m, a, b) -> Est:
    cv = m.cov_params()
    d = float(m.fe_params[a] - m.fe_params[b])
    se = float(np.sqrt(cv.loc[a, a] + cv.loc[b, b] - 2 * cv.loc[a, b]))
    return Est(d, se, 2 * (1 - stats.norm.cdf(abs(d / se))))


def total_sd(m) -> float:
    return float(np.sqrt(sum(m.vcomp) + m.scale))


def bt(q: pd.DataFrame):
    return smf.glm("left_won ~ 1 + AESTHETIC + BASE", q, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": q.participant_id})


def bt_est(m, name) -> Est:
    return Est(float(m.params[name]), float(m.bse[name]), float(m.pvalues[name]))


def bic(llf, k, n):
    return -2 * llf + k * math.log(n)


# ============================================================================ sections
def section_a(d: Data, rep: Report, label="all participants"):
    r, q = d.r, decisive(d.p)
    rep.h(f"2. Aesthetic transfer — AESTHETIC, CONTROL and BASE ({label})")
    rep.p(f"Ratings: {len(r)} from {r.participant_id.nunique()} participants · pairwise: {len(d.p)} choices "
          f"({len(q)} decisive, {len(d.p) - len(q)} no preference = {(len(d.p) - len(q)) / max(1, len(d.p)):.1%}) from {d.p.participant_id.nunique()} participants.")
    rep.table(["condition", "N ratings", "mean", "SD", "participant-centred mean"],
              [[c, len(g), f"{g.score.mean():.2f}", f"{g.score.std():.2f}", f"{g.c.mean():+.3f}"] for c, g in r.groupby("condition_code")])

    m, slopes = lmm(r)
    a, b, ab = coef(m, "A"), coef(m, "B"), contrast(m, "A", "B")
    sd = total_sd(m)
    rep.p(f"**1. Linear mixed model** (random intercepts participant, prompt, scene{'; random condition slopes by participant' if slopes else ' — slopes did not converge, intercepts only'}):")
    rep.table(["contrast", "difference (points)", "95% CI", "standardised d", "p"],
              [["**AESTHETIC − CONTROL**", f"{a.est:+.3f}", f"{a.lo:+.3f} / {a.hi:+.3f}", f"{a.est / sd:+.3f}", fmt_p(a.p)],
               ["BASE − CONTROL", f"{b.est:+.3f}", f"{b.lo:+.3f} / {b.hi:+.3f}", f"{b.est / sd:+.3f}", fmt_p(b.p)],
               ["AESTHETIC − BASE", f"{ab.est:+.3f}", f"{ab.lo:+.3f} / {ab.hi:+.3f}", f"{ab.est / sd:+.3f}", fmt_p(ab.p)]])
    names = m.model.exog_vc.names
    rep.p("Variance components: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(names, m.vcomp)) + f", residual {m.scale:.3f}.")
    # the same difference on the scale of phase 2: spread between generated scenes (prompt + scene variance of the mixed model),
    # and the pairwise preference it predicts if a choice goes to the image the person would rate higher
    vcs = dict(zip(names, m.vcomp))
    sd_scene = math.sqrt(max(0.0, vcs.get("prompt", 0.0) + vcs.get("scene", 0.0)))
    if sd_scene > 0:
        sd_pair = math.sqrt(2 * m.scale)
        exp = lambda x: stats.norm.cdf(x / sd_pair)
        rep.p(f"**AESTHETIC − CONTROL on the scale of the generated scenes**: {a.est / sd_scene:+.2f} SD of the scene means "
              f"(95% CI {a.lo / sd_scene:+.2f} / {a.hi / sd_scene:+.2f}; SD between scenes = {sd_scene:.2f} points). "
              f"**Pairwise preference expected from the ratings**: {exp(a.est):.1%} for AESTHETIC (95% CI {exp(a.lo):.1%} / {exp(a.hi):.1%}; "
              "50% = none), to be compared with the Bradley–Terry estimate below.")
        rep.add("A1", "AESTHETIC−CONTROL in SD of the scene means", a.est / sd_scene, a.lo / sd_scene, a.hi / sd_scene, None, len(r))
        rep.add("A1", "pairwise preference expected from the ratings", exp(a.est), exp(a.lo), exp(a.hi), None, len(r))
    rep.add("A1", "LMM AESTHETIC−CONTROL (points)", a.est, a.lo, a.hi, a.p, len(r))
    rep.add("A1", "LMM BASE−CONTROL (points)", b.est, b.lo, b.hi, b.p, len(r))
    rep.add("A1", "LMM AESTHETIC−BASE (points)", ab.est, ab.lo, ab.hi, ab.p, len(r))
    rep.add("A4", "standardised d AESTHETIC−CONTROL", a.est / sd, a.lo / sd, a.hi / sd, None, len(r))

    # 2. ordinal
    X = pd.get_dummies(r[["condition_code"]], drop_first=False).astype(float).drop(columns=["condition_code_CONTROL"])
    Xp = pd.get_dummies(r.participant_id.astype(str), prefix="p", drop_first=True).astype(float)
    try:
        om = OrderedModel(r.score.values, pd.concat([X, Xp], axis=1).values, distr="logit").fit(
            method="bfgs", disp=False, maxiter=5000, cov_type="cluster", cov_kwds={"groups": r.participant_id.values})
        rows = []
        for i, c in enumerate(X.columns):
            e = Est(om.params[i], om.bse[i], om.pvalues[i])
            rows.append([c.replace("condition_code_", "") + " vs CONTROL", f"{math.exp(e.est):.2f}", f"{math.exp(e.lo):.2f}–{math.exp(e.hi):.2f}", fmt_p(e.p)])
            rep.add("A2", f"ordinal OR {c.replace('condition_code_', '')} vs CONTROL", math.exp(e.est), math.exp(e.lo), math.exp(e.hi), e.p, len(r))
        rep.p("**2. Ordinal model** (odds of a higher score; participant fixed effects; SE clustered by participant):")
        rep.table(["contrast", "odds ratio", "95% CI", "p"], rows)
    except Exception as ex:
        rep.p(f"**2. Ordinal model** could not be fitted ({ex.__class__.__name__}).")

    # 3. Bradley–Terry
    g = bt(q)
    rows = []
    for name, label2 in [("Intercept", "position: left chosen"), ("AESTHETIC", "**AESTHETIC vs CONTROL**"), ("BASE", "BASE vs CONTROL")]:
        e = bt_est(g, name)
        rows.append([label2, f"{math.exp(e.est):.2f}", f"{math.exp(e.lo):.2f}–{math.exp(e.hi):.2f}", fmt_p(e.p),
                     f"{1 / (1 + math.exp(-e.est)):.1%}"])
        rep.add("A3", f"Bradley–Terry OR {name}", math.exp(e.est), math.exp(e.lo), math.exp(e.hi), e.p, len(q))
    cv = g.cov_params()
    dd = g.params["AESTHETIC"] - g.params["BASE"]
    dse = math.sqrt(cv.loc["AESTHETIC", "AESTHETIC"] + cv.loc["BASE", "BASE"] - 2 * cv.loc["AESTHETIC", "BASE"])
    pab = 2 * (1 - stats.norm.cdf(abs(dd / dse)))
    rows.append(["AESTHETIC vs BASE", f"{math.exp(dd):.2f}", f"{math.exp(dd - 1.96 * dse):.2f}–{math.exp(dd + 1.96 * dse):.2f}", fmt_p(pab), f"{1 / (1 + math.exp(-dd)):.1%}"])
    rep.add("A3", "Bradley–Terry OR AESTHETIC vs BASE", math.exp(dd), math.exp(dd - 1.96 * dse), math.exp(dd + 1.96 * dse), pab, len(q))
    rep.p("**3. Bradley–Terry model** (logistic, SE clustered by participant; last column = probability of being chosen, position-neutral):")
    rep.table(["term", "odds ratio", "95% CI", "p", "P(chosen)"], rows)
    try:
        vb = BinomialBayesMixedGLM.from_formula("left_won ~ 1 + AESTHETIC + BASE", {"participant": "0 + C(participant_id)", "scene": "0 + C(scene)"}, q).fit_vb()
        rep.p("Mixed version (random participant and scene, variational Bayes): " + "; ".join(
            f"{n} OR {math.exp(mu):.2f} ({math.exp(mu - 1.96 * s):.2f}–{math.exp(mu + 1.96 * s):.2f})"
            for n, mu, s in zip(vb.model.exog_names, vb.fe_mean, vb.fe_sd)) + ".")
    except Exception:
        pass

    # 5. equivalence
    lo90, hi90 = a.est - 1.645 * a.se, a.est + 1.645 * a.se
    ea = bt_est(g, "AESTHETIC")
    olo, ohi = math.exp(ea.est - 1.645 * ea.se), math.exp(ea.est + 1.645 * ea.se)
    eq_r = lo90 > -MARGIN_POINTS and hi90 < MARGIN_POINTS
    eq_p = olo > MARGIN_OR[0] and ohi < MARGIN_OR[1]
    rep.p(f"**5. Equivalence (TOST)** — ratings: 90% CI {lo90:+.3f} / {hi90:+.3f} vs margin ±{MARGIN_POINTS} → "
          f"**{'equivalent' if eq_r else 'not shown'}**; pairwise: 90% CI OR {olo:.2f}–{ohi:.2f} vs {MARGIN_OR[0]}–{MARGIN_OR[1]} → **{'equivalent' if eq_p else 'not shown'}**.")
    rep.add("A5", "TOST ratings 90% CI", a.est, lo90, hi90, None, len(r), "equivalent" if eq_r else "not shown")
    rep.add("A5", "TOST pairwise 90% CI (OR)", math.exp(ea.est), olo, ohi, None, len(q), "equivalent" if eq_p else "not shown")

    # 6. Bayes (BIC approximation)
    sub = r[r.condition_code.isin(["AESTHETIC", "CONTROL"])].assign(one=1)
    vc0 = {"participant": "0 + C(participant_id)", "prompt": "0 + C(prompt_code)", "scene": "0 + C(scene)"}
    try:
        # same random structure with and without the condition term, maximum likelihood (not REML) so the fits are comparable
        m1 = smf.mixedlm("score ~ A", sub, groups="one", re_formula="0", vc_formula=vc0).fit(reml=False, method="lbfgs")
        m0 = smf.mixedlm("score ~ 1", sub, groups="one", re_formula="0", vc_formula=vc0).fit(reml=False, method="lbfgs")
        n = len(sub)
        bf01_r = math.exp((bic(m1.llf, len(m1.params), n) - bic(m0.llf, len(m0.params), n)) / 2)
    except Exception:
        bf01_r = float("nan")
    qa = q[q.left_condition.isin(["AESTHETIC", "CONTROL"]) & q.right_condition.isin(["AESTHETIC", "CONTROL"])]
    g1 = smf.glm("left_won ~ 1 + AESTHETIC", qa, family=sm.families.Binomial()).fit()
    g0 = smf.glm("left_won ~ 1", qa, family=sm.families.Binomial()).fit()
    bf01_p = math.exp((bic(g1.llf, 2, len(qa)) - bic(g0.llf, 1, len(qa))) / 2)
    pr = stats.norm.cdf(a.est / a.se)
    pp = stats.norm.cdf(ea.est / ea.se)
    rep.p(f"**6. Bayesian evidence (approximate, BIC / unit-information prior)** — BF01 (evidence for *no* difference) ratings **{bf01_r:.1f}**, "
          f"pairwise **{bf01_p:.1f}** (BF01 > 3 moderate, > 10 strong evidence for no effect; < 1/3 evidence for an effect). "
          f"P(AESTHETIC > CONTROL): ratings {pr:.0%}, pairwise {pp:.0%}.")
    rep.add("A6", "BF01 ratings (BIC)", bf01_r, n=len(sub))
    rep.add("A6", "BF01 pairwise (BIC)", bf01_p, n=len(qa))
    rep.add("A6", "P(AESTHETIC>CONTROL) ratings", pr)
    rep.add("A6", "P(AESTHETIC>CONTROL) pairwise", pp)
    return a, ea


def section_b(d: Data, rep: Report):
    rep.h("3. Personal transfer — the raters who built the consensus")
    ppl = d.people.set_index("participant_id")
    post_ids = set(d.r.participant_id) | set(d.p.participant_id)
    linked = {i for i in post_ids if i in ppl.index and bool(ppl.loc[i, "linked"])}
    declared = {i for i in post_ids if i in ppl.index and str(ppl.loc[i, "prior_participation_self_report"]) == "Yes"} - linked
    only2 = post_ids - linked - declared
    rep.p(f"Phase-2 participants: {len(post_ids)} · linked to phase 1: **{len(linked)}** · declared phase 1 (not linked): {len(declared)} · phase 2 only: {len(only2)}.")

    # 7. subgroup models + interaction
    rows = []
    for name, ids in [("linked to phase 1", linked), ("declared phase 1", declared), ("phase 2 only", only2)]:
        rr, qq = d.r[d.r.participant_id.isin(ids)], decisive(d.p[d.p.participant_id.isin(ids)])
        if rr.participant_id.nunique() < 5:
            rows.append([name, rr.participant_id.nunique(), "too few", "", "", ""])
            continue
        try:
            m, _ = lmm(rr, slopes=False)
            a = coef(m, "A")
            g = bt(qq) if qq.participant_id.nunique() >= 5 else None
            ga = bt_est(g, "AESTHETIC") if g is not None else None
            rows.append([name, rr.participant_id.nunique(), f"{a.est:+.3f} ({a.lo:+.2f}/{a.hi:+.2f})", fmt_p(a.p),
                         f"{math.exp(ga.est):.2f} ({math.exp(ga.lo):.2f}–{math.exp(ga.hi):.2f})" if ga else "–", fmt_p(ga.p) if ga else "–"])
            rep.add("B7", f"AESTHETIC−CONTROL points · {name}", a.est, a.lo, a.hi, a.p, len(rr))
        except Exception as ex:
            rows.append([name, rr.participant_id.nunique(), f"not estimable ({ex.__class__.__name__})", "", "", ""])
    rep.p("**7. Primary contrast by group** (ratings: LMM points; pairwise: Bradley–Terry OR):")
    rep.table(["group", "participants", "AESTHETIC−CONTROL (95% CI)", "p", "pairwise OR (95% CI)", "p"], rows)
    rr = d.r.copy()
    rr["G"] = rr.participant_id.isin(linked).astype(int)
    if rr.G.sum() and (1 - rr.G).sum():
        try:
            m, _ = lmm(rr, slopes=False, extra=" + G + A:G + B:G")
            e = coef(m, "A:G")
            rep.p(f"Interaction AESTHETIC × linked-to-phase-1: {e.est:+.3f} points (95% CI {e.lo:+.3f}/{e.hi:+.3f}), p = {fmt_p(e.p)}.")
            rep.add("B7", "interaction AESTHETIC × linked", e.est, e.lo, e.hi, e.p, len(rr))
        except Exception:
            pass

    # 8. individual alignment
    pre = d.pre[d.pre.participant_id.isin(linked)]
    rows8 = []
    for pid, g in pre.groupby("participant_id"):
        a1, c1 = g[g.image_code.isin(d.aes_codes)].score, g[g.image_code.isin(d.ctl_codes)].score
        if len(a1) < 3 or len(c1) < 3:
            continue
        g2 = d.r[d.r.participant_id == pid]
        a2, c2 = g2[g2.condition_code == "AESTHETIC"].score, g2[g2.condition_code == "CONTROL"].score
        qq = decisive(d.p[d.p.participant_id == pid])
        qa = qq[qq.left_condition.isin(["AESTHETIC", "CONTROL"]) & qq.right_condition.isin(["AESTHETIC", "CONTROL"])]
        wins = ((qa.left_condition == "AESTHETIC") & (qa.left_won == 1)) | ((qa.right_condition == "AESTHETIC") & (qa.left_won == 0))
        rows8.append(dict(pid=pid, pref1=a1.mean() - c1.mean(), pref2=(a2.mean() - c2.mean()) if len(a2) and len(c2) else np.nan,
                          win2=wins.mean() if len(qa) else np.nan))
    al = pd.DataFrame(rows8)
    rep.p("**8. Individual alignment** — does the participant's own phase-1 preference for the photographs selected for "
          "AESTHETIC (vs CONTROL) predict their phase-2 preference for AESTHETIC images?")
    if len(al) >= 8:
        for col, lab in [("pref2", "rating difference AESTHETIC−CONTROL"), ("win2", "share of AESTHETIC choices vs CONTROL")]:
            x = al.dropna(subset=[col])
            if len(x) < 8:
                continue
            rho, p = stats.spearmanr(x.pref1, x[col])
            boots = [stats.spearmanr(*x.sample(len(x), replace=True, random_state=int(RNG.integers(1e9)))[["pref1", col]].T.values)[0] for _ in range(1000)]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rep.p(f"- phase-1 preference vs phase-2 {lab}: Spearman ρ = **{rho:+.2f}** (95% CI {lo:+.2f}/{hi:+.2f}), p = {fmt_p(p)}, n = {len(x)}.")
            rep.add("B8", f"alignment ρ · {lab}", rho, lo, hi, p, len(x))
        rep.p(f"- mean phase-1 preference for the AESTHETIC photographs: {al.pref1.mean():+.2f} points "
              f"({(al.pref1 > 0).mean():.0%} of these participants preferred them).")
        rrl = d.r[d.r.participant_id.isin(al.pid)].merge(al[["pid", "pref1"]], left_on="participant_id", right_on="pid")
        rrl["P1"] = (rrl.pref1 - rrl.pref1.mean()) / (rrl.pref1.std() or 1)
        try:
            m, _ = lmm(rrl, slopes=False, extra=" + A:P1 + B:P1")
            e = coef(m, "A:P1")
            rep.p(f"- interaction AESTHETIC × phase-1 preference (per SD): {e.est:+.3f} points (95% CI {e.lo:+.3f}/{e.hi:+.3f}), p = {fmt_p(e.p)}.")
            rep.add("B8", "interaction AESTHETIC × phase-1 preference (per SD)", e.est, e.lo, e.hi, e.p, len(rrl))
        except Exception:
            pass
    else:
        rep.p(f"- not enough linked participants with ratings of both photograph sets ({len(al)}; at least 8 needed).")

    # 9. agreement with the consensus
    rep.p("**9. Agreement with the phase-1 consensus** (correlation of the participant's ratings with the leave-one-out image means):")
    allpre = d.pre
    s = allpre.groupby("image_code").score.agg(["sum", "count"])
    agr = []
    for pid, g in pre.groupby("participant_id"):
        g = g.join(s, on="image_code")
        g = g[g["count"] > 1]
        loo = (g["sum"] - g.score) / (g["count"] - 1)
        if len(g) >= 10 and g.score.std() > 0:
            agr.append(dict(pid=pid, agreement=np.corrcoef(g.score, loo)[0, 1]))
    ag = pd.DataFrame(agr)
    if len(ag) >= 8 and len(al):
        x = ag.merge(al, on="pid").dropna(subset=["pref2"])
        if len(x) >= 8:
            rho, p = stats.spearmanr(x.agreement, x.pref2)
            rep.p(f"- mean agreement r = {ag.agreement.mean():.2f}; agreement vs phase-2 AESTHETIC−CONTROL preference: ρ = **{rho:+.2f}**, p = {fmt_p(p)}, n = {len(x)}.")
            rep.add("B9", "consensus agreement vs phase-2 preference ρ", rho, None, None, p, len(x))
    else:
        rep.p("- not enough linked participants.")


def icc(df, item, score):
    g = df.groupby(item)[score]
    k = g.count()
    df = df[df[item].isin(k[k >= 2].index)]
    g = df.groupby(item)[score]
    n_i, grand = g.count(), df[score].mean()
    a, N = len(n_i), len(df)
    if a < 3:
        return float("nan"), float("nan"), float("nan"), 0
    msb = (n_i * (g.mean() - grand) ** 2).sum() / (a - 1)
    raters = df["participant_id"].nunique() if "participant_id" in df else 1
    msw = ((df[score] - g.transform("mean")) ** 2).sum() / (N - a - (raters - 1))   # scores are centred per participant
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    icc1 = (msb - msw) / (msb + (k0 - 1) * msw)
    iccm = (msb - msw) / msb
    # split-half
    sh = []
    for _ in range(100):
        dd = df.assign(u=RNG.random(len(df))).sort_values([item, "u"])
        dd["half"] = dd.groupby(item).cumcount() % 2
        m = dd.groupby([item, "half"])[score].mean().unstack().dropna()
        if len(m) > 3:
            rr = np.corrcoef(m[0], m[1])[0, 1]
            sh.append(2 * rr / (1 + rr))
    return icc1, iccm, float(np.nanmean(sh)) if sh else float("nan"), a


def section_c(d: Data, raw: Data, rep: Report):
    rep.h("4. Human evaluation")
    r, q = d.r, decisive(d.p)
    # 10. reliability
    pre = d.pre.copy()
    pre["c"] = pre.score - pre.groupby("participant_id").score.transform("mean")
    rows = []
    for lab, df, item in [("phase 2 (generated images)", r, "generated_image_id")]:
        i1, im, sh, n = icc(df, item, "c")
        rows.append([lab, n, f"{i1:.2f}", f"{im:.2f}", f"{sh:.2f}"])
        rep.add("C10", f"ICC(1) · {lab}", i1, n=n)
        rep.add("C10", f"split-half (Spearman–Brown) · {lab}", sh, n=n)
    rep.p("**10. Reliability of the phase-2 judgements** (participant-centred scores; images with ≥ 2 ratings):")
    rep.table(["phase", "images", "ICC(1) single rater", "ICC(k) image mean", "split-half"], rows)
    rep.p("ICC(1) = how much raters agree on a single image; low values mean that small differences between conditions need many ratings.")

    # 11. convergent validity
    im = r.groupby("generated_image_id").c.mean()
    win = pd.concat([
        pd.DataFrame({"img": q.left_generated_image_id, "won": q.left_won}),
        pd.DataFrame({"img": q.right_generated_image_id, "won": 1 - q.left_won})]).groupby("img").won.agg(["mean", "count"])
    win = win[win["count"] >= 3].join(im.rename("rating"), how="inner")
    if len(win) >= 10:
        rho, p = stats.spearmanr(win["mean"], win.rating)
        rep.p(f"**11. Convergent validity** — image mean rating vs share of pairwise wins: Spearman ρ = **{rho:+.2f}**, p = {fmt_p(p)}, {len(win)} images.")
        rep.add("C11", "rating vs pairwise wins ρ", rho, None, None, p, len(win))

    # 12. order and fatigue
    ses = d.sessions
    first = ses.sort_values("started_at").groupby("participant_id").mode.first()
    rr = r.copy()
    rr["RF"] = rr.participant_id.map(first).eq("PostTrainingRating").astype(int)
    rr["POS"] = rr.position / 60.0
    try:
        m, _ = lmm(rr, slopes=False, extra=" + RF + A:RF + POS + A:POS")
        e1, e2, e3 = coef(m, "A:RF"), coef(m, "POS"), coef(m, "A:POS")
        rep.p(f"**12. Order and fatigue** — AESTHETIC effect when ratings came first vs pairs first: {e1.est:+.3f} (p = {fmt_p(e1.p)}); "
              f"change of scores over the session (per 60 items): {e2.est:+.3f} (p = {fmt_p(e2.p)}); "
              f"AESTHETIC × position: {e3.est:+.3f} (p = {fmt_p(e3.p)}).")
        rep.add("C12", "AESTHETIC × ratings-first", e1.est, e1.lo, e1.hi, e1.p, len(rr))
        rep.add("C12", "position slope (per 60 items)", e2.est, e2.lo, e2.hi, e2.p, len(rr))
    except Exception:
        pass
    left = q.left_won.mean()
    rep.p(f"Position in pairwise choices: left (or top) chosen {left:.1%} of decisive choices (50% expected without bias).")

    # 13. sensitivity
    rows = []

    def quick(rr_, qq_):
        m, _ = lmm(rr_, slopes=False)
        a = coef(m, "A")
        g = bt(qq_)
        e = bt_est(g, "AESTHETIC")
        return f"{a.est:+.3f} ({a.lo:+.2f}/{a.hi:+.2f})", fmt_p(a.p), f"{math.exp(e.est):.2f} ({math.exp(e.lo):.2f}–{math.exp(e.hi):.2f})", fmt_p(e.p)

    speed = pd.concat([r[["participant_id", "response_time_ms"]], d.p[["participant_id", "response_time_ms"]]]).groupby("participant_id").response_time_ms.median()
    fastest = set(speed[speed <= speed.quantile(0.10)].index)
    done = ses[ses.status == "Completed"].groupby("participant_id").size()
    complete = set(done[done >= 2].index)
    variants = [
        ("as planned (exclusions applied)", r, d.p),
        ("no exclusions", raw.r, raw.p),
        ("without answers < 500 ms", r[r.response_time_ms >= 500], d.p[d.p.response_time_ms.fillna(0) >= 500]),
        ("without the 10% fastest participants", r[~r.participant_id.isin(fastest)], d.p[~d.p.participant_id.isin(fastest)]),
        ("completed participants only", r[r.participant_id.isin(complete)], d.p[d.p.participant_id.isin(complete)]),
    ]
    for lab, rr_, pp_ in variants:
        try:
            rows.append([lab, *quick(rr_, decisive(pp_))])
        except Exception as ex:
            rows.append([lab, f"not estimable ({ex.__class__.__name__})", "", "", ""])
    rep.p("**13. Sensitivity** — AESTHETIC − CONTROL under different data choices (LMM intercepts only; Bradley–Terry):")
    rep.table(["data", "AESTHETIC−CONTROL points (95% CI)", "p", "pairwise OR (95% CI)", "p"], rows)


def section_d(d: Data, rep: Report, metrics: str | None):
    rep.h("2b. What is transferred — building types and image characteristics")
    # 16. by category
    rc = d.r.merge(d.prompts[["prompt_code", "category"]], on="prompt_code", how="left")
    rows = []
    for cat, g in rc.groupby("category"):
        per = g.groupby(["participant_id", "condition_code"]).c.mean().unstack()
        if "AESTHETIC" not in per or "CONTROL" not in per:
            continue
        diff = (per["AESTHETIC"] - per["CONTROL"]).dropna()
        if len(diff) < 5:
            continue
        boots = [diff.sample(len(diff), replace=True, random_state=int(RNG.integers(1e9))).mean() for _ in range(1000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append([cat, len(diff), f"{diff.mean():+.2f}", f"{lo:+.2f} / {hi:+.2f}", "*" if lo > 0 or hi < 0 else ""])
        rep.add("D16", f"AESTHETIC−CONTROL · {cat}", diff.mean(), lo, hi, None, len(diff))
    rep.p("**16. AESTHETIC − CONTROL by building category** (participant-centred; bootstrap over participants; * = CI excludes 0):")
    rep.table(["category", "participants", "difference", "95% CI", ""], sorted(rows, key=lambda x: float(x[2])))

    # 17. image covariates
    if not metrics:
        rep.p("**17. Image covariates** — skipped (no `--metrics` folder).")
        return
    try:
        key = pd.read_csv(os.path.join(metrics, "triplet_key.csv"))
        im = pd.read_csv(os.path.join(metrics, "image_metrics.csv"))
        cm = pd.read_csv(os.path.join(metrics, "content_metrics.csv"))
    except FileNotFoundError:
        rep.p("**17. Image covariates** — skipped (metric files not found).")
        return
    feats = im.merge(cm[["triplet", "condition", "whole_building", "elegant", "iconic_design", "prompt_similarity"]], on=["triplet", "condition"])
    rr = d.r.assign(triplet=d.r.prompt_code + "_" + d.r.seed.astype(str)).merge(
        feats, left_on=["triplet", "condition_code"], right_on=["triplet", "condition"], how="inner")
    cols = ["saturation", "colorfulness", "brightness", "sharpness", "whole_building", "elegant", "prompt_similarity"]
    for c in cols:
        rr["z_" + c] = (rr[c] - rr[c].mean()) / (rr[c].std() or 1)
    try:
        m0, _ = lmm(rr, slopes=False)
        m1, _ = lmm(rr, slopes=False, extra=" + " + " + ".join("z_" + c for c in cols))
        a0, a1 = coef(m0, "A"), coef(m1, "A")
        rows = [[c, f"{coef(m1, 'z_' + c).est:+.3f}", fmt_p(coef(m1, 'z_' + c).p)] for c in cols]
        rep.p(f"**17. Image covariates** ({len(rr)} ratings with metrics) — AESTHETIC − CONTROL without covariates {a0.est:+.3f} (p = {fmt_p(a0.p)}); "
              f"at equal image characteristics {a1.est:+.3f} (95% CI {a1.lo:+.3f}/{a1.hi:+.3f}, p = {fmt_p(a1.p)}). Effect of each characteristic (per SD):")
        rep.table(["characteristic", "points per SD", "p"], rows)
        rep.add("D17", "AESTHETIC−CONTROL adjusted for image characteristics", a1.est, a1.lo, a1.hi, a1.p, len(rr))
    except Exception as ex:
        rep.p(f"**17. Image covariates** — model not estimable ({ex.__class__.__name__}).")


def section_consensus(d: Data, rep: Report):
    """Returns the aesthetic signal (leave-one-rater-out when available) used for the transfer ratio."""
    rep.h("1. Consensus (phase 1)")
    pre = d.pre.copy()
    signal = float("nan")
    if pre.empty:
        rep.p("No phase-1 ratings.")
        return signal
    pre["c"] = pre.score - pre.groupby("participant_id").score.transform("mean")
    i1, im, sh, n = icc(pre, "image_code", "c")
    rep.p(f"**Agreement between raters** ({n} photographs, participant-centred): ICC(1) single rater **{i1:.2f}**, "
          f"ICC(k) photograph mean **{im:.2f}**, split-half **{sh:.2f}**. ICC(1) says how much two raters agree on one photograph; "
          "ICC(k) how stable the consensus (the mean of all ratings) is.")
    rep.add("S1", "ICC(1) phase 1", i1, n=n)
    rep.add("S1", "ICC(k) phase 1", im, n=n)
    rep.add("S1", "split-half phase 1", sh, n=n)
    # each rater against the others (leave-one-out photograph means)
    tot = pre.groupby("image_code").score.agg(["sum", "count"])
    x = pre.join(tot, on="image_code")
    x = x[x["count"] > 1].assign(loo=lambda z: (z["sum"] - z.score) / (z["count"] - 1))
    rs = [g.score.corr(g.loo) for _, g in x.groupby("participant_id") if len(g) >= 10]
    rs = [v for v in rs if not math.isnan(v)]
    if len(rs) >= 3:
        m, se = float(np.mean(rs)), float(np.std(rs, ddof=1) / math.sqrt(len(rs)))
        rep.p(f"**Agreement of each rater with the others**: mean r = **{m:.2f}** (95% CI {m - 1.96 * se:.2f} / {m + 1.96 * se:.2f}), {len(rs)} raters.")
        rep.add("S1", "mean leave-one-out agreement r", m, m - 1.96 * se, m + 1.96 * se, None, len(rs))
    # the aesthetic signal put into the fine-tuning
    if d.aes_codes and d.ctl_codes:
        per = []
        for _, g in pre.groupby("participant_id"):
            a_ = g[g.image_code.isin(d.aes_codes)].score
            c_ = g[g.image_code.isin(d.ctl_codes)].score
            if len(a_) >= 3 and len(c_) >= 3:
                per.append(a_.mean() - c_.mean())
        if len(per) >= 3:
            t = stats.ttest_1samp(per, 0)
            m, se = float(np.mean(per)), float(np.std(per, ddof=1) / math.sqrt(len(per)))
            q = stats.t.ppf(0.975, len(per) - 1)
            rep.p(f"**Aesthetic signal** — photos chosen for AESTHETIC minus photos in CONTROL, per rater: **{m:+.2f} points** "
                  f"(95% CI {m - q * se:+.2f} / {m + q * se:+.2f}), p = {fmt_p(t.pvalue)}, {len(per)} raters. "
                  "This is the difference in taste the fine-tuning could transfer (large by construction: AESTHETIC was selected with these ratings).")
            rep.add("S1", "aesthetic signal AESTHETIC−CONTROL photos (points, per rater; same ratings as the selection)", m, m - q * se, m + q * se, t.pvalue, len(per))
            signal = m
        # the same difference for a rater who took no part in the choice: best photographs re-selected from the others' means
        loo = []
        for _, g in pre.groupby("participant_id"):
            own = g.groupby("image_code").score.first()
            mean = tot["sum"] / tot["count"]
            idx = own.index
            mean.loc[idx] = ((tot.loc[idx, "sum"] - own) / (tot.loc[idx, "count"] - 1)).where(tot.loc[idx, "count"] > 1)
            top = set(mean.dropna().sort_values(ascending=False, kind="stable").index[:len(d.aes_codes)])
            a_, c_ = own[own.index.isin(top)], own[own.index.isin(d.ctl_codes - top)]
            if len(a_) >= 3 and len(c_) >= 3:
                loo.append(a_.mean() - c_.mean())
        if len(loo) >= 3:
            m, se = float(np.mean(loo)), float(np.std(loo, ddof=1) / math.sqrt(len(loo)))
            q = stats.t.ppf(0.975, len(loo) - 1)
            rep.p(f"**Aesthetic signal with an independent rater** (leave-one-rater-out: the best {len(d.aes_codes)} photos are re-selected without "
                  f"the rater, then scored with the rater's own votes): **{m:+.2f} points** (95% CI {m - q * se:+.2f} / {m + q * se:+.2f}), "
                  f"{len(loo)} raters. This is the unbiased estimate; the value above is inflated by regression to the mean.")
            rep.add("S1", "aesthetic signal, leave-one-rater-out (points)", m, m - q * se, m + q * se, None, len(loo))
            signal = m
    return signal


def davidson(p: pd.DataFrame):
    """Davidson model: Bradley–Terry with "no preference" as a third outcome and a position term (CONTROL = reference).
    Weights: left exp(eta), right 1, tie exp(delta + eta/2). Maximum likelihood (BFGS), numerical Hessian and scores,
    cluster-robust covariance by participant (CR1), t inference on G - 1 degrees of freedom."""
    from scipy.optimize import minimize
    names = ["position", "AESTHETIC", "BASE", "tie"]
    X = np.column_stack([np.ones(len(p))] + [(p.left_condition == c).astype(float) - (p.right_condition == c).astype(float) for c in ["AESTHETIC", "BASE"]])
    y = p.selected_side.map({"Left": 1, "Right": 0, "Tie": 2}).values
    idx = np.arange(len(p))

    def ll_i(th):
        eta = X @ th[:-1]
        w = np.column_stack([np.zeros(len(p)), eta, th[-1] + eta / 2])
        return w[idx, y] - np.log(np.exp(w).sum(1))

    th = minimize(lambda t: -ll_i(t).sum(), np.zeros(4), method="BFGS", options={"gtol": 1e-10}).x
    m, eye = 4, np.eye(4)
    S = np.column_stack([(ll_i(th + 1e-5 * eye[j]) - ll_i(th - 1e-5 * eye[j])) / 2e-5 for j in range(m)])
    H = np.array([[-(ll_i(th + 1e-4 * (eye[j] + eye[k])).sum() - ll_i(th + 1e-4 * (eye[j] - eye[k])).sum()
                     - ll_i(th - 1e-4 * (eye[j] - eye[k])).sum() + ll_i(th - 1e-4 * (eye[j] + eye[k])).sum()) / 4e-8 for k in range(m)] for j in range(m)])
    Sg = pd.DataFrame(S).groupby(p.participant_id.values).sum().values
    G, n = len(Sg), len(p)
    Hi = np.linalg.inv(H)
    se = np.sqrt(np.diag(Hi @ (Sg.T @ Sg) @ Hi * G / (G - 1) * (n - 1) / (n - m)))
    q = stats.t.ppf(0.975, G - 1)
    return {nm: (th[i], th[i] - q * se[i], th[i] + q * se[i], 2 * stats.t.sf(abs(th[i] / se[i]), G - 1)) for i, nm in enumerate(names)}


def sample_size(dz: float) -> float:
    """Participants for a paired comparison, 80% power, two-sided α = .05 (normal approximation + z²/2)."""
    if not math.isfinite(dz) or abs(dz) < 0.02:
        return float("nan")
    z, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    return math.ceil(((z + zb) / abs(dz)) ** 2 + z * z / 2)


def section_future(d: Data, rep: Report, a_primary: Est, signal: float = float("nan")):
    r, q = d.r, decisive(d.p)
    if math.isfinite(signal) and signal > 0.05:
        rep.h("2c. Transfer ratio and \"no preference\" answers")
        rep.p(f"**Transfer ratio** — AESTHETIC − CONTROL in phase 2 ({a_primary.est:+.3f} points, mixed model) divided by the aesthetic signal of phase 1 "
              f"({signal:+.2f} points): **{a_primary.est / signal:.0%}** (95% CI {a_primary.lo / signal:.0%} / {a_primary.hi / signal:.0%}; "
              "0% = nothing transferred, 100% = the whole difference between the training photos is found between the generated images).")
        rep.add("S2", "transfer ratio (phase-2 difference / phase-1 signal)", a_primary.est / signal, a_primary.lo / signal, a_primary.hi / signal)
    pa = d.p[d.p.left_condition.isin(["AESTHETIC", "CONTROL"]) & d.p.right_condition.isin(["AESTHETIC", "CONTROL"])]
    if len(pa):
        chosen = np.where(pa.selected_side == "Left", pa.left_condition, np.where(pa.selected_side == "Right", pa.right_condition, ""))
        half = pd.Series(np.where(chosen == "AESTHETIC", 1.0, np.where(chosen == "CONTROL", 0.0, 0.5)), index=pa.index).groupby(pa.participant_id).mean()
        if len(half) >= 3:
            t = stats.ttest_1samp(half, 0.5)
            se = float(half.std(ddof=1) / math.sqrt(len(half))); qq = stats.t.ppf(0.975, len(half) - 1)
            rep.p(f"**\"No preference\" kept as half a choice** — AESTHETIC share per participant: {half.mean():.1%} "
                  f"(95% CI {half.mean() - qq * se:.1%} / {half.mean() + qq * se:.1%}), p = {fmt_p(t.pvalue)}, {len(half)} participants.")
            rep.add("S2", "AESTHETIC share vs CONTROL, ties as half (per participant)", half.mean(), half.mean() - qq * se, half.mean() + qq * se, t.pvalue, len(half))
    if (d.p.selected_side == "Tie").any() and d.p.participant_id.nunique() >= 3:
        try:
            dv = davidson(d.p)
            tie = lambda x: math.exp(x) / (2 + math.exp(x))
            rows = [[lab, f"{math.exp(dv[k][0]):.2f}", f"{math.exp(dv[k][1]):.2f}–{math.exp(dv[k][2]):.2f}", fmt_p(dv[k][3])]
                    for k, lab in (("AESTHETIC", "**AESTHETIC vs CONTROL**"), ("BASE", "BASE vs CONTROL"), ("position", "position: left chosen"))]
            rep.p(f"**Davidson model** (Bradley–Terry with \"no preference\" as a third outcome; all {len(d.p)} answers; SE clustered by participant, t tests):")
            rep.table(["term", "odds ratio", "95% CI", "p"], rows)
            rep.p(f"Probability of \"no preference\" between two equally liked images: {tie(dv['tie'][0]):.1%} (95% CI {tie(dv['tie'][1]):.1%} / {tie(dv['tie'][2]):.1%}).")
            for k in ("AESTHETIC", "BASE", "position"):
                rep.add("S2", f"Davidson OR {k}", math.exp(dv[k][0]), math.exp(dv[k][1]), math.exp(dv[k][2]), dv[k][3], len(d.p))
            rep.add("S2", "Davidson probability of no preference at parity", tie(dv["tie"][0]), tie(dv["tie"][1]), tie(dv["tie"][2]), None, len(d.p))
        except Exception as ex:
            rep.p(f"**Davidson model** not estimable ({ex.__class__.__name__}).")
    rep.h("5. Towards a full study")
    n_part = d.r.participant_id.nunique()
    f_t = stats.t.ppf(0.975, n_part - 1) + stats.t.ppf(0.80, n_part - 1)   # 2.80 with many participants, larger with few
    mde = f_t * a_primary.se
    g = bt(q)
    mde_or = math.exp(f_t * float(g.bse["AESTHETIC"]))
    rep.p(f"**14. Minimum detectable effect** (80% power, α = .05) with the current data: **±{mde:.2f} points** on the 1–7 scale; "
          f"pairwise odds ratio **{mde_or:.2f}** (≈ {mde_or / (1 + mde_or):.0%} vs {1 / (1 + mde_or):.0%}). Smaller true effects would likely be missed.")
    rep.add("S5", "MDE points", mde)
    rep.add("S5", "MDE pairwise OR", mde_or)
    per = r[r.condition_code.isin(["AESTHETIC", "CONTROL"])].groupby(["participant_id", "condition_code"]).c.mean().unstack().dropna()
    diff = per["AESTHETIC"] - per["CONTROL"]
    sd = float(diff.std(ddof=1))
    qa = q[q.left_condition.isin(["AESTHETIC", "CONTROL"]) & q.right_condition.isin(["AESTHETIC", "CONTROL"])]
    share = qa.assign(a=np.where(qa.left_condition == "AESTHETIC", qa.left_won, 1 - qa.left_won)).groupby("participant_id").a.mean()
    n_margin, n_obs = sample_size(MARGIN_POINTS / sd), sample_size(diff.mean() / sd)
    n_pairs = sample_size((share.mean() - 0.5) / share.std(ddof=1)) if len(share) >= 3 else float("nan")
    rep.p(f"**15. Sample size for a full study** (paired comparison, 80% power, α = .05; SD of the per-participant AESTHETIC − CONTROL "
          f"differences in the pilot = {sd:.2f}): to detect **{MARGIN_POINTS} points** ≈ **{n_margin:.0f} participants**; to detect the observed "
          f"difference ({diff.mean():+.3f} points) ≈ {n_obs:.0f}; to detect the observed pairwise preference ({share.mean():.1%} AESTHETIC) ≈ {n_pairs:.0f}. "
          f"The pilot has {len(diff)} participants with both conditions rated. Orders of magnitude, assuming the pilot's variability.")
    rep.add("S5", f"participants needed for {MARGIN_POINTS} points", n_margin, n=len(diff))
    rep.add("S5", "participants needed for the observed rating difference", n_obs, n=len(diff))
    rep.add("S5", "participants needed for the observed pairwise preference", n_pairs, n=len(share))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--out", default="arch300_report")
    ap.add_argument("--metrics", default=None)
    args = ap.parse_args()
    folder = args.export
    if zipfile.is_zipfile(folder):
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(folder).extractall(tmp)
        folder = tmp
    os.makedirs(args.out, exist_ok=True)
    d, raw = load(folder, True), load(folder, False)
    rep = Report()
    rep.p("# ARCH300 — pilot study analysis")
    rep.p("*Investigating Aesthetic Transfer in Generative Models for Architectural Design: A Pilot Study on Consensus-Based "
          "Fine-Tuning and Human Evaluation* (ANALYSIS_PLAN.md). Estimates with 95% intervals; p values are uncorrected indications, "
          "not confirmatory tests.")
    signal = section_consensus(d, rep)
    a, _ = section_a(d, rep)
    section_d(d, rep, args.metrics)
    section_b(d, rep)
    section_c(d, raw, rep)
    section_future(d, rep, a, signal)

    open(os.path.join(args.out, "report_human_votes.md"), "w", encoding="utf-8").write("\n".join(rep.lines) + "\n")
    pd.DataFrame(rep.rows).to_csv(os.path.join(args.out, "results_human_votes.csv"), index=False)
    print("\n".join(rep.lines))


if __name__ == "__main__":
    main()
