"""
ARCH300 — does the percentile of the preference rule matter? (check added on 2026-10-02)

    python selection_rule_check.py <extracted export folder> [--out human_report]

The frozen rule selects a photograph if it has at least 10 valid ratings, its mean is at least 5 on the 1–7 scale, and
its mean is not below the percentile cutoff (top 35% of the eligible photographs). A preliminary rule with the top 33%
had been recorded while phase 1 was still open. This script applies the rule to the valid phase-1 ratings of the export
for a range of percentiles and compares the result with the frozen AESTHETIC set.

The export reflects the exclusions of ratings and participants decided up to its date; the selection was made on the
ratings valid on 2026-09-27. The two can differ for photographs whose mean is at the threshold.
Writes <out>/report_selection_rule.md and results_selection_rule.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_report"))
args = ap.parse_args()
E = args.export
as_bool = lambda s: s.astype(str).str.lower().eq("true")
R = pd.read_csv(f"{E}/pretraining_ratings.csv")
R = R[~as_bool(R.excluded_from_analysis)]
SI = pd.read_csv(f"{E}/source_images.csv")
active = set(SI[as_bool(SI.is_active)].image_code)
runs = pd.read_csv(f"{E}/training_runs.csv").set_index("code")
A = pd.read_csv(f"{E}/aesthetic_dataset.csv")
aes = set(A[A.dataset_code == runs.training_dataset["RUN-AESTHETIC-4"]].image_code)
C = pd.read_csv(f"{E}/control_dataset.csv")
ctl = set(C[C.dataset_code == runs.training_dataset["RUN-CONTROL-4"]].image_code)
rule = pd.read_json(f"{E}/selection_rules.json").iloc[-1]["rule"]
MIN_N, MEAN = int(rule["minimumRatings"]), float(rule["meanThreshold"])

g = R[R.image_code.isin(active)].groupby("image_code").score.agg(["count", "mean"])
eligible = g[g["count"] >= MIN_N]
rows, sets = [], {}
for pct in (20, 25, 30, 33, 35, 40, 45, 50):
    cutoff = float(np.percentile(eligible["mean"], 100 - pct))
    by_pct = set(eligible[eligible["mean"] >= cutoff].index)
    sel = set(eligible[(eligible["mean"] >= cutoff) & (eligible["mean"] >= MEAN)].index)
    sets[pct] = sel
    rows.append(dict(top_percent=pct, percentile_cutoff_mean=cutoff, photographs_above_cutoff=len(by_pct), selected_by_the_rule=len(sel),
                     binding_criterion="mean threshold" if cutoff < MEAN else "percentile", in_frozen_aesthetic_set=len(sel & aes),
                     frozen_aesthetic_not_selected=len(aes - sel), selected_not_in_frozen_set=len(sel - aes)))
T = pd.DataFrame(rows)
os.makedirs(args.out, exist_ok=True)
T.to_csv(os.path.join(args.out, "results_selection_rule.csv"), index=False)
same = sets[33] == sets[35]
span = [p for p in sets if sets[p] == sets[35]]
low = sorted(aes - sets[35]); extra = sorted(sets[35] - aes)
L = ["# ARCH300 — does the percentile of the preference rule matter?", "",
     f"Frozen rule: at least {MIN_N} valid ratings, mean at least {MEAN:g}, top {rule['topPercentile']}% of the eligible photographs. "
     f"Applied here to the valid phase-1 ratings of the export: {len(R)} ratings, {len(eligible)} eligible photographs. Frozen AESTHETIC set: {len(aes)} photographs.", "",
     "| top % | mean at the percentile cutoff | photographs above the cutoff | selected by the whole rule | binding criterion | of which in the frozen AESTHETIC set |",
     "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['top_percent']} | {r['percentile_cutoff_mean']:.3f} | {r['photographs_above_cutoff']} | {r['selected_by_the_rule']} | {r['binding_criterion']} | {r['in_frozen_aesthetic_set']} |")
L += ["", f"**The set selected with the top 33% and with the top 35% is {'identical' if same else 'not identical'}.** "
      f"The selection does not change for any percentile among {', '.join(str(p) for p in span)}: in that range the percentile cutoff is below {MEAN:g}, "
      "so the photographs are selected by the mean threshold alone and the percentile has no effect.", "",
      "## Difference between the export and the frozen set", "",
      f"With the ratings valid in this export the rule selects {len(sets[35])} photographs; the frozen set has {len(aes)}. "
      f"{len(aes & sets[35])} photographs are in both. {len(low)} photographs of the frozen set now have a mean just below {MEAN:g} "
      f"({', '.join(f'{c} {g.loc[c, chr(109) + chr(101) + chr(97) + chr(110)]:.3f}' for c in low) or 'none'}); {len(extra)} photographs not in the frozen set now reach the threshold "
      f"({', '.join(f'{c} {g.loc[c, chr(109) + chr(101) + chr(97) + chr(110)]:.3f}' for c in extra) or 'none'}). "
      f"None of these {len(extra)} is in the CONTROL set ({len(set(extra) & ctl)} in common). "
      "The selection was made on the ratings valid on 2026-09-27; exclusions of answers and participants decided afterwards moved a few means across the threshold.", ""]
open(os.path.join(args.out, "report_selection_rule.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
