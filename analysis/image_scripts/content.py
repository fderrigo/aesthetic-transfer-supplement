"""Content analysis of the generated images with CLIP (ViT-L/14, OpenAI weights): prompt adherence and zero-shot
attributes (architecture, context, materials, framing), compared within each triplet. Writes content_metrics.csv."""
import sys
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from scipy import stats

B = sys.argv[1]
key = pd.read_csv(f"{B}/analysis/triplet_key.csv").set_index("triplet")
trip = pd.read_csv(f"{B}/analysis/triplets.csv").set_index("triplet")
prompts = pd.read_csv(f"{B}/data/prompts.csv").set_index("prompt_code")
W, H, HEAD = 640, round(640 * 832 / 1216), 44

torch.set_num_threads(8)
model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-L-14")
model.eval()

def text_emb(texts):
    with torch.no_grad():
        e = model.encode_text(tok(texts))
    return e / e.norm(dim=-1, keepdim=True)

# ---- images
names, feats = [], []
for t, k in key.iterrows():
    sheet = Image.open(f"{B}/triplets_blind/{t}.jpg").convert("RGB")
    crops = [sheet.crop((4 + i * (W + 4), HEAD, 4 + i * (W + 4) + W, HEAD + H)) for i in range(3)]
    with torch.no_grad():
        f = model.encode_image(torch.stack([prep(c) for c in crops]))
    f = f / f.norm(dim=-1, keepdim=True)
    for i, panel in enumerate("ABC"):
        names.append((t, k[panel]))
        feats.append(f[i])
F = torch.stack(feats)
df = pd.DataFrame(names, columns=["triplet", "condition"])
df["prompt_code"] = df.triplet.str.split("_").str[0]

# ---- 1. prompt adherence: similarity to its own prompt (the subject part, before the photographic suffix)
subject = {c: prompts.loc[c, "text"].split(", whole building in view")[0] for c in prompts.index}
codes = sorted(subject)
P = text_emb([subject[c] for c in codes])
S = (F @ P.T) * 100
own = torch.tensor([codes.index(c) for c in df.prompt_code])
df["prompt_similarity"] = S[torch.arange(len(df)), own].numpy()
df["prompt_rank"] = (S > S[torch.arange(len(df)), own].unsqueeze(1)).sum(1).numpy() + 1   # 1 = own prompt is the best match among 48

# ---- 2. zero-shot attributes: probability of the first description vs the second (CLIP softmax, logit scale 100)
PAIRS = {
    "iconic_design":   ("a photo of an iconic, innovative, sculptural building by a famous architect", "a photo of an ordinary, generic, anonymous building"),
    "complex_form":    ("a building with a complex articulated form with many volumes and details", "a building with a simple plain box shape"),
    "curved_organic":  ("a building with curved, flowing, organic shapes", "a building with straight lines and right angles"),
    "monumental":      ("a monumental, imposing building", "a modest, small-scale building"),
    "elegant":         ("an elegant, refined, beautiful building", "an ugly, clumsy, awkward building"),
    "contemporary":    ("a contemporary building from the 21st century", "a building from the 20th century"),
    "whole_building":  ("a photo showing the whole building from a distance", "a close-up photo of part of a building facade"),
    "low_angle":       ("a dramatic low-angle photo of a building", "an eye-level photo of a building"),
    "people":          ("a photo of a building with people walking", "a photo of a building with no people"),
    "cars":            ("a photo of a building with cars parked in front", "a photo of a building with no cars"),
    "greenery":        ("a building surrounded by trees and lush vegetation", "a building surrounded by paved ground and asphalt"),
    "water":           ("a building next to water, a lake or the sea", "a building on dry land"),
    "urban":           ("a building in a dense city street", "a building in an open landscape"),
    "sunny":           ("a photo taken on a sunny day with blue sky", "a photo taken on an overcast grey day"),
    "vintage_photo":   ("an old vintage film photograph from the 1970s", "a modern digital photograph"),
    "real_photo":      ("a real photograph of a building", "a 3d render, computer generated image of a building"),
    "glass":           ("a building with a glass facade", "a building with an opaque facade"),
    "concrete":        ("a building made of exposed concrete", "a building not made of concrete"),
    "wood":            ("a building clad in wood", "a building not clad in wood"),
    "white":           ("a white building", "a building that is not white"),
    "colourful":       ("a colourful building with bright colours", "a building in neutral grey and beige tones"),
    "interior":        ("a photo of an interior space inside a building", "a photo of the exterior of a building"),
}
for name, (a, b) in PAIRS.items():
    T = text_emb([a, b])
    logits = (F @ T.T) * 100
    df[name] = torch.softmax(logits, dim=1)[:, 0].numpy()

df = df.merge(trip[["active"]], left_on="triplet", right_index=True)
df.to_csv(f"{B}/analysis/content_metrics.csv", index=False)

cols = ["prompt_similarity", "prompt_rank"] + list(PAIRS)
print("MEAN PER CONDITION (192 triplets; attributes = CLIP probability 0–1)")
print(df.groupby("condition")[cols].mean().reindex(["BASE", "CONTROL", "AESTHETIC"]).round(3).T.to_string())
print("\nprompt matched best among all 48 prompts (rank 1):", df.assign(top=df.prompt_rank == 1).groupby("condition").top.mean().round(3).to_dict())

Pv = df.pivot(index="triplet", columns="condition", values=cols)
print("\nWITHIN-TRIPLET DIFFERENCES: median, share of triplets where the first is higher, Wilcoxon p (* p < .005)")
for a, b in [("AESTHETIC", "CONTROL"), ("AESTHETIC", "BASE"), ("CONTROL", "BASE")]:
    print(f"\n  {a} − {b}")
    for c in cols:
        d = Pv[(c, a)] - Pv[(c, b)]
        p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1
        print(f"    {c:>17}: median {d.median():+7.3f}   {a} higher in {(d > 0).mean():4.0%}   p = {p:.4f}{'  *' if p < 0.005 else ''}")

# ---- 3. which content goes with higher ratings? within-triplet, participant-centred scores (active triplets)
r = pd.read_csv(f"{B}/data/posttraining_ratings.csv")
r = r[~r.excluded_from_analysis.astype(str).str.lower().eq("true")]
r["c"] = r.score - r.groupby("participant_id").score.transform("mean")
r["triplet"] = r.prompt_code + "_" + r.seed.astype(str)
sc = r.groupby(["triplet", "condition_code"]).c.mean().rename("score").reset_index().rename(columns={"condition_code": "condition"})
m = df.merge(sc, on=["triplet", "condition"])
m = m[m.active]
m["score_w"] = m.score - m.groupby("triplet").score.transform("mean")
print("\nCONTENT vs RATINGS (within triplet: does the image of the triplet with more of X get a higher score?) Spearman ρ, p")
for c in cols:
    x = m[c] - m.groupby("triplet")[c].transform("mean")
    rho, p = stats.spearmanr(x, m.score_w)
    print(f"    {c:>17}: ρ = {rho:+.3f}  p = {p:.3f}{'  *' if p < 0.005 else ''}")
