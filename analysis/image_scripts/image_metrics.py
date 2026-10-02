"""Objective image measures per condition, compared within each triplet (same prompt and seed).
Reads the blind triplet sheets (A|B|C, 640x438 each) and the key; writes image_metrics.csv and prints a summary."""
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import stats

B = sys.argv[1]
key = pd.read_csv(f"{B}/analysis/triplet_key.csv").set_index("triplet")
trip = pd.read_csv(f"{B}/analysis/triplets.csv").set_index("triplet")
W, H, HEAD = 640, round(640 * 832 / 1216), 44

def measures(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)   # Hasler & Süsstrunk
    grey = img.convert("L")
    lap = np.asarray(grey.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128)), dtype=np.float64) - 128
    edges = np.asarray(grey.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    top = lum[: lum.shape[0] // 3]
    return {
        "brightness": lum.mean(),                      # 0–255
        "contrast": lum.std(),                         # RMS contrast
        "saturation": hsv[..., 1].mean() / 2.55,       # 0–100
        "colorfulness": colorful,
        "warmth": (r - b).mean(),                      # >0 warmer (red/yellow), <0 cooler (blue)
        "sharpness": lap.var(),                        # variance of Laplacian
        "detail": (edges > 40).mean() * 100,           # % of edge pixels
        "sky_brightness": top.mean(),                  # upper third
        "dark_share": (lum < 40).mean() * 100,         # % near-black
        "bright_share": (lum > 235).mean() * 100,      # % near-white (clipping)
    }

rows = []
for t, k in key.iterrows():
    sheet = Image.open(f"{B}/triplets_blind/{t}.jpg")
    for i, panel in enumerate("ABC"):
        x = 4 + i * (W + 4)
        crop = sheet.crop((x, HEAD, x + W, HEAD + H))
        m = measures(crop)
        m.update(triplet=t, condition=k[panel], active=bool(trip.loc[t, "active"]))
        rows.append(m)
M = pd.DataFrame(rows)
M.to_csv(f"{B}/analysis/image_metrics.csv", index=False)

cols = ["brightness", "contrast", "saturation", "colorfulness", "warmth", "sharpness", "detail", "sky_brightness", "dark_share", "bright_share"]
print("MEAN PER CONDITION (all 192 triplets)")
print(M.groupby("condition")[cols].mean().reindex(["BASE", "CONTROL", "AESTHETIC"]).round(2).T.to_string())
print("\nWITHIN-TRIPLET DIFFERENCES (paired over 192 triplets): median difference, share of triplets where the first is higher, Wilcoxon p")
P = M.pivot(index="triplet", columns="condition", values=cols)
for a, b in [("AESTHETIC", "CONTROL"), ("AESTHETIC", "BASE"), ("CONTROL", "BASE")]:
    print(f"\n  {a} − {b}")
    for c in cols:
        d = P[(c, a)] - P[(c, b)]
        p = stats.wilcoxon(d).pvalue
        print(f"    {c:>15}: median {d.median():+8.2f}   {a} higher in {(d > 0).mean():5.0%}   p = {p:.4f}{'  *' if p < 0.005 else ''}")
# how different are the two LoRA images from BASE (pixel distance on 64x44 thumbnails)
print("\nHOW MUCH EACH LoRA CHANGES THE BASE IMAGE (mean absolute pixel difference on the same triplet, 0–255)")
dist = {"CONTROL": [], "AESTHETIC": [], "CONTROL_vs_AESTHETIC": []}
for t, k in key.iterrows():
    sheet = Image.open(f"{B}/triplets_blind/{t}.jpg")
    arr = {}
    for i, panel in enumerate("ABC"):
        x = 4 + i * (W + 4)
        arr[k[panel]] = np.asarray(sheet.crop((x, HEAD, x + W, HEAD + H)).resize((64, 44)).convert("RGB"), dtype=np.float64)
    dist["CONTROL"].append(np.abs(arr["CONTROL"] - arr["BASE"]).mean())
    dist["AESTHETIC"].append(np.abs(arr["AESTHETIC"] - arr["BASE"]).mean())
    dist["CONTROL_vs_AESTHETIC"].append(np.abs(arr["CONTROL"] - arr["AESTHETIC"]).mean())
for k, v in dist.items():
    v = np.array(v)
    print(f"  {k:>22}: median {np.median(v):5.1f}   (25–75%: {np.percentile(v,25):.1f}–{np.percentile(v,75):.1f})   share < 20: {(v < 20).mean():.0%}")
