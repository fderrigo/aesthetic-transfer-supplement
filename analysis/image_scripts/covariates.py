"""
Image-derived covariates for the decomposition of the source-set direction (ANALYSIS_PLAN.md, section 9.1).

    python covariates.py <out .csv> <kind>=<folder> [<kind>=<folder> ...]

For every .jpg of the folders (the 480-px thumbnails used for the embeddings): the ten photographic measures of
image_metrics.py and the CLIP ViT-L/14 zero-shot attributes of content.py (probability of the first description against
the second). One row per image: kind, id (file name without extension), measures. Deterministic on CPU.
"""
import os
import sys

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image, ImageFilter

PHOTO = ["brightness", "contrast", "saturation", "colorfulness", "warmth", "sharpness", "detail", "sky_brightness", "dark_share", "bright_share"]
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
FRAMING = ["whole_building", "low_angle", "sunny", "vintage_photo", "real_photo", "people", "cars", "greenery", "water", "urban", "interior"]
CONTENT = ["iconic_design", "complex_form", "curved_organic", "monumental", "contemporary", "glass", "concrete", "wood", "white", "colourful"]


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
    return {"brightness": lum.mean(), "contrast": lum.std(), "saturation": hsv[..., 1].mean() / 2.55, "colorfulness": colorful, "warmth": (r - b).mean(),
            "sharpness": lap.var(), "detail": (edges > 40).mean() * 100, "sky_brightness": top.mean(), "dark_share": (lum < 40).mean() * 100, "bright_share": (lum > 235).mean() * 100}


def main():
    out = sys.argv[1]
    files = []
    for arg in sys.argv[2:]:
        kind, folder = arg.split("=", 1)
        files += [(kind, n[:-4], os.path.join(folder, n)) for n in sorted(os.listdir(folder)) if n.lower().endswith(".jpg")]
    seen, uniq = set(), []
    for k, i, p in files:                       # an image present in two folders is taken once
        if i not in seen:
            seen.add(i); uniq.append((k, i, p))
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    with torch.no_grad():
        T = {k: torch.nn.functional.normalize(model.encode_text(tok([a, b])), dim=-1) for k, (a, b) in PAIRS.items()}
    rows = []
    for n in range(0, len(uniq), 16):
        batch = uniq[n:n + 16]
        imgs = [Image.open(p).convert("RGB") for _, _, p in batch]
        with torch.no_grad():
            f = torch.nn.functional.normalize(model.encode_image(torch.stack([prep(im) for im in imgs])), dim=-1)
        attr = {k: torch.softmax((f @ t.T) * 100, dim=1)[:, 0].numpy() for k, t in T.items()}
        for j, ((kind, i, _), im) in enumerate(zip(batch, imgs)):
            rows.append({"kind": kind, "id": i, **measures(im), **{k: float(v[j]) for k, v in attr.items()}})
        if (n // 16) % 10 == 0:
            print(f"{min(n + 16, len(uniq))}/{len(uniq)}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
