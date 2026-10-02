"""
DINOv2 embeddings of the ARCH300 images: the 600 corpus photographs (phase 1) and the generated images (phase 2).

    python embeddings.py <images folder> <out .npz> [--model facebook/dinov2-base]

<images folder> holds source/<image_code>.jpg, generated/<opaque_id>.jpg and manifest.csv (kind,image_code,opaque_id),
i.e. the 480-px thumbnails served by the platform. Every image is seen whole (no crop): it is resized so that the short
side is 224 px and both sides are multiples of the 14-px patch, then normalised with the ImageNet statistics.
Saved per image: the CLS token after the final layer norm (`cls`) and the mean of the patch tokens (`patch_mean`),
both float32 and not normalised. Deterministic on CPU (no augmentation, eval mode).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

PATCH, SHORT = 14, 224
MEAN, STD = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load(path: str) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = SHORT / min(w, h)
    nw, nh = max(PATCH, round(w * s / PATCH) * PATCH), max(PATCH, round(h * s / PATCH) * PATCH)
    x = np.asarray(im.resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--model", default="facebook/dinov2-base")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    model = AutoModel.from_pretrained(args.model).eval()
    revision = getattr(model.config, "_commit_hash", None) or ""
    rows = [l.strip().split(",") for l in open(os.path.join(args.images, "manifest.csv"), encoding="utf-8").read().splitlines()[1:] if l.strip()]
    kinds, ids, cls, pm, sizes = [], [], [], [], []
    with torch.no_grad():
        for n, (kind, code, opaque) in enumerate(rows):
            name = code if kind == "source" else opaque
            x = load(os.path.join(args.images, kind, name + ".jpg"))
            out = model(pixel_values=x.unsqueeze(0)).last_hidden_state[0]      # [1 + patches, dim], after the final layer norm
            kinds.append(kind); ids.append(name); sizes.append(list(x.shape[1:]))
            cls.append(out[0].numpy().astype(np.float32)); pm.append(out[1:].mean(0).numpy().astype(np.float32))
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(rows)}", flush=True)
    np.savez_compressed(args.out, kind=np.array(kinds), id=np.array(ids), cls=np.stack(cls), patch_mean=np.stack(pm), size=np.array(sizes),
                        model=np.array(args.model), revision=np.array(revision), preprocessing=np.array(f"whole image, short side {SHORT}px, multiples of {PATCH}px, bicubic, ImageNet mean/std"))
    print(f"saved {len(ids)} embeddings of dimension {cls[0].shape[0]} to {args.out} (model {args.model} {revision})")


if __name__ == "__main__":
    main()
