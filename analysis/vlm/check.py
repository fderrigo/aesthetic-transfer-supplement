"""
Checks the three keys and model identifiers before any evaluation (ANALYSIS_PLAN.md, 7.1):

    python check.py

For each provider: the list of models available to the key, whether the configured identifier is in it, and one real
request with a neutral test image (a grey gradient generated here, not an image of the study) to verify that the image
input and the structured answer work. Keys are never printed.
"""
import io
import json

from PIL import Image

import providers as P


def test_image() -> bytes:
    im = Image.new("RGB", (1216, 832))
    px = im.load()
    for x in range(1216):
        for y in range(832):
            px[x, y] = (x * 255 // 1216, (x + y) * 255 // 2048, y * 255 // 832)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82)
    return b.getvalue()


def main():
    cfg = P.load_config()
    prompt = open(P.os.path.join(P.HERE, "rubric_v1.txt"), encoding="utf-8").read()
    img = test_image()
    for prov in P.PROVIDERS:
        model = cfg["providers"][prov]["model"]
        s, models = P.list_models(prov, cfg)
        print(f"\n== {prov}: key {'accepted' if s == 200 else 'REJECTED (HTTP %s)' % s}; configured model {model}: "
              f"{'listed' if model in models else 'NOT listed'}")
        if s != 200:
            print("  ", models[0][:300])
            continue
        if model not in models:
            print("   available:", ", ".join(sorted(models))[:1500])
        r = P.evaluate(prov, cfg, img, prompt, notes=False)
        print(f"   request: HTTP {r['http_status']}, {r['duration_s']} s, outcome {r['outcome']}, model returned {r['model_returned']}, stop {r['stop']}")
        print("   answer:", json.dumps(r["answer"]) if r["answer"] else json.dumps(r["raw"])[:900])
        print("   usage:", json.dumps(r["usage"])[:300])


if __name__ == "__main__":
    main()
