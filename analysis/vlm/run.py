"""
Runs the VLM evaluation on a blind bundle (ANALYSIS_PLAN.md, 7.3):

    python run.py pilot|full|replication [--providers openai,anthropic,google] [--limit N] [--workers 3]
    python run.py pilot|full|replication --collect          only rebuilds the long table from the raw outputs

One image per request, each request new and independent. The runner reads only manifests/<stage>_manifest_blind.csv and
the anonymous files: it never opens the key. Every attempt is appended to outputs/<stage>/<provider>.jsonl with the raw
response, so the run can be interrupted and restarted (evaluations already valid are not repeated). An invalid or refused
answer is retried `max_retries` times, then left missing. A provider that has no credit left, or that fails eight times
in a row, is stopped; the others continue.
In the pilot two short notes are asked in addition to the nine scores (7.5).
"""
import argparse
import datetime as dt
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import providers as P

PILOT_NOTES = """

NOTES
architectural_note: one short sentence stating the main visible reason for your ARCHITECTURAL APPEARANCE scores.
representation_note: one short sentence stating the main visible reason for your IMAGE / REPRESENTATION QUALITY scores."""


def read_attempts(file: str) -> list[dict]:
    if not os.path.exists(file):
        return []
    with open(file, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_provider(prov: str, cfg: dict, stage: str, manifest: pd.DataFrame, prompt: str, out_dir: str, workers: int):
    notes = stage == "pilot"
    file = os.path.join(out_dir, f"{prov}.jsonl")
    done = {a["anon_id"] for a in read_attempts(file) if a["outcome"] == "ok"}
    todo = [a for a in manifest.sort_values("position").anon_id if a not in done]
    bundle = os.path.join(P.path(cfg, "blind_bundle"), stage)
    lock, state = threading.Lock(), dict(stop=None, failures=0, next_start=0.0, ok=0)
    interval = 60.0 / cfg["run"]["requests_per_minute"]
    model = cfg["providers"][prov]["model"]
    # the request as sent, without the image and without the key
    url, body = P.request_body(prov, model, b"", prompt, notes)
    with open(os.path.join(out_dir, f"request_{prov}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(url=url, body=body), f, indent=1, ensure_ascii=False)

    def one(anon_id: str):
        image = open(os.path.join(bundle, anon_id + ".jpg"), "rb").read()
        for attempt in range(1, cfg["run"]["max_retries"] + 2):
            with lock:
                if state["stop"]:
                    return
                wait = state["next_start"] - time.time()
                state["next_start"] = max(time.time(), state["next_start"]) + interval
            if wait > 0:
                time.sleep(wait)
            stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
            r = P.evaluate(prov, cfg, image, prompt, notes)
            line = dict(anon_id=anon_id, model_provider=prov, model_id=model, rubric_version=cfg["run"]["rubric_version"], run=stage,
                        attempt=attempt, timestamp=stamp, **r)
            with lock:
                with open(file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                if r["outcome"] == "ok":
                    state["failures"] = 0; state["ok"] += 1
                    if state["ok"] % 25 == 0:
                        print(f"  {prov}: {len(done) + state['ok']}/{len(manifest)}", flush=True)
                    return
                if r["outcome"] == "no_credit":
                    state["stop"] = "no credit left"
                    return
                if r["outcome"] == "http_error":
                    state["failures"] += 1
                    if state["failures"] >= 8:
                        state["stop"] = "eight failed requests in a row"
                    return                                   # not an answer of the model: left for the next start

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"{prov}: {len(done) + state['ok']}/{len(manifest)} valid" + (f" — STOPPED: {state['stop']}" if state["stop"] else ""), flush=True)


def collect(stage: str, out_dir: str) -> pd.DataFrame:
    rows, notes = [], []
    for prov in P.PROVIDERS:
        seen = set()
        for a in read_attempts(os.path.join(out_dir, f"{prov}.jsonl")):
            if a["outcome"] != "ok" or a["anon_id"] in seen:
                continue
            seen.add(a["anon_id"])
            rows.append(dict(anon_image_id=a["anon_id"], model_provider=prov, model_id=a["model_returned"] or a["model_id"], rubric_version=a["rubric_version"],
                             run=a["run"], **{k: a["answer"][k] for k in P.SCORES}, timestamp=a["timestamp"]))
            if stage == "pilot":
                notes.append(dict(anon_image_id=a["anon_id"], model_provider=prov, **{k: a["answer"].get(k, "") for k in P.NOTES}))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "vlm_scores_long.csv"), index=False)
    if notes:
        pd.DataFrame(notes).to_csv(os.path.join(out_dir, "pilot_notes.csv"), index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "full", "replication"])
    ap.add_argument("--providers", default=",".join(P.PROVIDERS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    cfg = P.load_config()
    out_dir = os.path.join(P.path(cfg, "raw_outputs"), args.stage)
    os.makedirs(out_dir, exist_ok=True)
    if not args.collect:
        manifest = pd.read_csv(os.path.join(P.HERE, "manifests", f"{args.stage}_manifest_blind.csv"))
        if args.limit:
            manifest = manifest[manifest.position <= args.limit]
        prompt = open(os.path.join(P.HERE, "rubric_v1.txt"), encoding="utf-8").read().rstrip("\n") + (PILOT_NOTES if args.stage == "pilot" else "")
        threads = [threading.Thread(target=run_provider, args=(p, cfg, args.stage, manifest, prompt, out_dir, args.workers)) for p in args.providers.split(",")]
        [t.start() for t in threads]
        [t.join() for t in threads]
    t = collect(args.stage, out_dir)
    print(f"long table: {len(t)} rows; by model:", t.groupby("model_provider").size().to_dict() if len(t) else {})


if __name__ == "__main__":
    main()
