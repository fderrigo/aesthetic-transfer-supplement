#!/usr/bin/env python3
"""
Aesthetic Architecture Research - cloud GPU worker.

Runs on a rented GPU instance created by the platform. It downloads its job specification from the
platform, executes it and sends the results back:

  training   : LoRA fine-tuning of a Stable Diffusion 1.x/2.x or SDXL base model on a frozen dataset
               (images/ + captions/ of the platform's training ZIP); uploads the .safetensors weights.
  generation : generates the missing images of a frozen generation plan (prompt x seed x condition)
               with the base model and, per condition, the LoRA weights; uploads the images with a manifest.

Environment: AAR_API (platform base URL), AAR_JOB (job id), AAR_TOKEN (per-job bearer token), HF_TOKEN (optional).
The script is served by the platform and its SHA-256 is recorded with the job, so every result can be traced
to the exact code that produced it.

Reproducibility: every random source is seeded from the run / plan seeds; library versions and the GPU model
are reported to the platform. Bit-identical results across different GPU models are not guaranteed by PyTorch.
"""
import math
import hashlib
import io
import json
import os
import platform
import random
import sys
import threading
import time
import traceback
import zipfile

import requests

API = os.environ["AAR_API"].rstrip("/")
JOB = os.environ["AAR_JOB"]
TOKEN = os.environ["AAR_TOKEN"]
HF_TOKEN = os.environ.get("HF_TOKEN") or None
BASE = f"{API}/api/worker/jobs/{JOB}"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
WORK = os.environ.get("AAR_WORK", "/opt/aar/work")


class Fatal(Exception):
    """Error that must not be retried (rejected request, invalid specification, cancelled job)."""


def call(method, path, *, json_body=None, files=None, data=None, headers=None, stream=False, timeout=300, retries=6):
    last = None
    for attempt in range(retries):
        try:
            h = dict(HEADERS)
            if headers:
                h.update(headers)
            body = data() if callable(data) else data
            r = requests.request(method, BASE + path, headers=h, json=json_body, files=files, data=body, stream=stream, timeout=timeout)
            if r.status_code in (400, 401, 403, 404, 409, 413, 422):
                raise Fatal(f"{method} {path}: HTTP {r.status_code} {r.text[:500]}")
            r.raise_for_status()
            return r
        except Fatal:
            raise
        except Exception as e:  # network errors and 5xx are retried
            last = e
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"{method} {path} failed after {retries} attempts: {last}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Reporter:
    """Keeps the platform informed (heartbeat every 60 s) and notices cancellation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.message = "worker started"
        self.progress = None
        self.total = None
        self.lines = []
        self.cancelled = False
        self._stop = threading.Event()

    def update(self, message=None, progress=None, total=None, log=True):
        with self.lock:
            if message is not None:
                self.message = message
                if log:
                    self.lines.append(time.strftime("%H:%M:%S ") + message)
                    self.lines = self.lines[-200:]
                    print(message, flush=True)
            if progress is not None:
                self.progress = progress
            if total is not None:
                self.total = total

    def send(self, environment=None):
        with self.lock:
            body = {"message": self.message, "progress": self.progress, "total": self.total, "log": "\n".join(self.lines[-80:])}
        if environment is not None:
            body["environment"] = environment
        try:
            call("POST", "/status", json_body=body, timeout=60, retries=2)
        except Fatal:
            self.cancelled = True
        except Exception as e:
            print(f"heartbeat failed: {e}", flush=True)

    def loop(self):
        while not self._stop.wait(60):
            self.send()

    def stop(self):
        self._stop.set()

    def check(self):
        if self.cancelled:
            raise Fatal("job cancelled or no longer accepted by the platform")

    def tail(self):
        with self.lock:
            return "\n".join(self.lines[-80:])


def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "worker_sha256": sha256_file(os.path.abspath(__file__))}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        info["torch_error"] = str(e)
    for mod in ("diffusers", "transformers", "peft", "accelerate", "safetensors"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def resolve_commit(model_id, revision):
    """Exact commit of the base model on Hugging Face (the requested revision, or the current head of the default branch).
    The model files are then downloaded from Hugging Face at that commit into HF_HOME on the instance."""
    from huggingface_hub import model_info
    try:
        return model_info(model_id, revision=revision or None, token=HF_TOKEN).sha
    except Exception as e:
        raise Fatal(f"base model '{model_id}' revision '{revision or 'main'}' not available on Hugging Face: {e}")


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download(path, target, rep, label):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    r = call("GET", path, stream=True, timeout=600)
    done = 0
    with open(target, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            done += len(chunk)
    rep.update(f"{label} downloaded ({done / 1e6:.1f} MB)")
    return target


# ------------------------------------------------------------------ training

def load_dataset(zip_path):
    root = os.path.join(WORK, "dataset")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    images_dir, captions_dir = os.path.join(root, "images"), os.path.join(root, "captions")
    pairs = []
    for name in sorted(os.listdir(images_dir)):
        code = os.path.splitext(name)[0]
        cap_path = os.path.join(captions_dir, code + ".txt")
        caption = open(cap_path, encoding="utf-8").read().strip() if os.path.exists(cap_path) else ""
        pairs.append((code, os.path.join(images_dir, name), caption))
    if not pairs:
        raise Fatal("the dataset contains no images")
    return pairs


def train(spec, rep):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from diffusers import DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    run = spec["run"]
    cfg = run.get("configuration") or {}
    seed = int(run["seed"])
    steps = int(run["steps"])
    res = int(run["resolution"])
    bs = int(run.get("batch_size") or 1)
    lr = float(run["learning_rate"])
    rank = int(run["lora_rank"])
    alpha = int(run.get("lora_alpha") or rank)
    dtype = torch.bfloat16
    seed_everything(seed)

    zip_path = download("/dataset", os.path.join(WORK, "dataset.zip"), rep, "dataset")
    pairs = load_dataset(zip_path)
    rep.update(f"dataset: {len(pairs)} images; loading base model {run['base_model']}")

    commit = resolve_commit(run["base_model"], run.get("base_model_revision"))
    rep.update(f"base model {run['base_model']} at commit {commit}")
    rep.send(environment={**environment_info(), "base_model_commits": {run["base_model"]: commit}})
    pipe = DiffusionPipeline.from_pretrained(run["base_model"], revision=commit, torch_dtype=dtype, token=HF_TOKEN)
    is_xl = isinstance(pipe, StableDiffusionXLPipeline)
    if not (is_xl or isinstance(pipe, StableDiffusionPipeline)):
        raise Fatal(f"training supports Stable Diffusion 1.x/2.x and SDXL base models, not {type(pipe).__name__}")
    pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)  # the SDXL VAE is unstable in half precision

    # Pre-compute latents and text conditioning once (the dataset is small); the VAE and text encoders are then unloaded.
    rep.update("encoding images and captions", 0, len(pairs))
    gen = torch.Generator("cuda").manual_seed(seed)
    samples = []
    with torch.no_grad():
        for i, (code, path, caption) in enumerate(pairs):
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            ow, oh = img.size
            scale = res / min(ow, oh)
            nw, nh = max(res, round(ow * scale)), max(res, round(oh * scale))
            img = img.resize((nw, nh), Image.BICUBIC)
            left, top = (nw - res) // 2, (nh - res) // 2
            img = img.crop((left, top, left + res, top + res))
            px = torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32)
            latent = pipe.vae.encode(px).latent_dist.sample(generator=gen) * pipe.vae.config.scaling_factor
            if is_xl:
                emb, _, pooled, _ = pipe.encode_prompt(prompt=caption, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
                time_ids = torch.tensor([[oh, ow, top, left, res, res]], dtype=torch.float32)
                samples.append({"latent": latent.cpu(), "emb": emb.cpu(), "pooled": pooled.cpu(), "time_ids": time_ids})
            else:
                emb, _ = pipe.encode_prompt(prompt=caption, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
                samples.append({"latent": latent.cpu(), "emb": emb.cpu()})
            rep.update(None, i + 1, len(pairs))
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    if is_xl:
        pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    unet = pipe.unet
    unet.requires_grad_(False)
    unet.to("cuda", dtype=dtype)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                                target_modules=cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"])))
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # trainable LoRA weights in fp32, frozen base in bf16
    if cfg.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=float(cfg.get("weight_decay", 1e-2)), eps=1e-8)
    lr_sched = get_scheduler(cfg.get("lr_scheduler", "constant"), optimizer=opt,
                             num_warmup_steps=int(cfg.get("lr_warmup_steps", 0)), num_training_steps=steps)
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    pred_type = noise_sched.config.prediction_type
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    snr_gamma = float(cfg["snr_gamma"]) if cfg.get("snr_gamma") else None
    # SDXL base was trained with offset noise (0.0357): fine-tuning without it drifts the LoRA towards flat, washed-out,
    # low-contrast images with a colour cast. Default on for SDXL, off for SD 1.x/2.x; "noise_offset": 0 disables it.
    noise_offset = float(cfg.get("noise_offset", 0.0357 if is_xl else 0.0))
    # Optional timestep range: training only the noisier steps (e.g. 300-1000) teaches form and composition while the
    # base model keeps its own rendering of colour, light and fine detail.
    num_t = noise_sched.config.num_train_timesteps
    t_min = max(0, int(cfg.get("timestep_min", 0)))
    t_max = min(num_t, int(cfg.get("timestep_max", num_t)))
    if t_min >= t_max:
        raise Fatal(f"invalid timestep range {t_min}-{t_max}")
    order_rng = random.Random(seed)
    order = []
    unet.train()
    rep.update(f"training: {steps} steps, batch {bs}, lr {lr}, rank {rank}, alpha {alpha}, {res}px, "
               f"scheduler {cfg.get('lr_scheduler', 'constant')}, warmup {cfg.get('lr_warmup_steps', 0)}, snr_gamma {snr_gamma}, "
               f"noise_offset {noise_offset}, timesteps {t_min}-{t_max}", 0, steps)
    losses = []
    for step in range(1, steps + 1):
        rep.check()
        batch = []
        while len(batch) < bs:
            if not order:
                order = list(range(len(samples)))
                order_rng.shuffle(order)
            batch.append(samples[order.pop()])
        latents = torch.cat([b["latent"] for b in batch]).to("cuda", torch.float32)
        emb = torch.cat([b["emb"] for b in batch]).to("cuda", dtype)
        noise = torch.randn_like(latents)
        if noise_offset:
            noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device="cuda")
        t = torch.randint(t_min, t_max, (latents.shape[0],), device="cuda").long()
        noisy = noise_sched.add_noise(latents, noise, t)
        kwargs = {}
        if is_xl:
            kwargs["added_cond_kwargs"] = {"text_embeds": torch.cat([b["pooled"] for b in batch]).to("cuda", dtype),
                                           "time_ids": torch.cat([b["time_ids"] for b in batch]).to("cuda", dtype)}
        with torch.autocast("cuda", dtype=dtype):
            pred = unet(noisy, t, encoder_hidden_states=emb, return_dict=False, **kwargs)[0]
        target = noise if pred_type == "epsilon" else noise_sched.get_velocity(latents, noise, t)
        if snr_gamma:
            # Min-SNR weighting (Hang et al. 2023): damps the easy low-noise timesteps, steadier training on small datasets
            ac = noise_sched.alphas_cumprod.to("cuda")[t]
            snr = ac / (1 - ac)
            w = torch.clamp(snr, max=snr_gamma) / (snr if pred_type == "epsilon" else snr + 1)
            loss = (F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3]) * w).mean()
        else:
            loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        lr_sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if not math.isfinite(losses[-1]):
            # diverged: the weights are garbage, never upload them as a completed run
            raise RuntimeError(f"training diverged: loss {losses[-1]} at step {step}/{steps} (learning rate {lr} too high?)")
        if step % 50 == 0 or step == steps:
            rep.update(f"step {step}/{steps} loss {sum(losses[-50:]) / len(losses[-50:]):.4f}", step, steps)
        else:
            rep.update(None, step, steps)

    out = os.path.join(WORK, "output")
    os.makedirs(out, exist_ok=True)
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
    cls.save_lora_weights(save_directory=out, unet_lora_layers=lora, safe_serialization=True)
    weights = os.path.join(out, "pytorch_lora_weights.safetensors")
    digest = sha256_file(weights)
    rep.update(f"uploading weights ({os.path.getsize(weights) / 1e6:.1f} MB, sha256 {digest[:12]}…)")

    def body():
        return open(weights, "rb")
    call("PUT", f"/artifact?file_name={run['code'].lower()}_lora.safetensors", data=body,
         headers={"Content-Type": "application/octet-stream", "X-Sha256": digest}, timeout=1800)
    rep.update(f"weights uploaded; final loss {sum(losses[-50:]) / len(losses[-50:]):.4f}")


# ------------------------------------------------------------------ generation

SAMPLERS = {
    "euler": ("EulerDiscreteScheduler", {}),
    "euler a": ("EulerAncestralDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "dpm++ 2m": ("DPMSolverMultistepScheduler", {}),
    "dpm++ 2m karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpm++ sde": ("DPMSolverSDEScheduler", {}),
    "dpm++ sde karras": ("DPMSolverSDEScheduler", {"use_karras_sigmas": True}),
}


def apply_sampler(pipe, name):
    import diffusers
    key = (name or "").strip().lower()
    if key in ("", "default"):
        return type(pipe.scheduler).__name__
    if "Flux" in type(pipe).__name__:
        raise Fatal(f"sampler '{name}' cannot be used with FLUX (leave the plan sampler empty)")
    if key not in SAMPLERS:
        raise Fatal(f"unknown sampler '{name}'; supported: {', '.join(sorted(SAMPLERS))}")
    cls_name, extra = SAMPLERS[key]
    pipe.scheduler = getattr(diffusers, cls_name).from_config(pipe.scheduler.config, **extra)
    return cls_name


def fetch_lora(cond, rep):
    lora = cond["lora"]
    target = os.path.join(WORK, "loras", f"{cond['code']}.safetensors")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if lora.get("platform"):
        download(f"/lora/{cond['code']}", target, rep, f"LoRA {cond['code']}")
    else:
        h = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN and "huggingface.co" in lora["url"] else {}
        with requests.get(lora["url"], headers=h, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    digest = sha256_file(target)
    if lora.get("sha256") and lora["sha256"].lower() != digest:
        raise Fatal(f"LoRA {cond['code']}: checksum mismatch")
    return target, digest


def generate(spec, rep):
    import inspect
    import torch
    from diffusers import AutoPipelineForText2Image

    plan = spec["plan"]
    cfg = plan.get("configuration") or {}
    conditions = spec["conditions"]
    items = spec["items"]
    total = len(items)
    rep.update(f"generation: {total} images, {len(conditions)} conditions", 0, total)
    done = 0
    pending = []

    def flush():
        nonlocal pending
        if not pending:
            return
        files = [("files", (name, content, "image/png")) for name, content, _ in pending]
        manifest = json.dumps([row for _, _, row in pending])
        res = call("POST", "/images", files=files, data={"manifest": manifest}, timeout=600).json()
        if res.get("invalid"):
            raise Fatal(f"platform rejected images: {json.dumps(res.get('lines'))[:800]}")
        pending = []

    bases = []  # (model id, revision)
    for c in conditions:
        key = (c["base_model"], c.get("base_model_revision"))
        if key not in bases:
            bases.append(key)
    commits = {}
    for base, revision in bases:
        conds = [c for c in conditions if (c["base_model"], c.get("base_model_revision")) == (base, revision)
                 and any(i["condition"] == c["code"] for i in items)]
        if not conds:
            continue
        flux = "flux" in (base + " " + " ".join(c.get("model_family") or "" for c in conds)).lower()
        dtype = torch.bfloat16 if flux else torch.float16
        commit = resolve_commit(base, revision)
        commits[base] = commit
        rep.update(f"loading base model {base}@{commit}")
        rep.send(environment={**environment_info(), "base_model_commits": commits})
        pipe = AutoPipelineForText2Image.from_pretrained(base, revision=commit, torch_dtype=dtype, token=HF_TOKEN).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        scheduler = apply_sampler(pipe, plan.get("sampler"))
        accepted = inspect.signature(pipe.__call__).parameters
        for c in conds:
            lora_sha = None
            if c.get("lora"):
                path, lora_sha = fetch_lora(c, rep)
                pipe.load_lora_weights(os.path.dirname(path), weight_name=os.path.basename(path), adapter_name="condition")
                if "lora_scale" in cfg:
                    pipe.set_adapters(["condition"], adapter_weights=[float(cfg["lora_scale"])])
            rep.update(f"condition {c['code']}")
            for it in [i for i in items if i["condition"] == c["code"]]:
                rep.check()
                kwargs = {
                    "prompt": it["prompt"],
                    "num_inference_steps": int(plan["steps"]),
                    "guidance_scale": float(plan["guidance_scale"]),
                    "width": int(plan["width"]),
                    "height": int(plan["height"]),
                    "generator": torch.Generator("cuda").manual_seed(int(it["seed"])),
                }
                if it.get("negative_prompt") and "negative_prompt" in accepted:
                    kwargs["negative_prompt"] = it["negative_prompt"]
                image = pipe(**kwargs).images[0]
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                params = {
                    "base_model": base, "base_model_commit": commit, "condition": c["code"], "lora_run": (c.get("lora") or {}).get("run_code"), "lora_sha256": lora_sha,
                    "pipeline": type(pipe).__name__, "scheduler": scheduler, "sampler": plan.get("sampler") or "",
                    "steps": int(plan["steps"]), "guidance_scale": float(plan["guidance_scale"]),
                    "width": int(plan["width"]), "height": int(plan["height"]), "seed": int(it["seed"]),
                    "prompt": it["prompt"], "negative_prompt": kwargs.get("negative_prompt"), "dtype": str(dtype).replace("torch.", ""),
                    "lora_scale": cfg.get("lora_scale"), "cloud_job": int(JOB),
                }
                pending.append((it["file_name"], buf.getvalue(), {
                    "file_name": it["file_name"], "condition_code": c["code"], "prompt_code": it["prompt_code"], "seed": int(it["seed"]), "parameters": params}))
                done += 1
                rep.update(f"{done}/{total} {it['file_name']}", done, total, log=(done % 10 == 0 or done == total))
                if len(pending) >= 8:
                    flush()
            flush()
            if c.get("lora"):
                pipe.unload_lora_weights()
        del pipe
        torch.cuda.empty_cache()
    flush()
    rep.update(f"generation completed: {done} images")


# ------------------------------------------------------------------ main

def main():
    os.makedirs(WORK, exist_ok=True)
    rep = Reporter()
    try:
        spec = call("GET", "/spec").json()
        rep.update(f"job {JOB}: {spec['kind']}")
        rep.send(environment=environment_info())
        threading.Thread(target=rep.loop, daemon=True).start()
        if spec["kind"] == "training":
            train(spec, rep)
        elif spec["kind"] == "generation":
            generate(spec, rep)
        else:
            raise Fatal(f"unknown job kind {spec['kind']}")
        rep.stop()
        rep.send()
        call("POST", "/complete", json_body={"success": True, "message": rep.message, "log": rep.tail()})
        return 0
    except BaseException as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        rep.stop()
        try:
            call("POST", "/complete", json_body={"success": False, "message": f"{type(e).__name__}: {e}"[:1000],
                                                 "log": (rep.tail() + "\n" + tb)[-6000:]}, retries=3)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
