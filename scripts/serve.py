#!/usr/bin/env python
"""
DeltaV2A Demo API Server (FastAPI)

Endpoints:
  GET  /api/health          -- liveness check
  POST /api/preview-effect  -- apply wand effect -> base64 PNG
  POST /api/infer           -- full pipeline: (image_delta + audio) -> effects

Usage:
  python scripts/serve.py
  python scripts/serve.py --port 8080
  DELTAV2A_CONFIG=configs/my.yaml python scripts/serve.py
"""

import asyncio
import base64
import io
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipeline = None
_vocab_terms: list = []
_inference_lock = threading.Lock()
_CONFIG_PATH = os.environ.get("DELTAV2A_CONFIG", "configs/pipeline.yaml")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="DeltaV2A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pipeline loading (runs once at server startup)
# ---------------------------------------------------------------------------
def _load_pipeline():
    global _pipeline, _vocab_terms

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda")
    output_dir = config["output"]["dir"]

    # CLAP argparse argv hack (CLAP uses argparse at import time)
    import sys as _sys
    _saved = _sys.argv[:]
    _sys.argv = [_sys.argv[0]]
    try:
        from src.models import CLIPEmbedder, CLAPEmbedder
    finally:
        _sys.argv = _saved

    clip_cfg = config["model"]["clip"]
    clap_cfg = config["model"]["clap"]

    print(f"[serve] Loading CLIP {clip_cfg['name']}...")
    clip = CLIPEmbedder(
        model_name=clip_cfg["name"],
        pretrained=clip_cfg["pretrained"],
        device=device,
    )

    print("[serve] Loading CLAP...")
    clap = CLAPEmbedder(
        model_id=clap_cfg["model_id"],
        enable_fusion=clap_cfg["enable_fusion"],
        max_duration=clap_cfg["max_duration"],
        device=device,
    )

    from src.inference import DeltaV2APipeline
    infer_cfg = config.get("inference", {})
    style_temperature = float(infer_cfg.get("style_temperature", 0.1))

    act_raw = infer_cfg.get("activity_threshold_override", None)
    activity_threshold_override = float(act_raw) if act_raw is not None else None

    nc_thr_raw = infer_cfg.get("norm_confidence_threshold", None)
    nc_sc_raw = infer_cfg.get("norm_confidence_scale", None)
    norm_confidence_threshold = float(nc_thr_raw) if nc_thr_raw is not None else None
    norm_confidence_scale = float(nc_sc_raw) if nc_sc_raw is not None else None

    print("[serve] Loading pipeline...")
    _pipeline = DeltaV2APipeline.load(
        artifacts_dir=str(output_dir),
        clip_embedder=clip,
        clap_embedder=clap,
        device=device,
        use_siamese_visual_encoder=False,
        style_temperature=style_temperature,
        activity_threshold_override=activity_threshold_override,
        norm_confidence_threshold=norm_confidence_threshold,
        norm_confidence_scale=norm_confidence_scale,
    )
    _vocab_terms = _pipeline.style_vocab.img_vocab.terms
    print(f"[serve] Pipeline ready. Effects: {_pipeline.effect_names}")
    print(f"[serve] Vocab size: {len(_vocab_terms)}")


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_pipeline)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pil_to_b64(img) -> str:
    """Convert PIL image to base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _bytes_to_audio_np(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """Load audio bytes to mono float32 numpy array, resampled to target_sr."""
    import soundfile as sf
    import librosa

    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def _audio_np_to_b64(audio: np.ndarray, sr: int) -> str:
    """Convert numpy audio array to base64-encoded WAV string."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": _pipeline is not None,
        "effects": _pipeline.effect_names if _pipeline else [],
    }


@app.post("/api/preview-effect")
async def preview_effect(
    image: UploadFile = File(...),
    effect: str = Form(...),
    intensity: float = Form(...),
):
    """Apply a wand image effect and return the result as base64 PNG.

    This endpoint does NOT require the pipeline to be loaded — wand only.
    """
    from PIL import Image as PILImage
    from src.effects.wand_image_effects import apply_effect

    try:
        img_bytes = await image.read()
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        edited_pil = apply_effect(pil_img, effect, float(intensity))
        return {"image_base64": _pil_to_b64(edited_pil)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/infer")
async def infer_endpoint(
    original: UploadFile = File(...),
    effect: str = Form(...),
    intensity: float = Form(...),
    audio: UploadFile = File(...),
):
    """Full inference pipeline.

    1. Apply wand effect to original image -> edited image
    2. CLIP-encode both images -> visual delta z
    3. Map to style vocab distribution
    4. CLAP-encode input audio
    5. Controller predicts audio effect params + activity mask
    6. Render output audio

    Returns style vocab scores, audio effect activations, and processed audio.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet. Try again shortly.")

    from PIL import Image as PILImage
    from src.effects.wand_image_effects import apply_effect
    import torchvision.transforms.functional as TF
    import torch

    orig_bytes = await original.read()
    audio_bytes = await audio.read()

    loop = asyncio.get_event_loop()

    def _run_sync():
        with _inference_lock:
            # --- Image processing ---
            orig_pil = PILImage.open(io.BytesIO(orig_bytes)).convert("RGB")
            edited_pil = apply_effect(orig_pil, effect, float(intensity))

            orig_t = TF.to_tensor(orig_pil).unsqueeze(0)
            edit_t = TF.to_tensor(edited_pil).unsqueeze(0)

            # --- Audio processing ---
            audio_np = _bytes_to_audio_np(audio_bytes, _pipeline.sample_rate)

            # --- Inference ---
            result = _pipeline.infer(orig_t, edit_t, audio_np)

            # --- Build response ---
            preview_b64 = _pil_to_b64(edited_pil)

            output_audio_b64 = (
                _audio_np_to_b64(result.output_audio, _pipeline.sample_rate)
                if result.output_audio is not None
                else ""
            )

            style_scores = [
                {"term": t, "score": round(float(s), 6)}
                for t, s in zip(_vocab_terms, result.img_style_scores)
            ]

            effect_activations = []
            for i, name in enumerate(_pipeline.effect_names):
                active = (
                    bool(result.predicted_activity_mask[i])
                    if result.predicted_activity_mask is not None
                    else False
                )
                prob = (
                    round(float(result.predicted_activity_probs[i]), 4)
                    if result.predicted_activity_probs is not None
                    else 0.5
                )
                params = result.predicted_params_dict.get(name, {})
                effect_activations.append({
                    "name": name,
                    "active": active,
                    "probability": prob,
                    "params": {k: round(float(v), 4) for k, v in params.items()},
                })

            return {
                "preview_image": preview_b64,
                "output_audio_b64": output_audio_b64,
                "sample_rate": int(_pipeline.sample_rate),
                "style_scores": style_scores,
                "top_k_terms": result.top_k_img_terms,
                "top_k_scores": [round(float(s), 4) for s in result.top_k_img_scores],
                "effect_activations": effect_activations,
            }

    try:
        return await loop.run_in_executor(None, _run_sync)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeltaV2A Demo API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--config", default=None, help="Path to pipeline.yaml")
    args = parser.parse_args()

    if args.config:
        _CONFIG_PATH = args.config

    print(f"[serve] Starting DeltaV2A API server on {args.host}:{args.port}")
    print(f"[serve] Config: {_CONFIG_PATH}")
    uvicorn.run(app, host=args.host, port=args.port)
