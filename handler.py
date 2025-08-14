import os
import re
import shlex
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import runpod
import requests

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ma_sam2_worker")

# ---------------- Globals / model state ----------------
MODEL_READY = False
MATANYONE_AVAILABLE = False
MATANYONE_API = None  # Python API object if available

SAM2_AVAILABLE = False
SAM2_PREDICTOR = None  # SAM2 Heavy image predictor

# ---------------- Helpers ----------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _suffix_from_headers(resp: requests.Response, default: str) -> str:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "mp4" in ctype:  return ".mp4"
    if "png" in ctype:  return ".png"
    if "jpeg" in ctype or "jpg" in ctype: return ".jpg"
    if "webp" in ctype: return ".webp"
    return default

def download_to_tmp(url: Optional[str], default_suffix: str) -> Optional[str]:
    if not url:
        return None
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"Invalid URL: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        suffix = _suffix_from_headers(r, default_suffix)
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path

def run_cmd(cmd: str, allow_fail: bool = False) -> Tuple[int, str]:
    log.info("RUN: %s", cmd)
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    out = proc.stdout or ""
    if proc.returncode != 0 and not allow_fail:
        log.error(out[-8000:])
        raise RuntimeError(f"Command failed ({proc.returncode})")
    if out:
        log.debug(out[-8000:])
    return proc.returncode, out

def probe_video(path: str) -> Tuple[float, int, int, float]:
    """
    Return (duration_sec, width, height, fps) via ffprobe.
    """
    cmd = (
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=width,height,avg_frame_rate '
        f'-show_entries format=duration '
        f'-of json "{path}"'
    )
    code, out = run_cmd(cmd)
    data = json.loads(out)
    width = int(data["streams"][0].get("width", 0))
    height = int(data["streams"][0].get("height", 0))
    duration = float(data["format"].get("duration", 0.0))
    afr = data["streams"][0].get("avg_frame_rate", "0/1")
    num, den = (afr.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    if fps <= 0:
        fps = 30.0
    return duration, width, height, fps

def detect_nvenc() -> bool:
    code, out = run_cmd("ffmpeg -hide_banner -encoders", allow_fail=True)
    if code != 0:
        return False
    return ("h264_nvenc" in out) or ("hevc_nvenc" in out)

# ---------------- SAM2: mask on first frame ----------------
def extract_first_frame(src_mp4: str, out_png: str) -> str:
    ensure_dir(os.path.dirname(out_png))
    # -frames:v 1: kun første frame
    run_cmd(f'ffmpeg -y -i "{src_mp4}" -frames:v 1 "{out_png}"')
    return out_png

def sam2_init_heavy() -> None:
    """
    Initialize SAM2 Heavy on CUDA if available (auto-download from HF hub).
    """
    global SAM2_AVAILABLE, SAM2_PREDICTOR
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large",
            device=device
        )
        SAM2_AVAILABLE = True
        log.info("✅ SAM2 Heavy loaded (device=%s).", device)
    except Exception as e:
        SAM2_AVAILABLE = False
        SAM2_PREDICTOR = None
        log.warning("SAM2 init failed (ignored): %s", e)

def sam2_mask(image_path: str) -> str:
    """
    Run SAM2 'predict_everything' on a single image and save a combined mask (uint8 PNG, 0/255).
    """
    if not (SAM2_AVAILABLE and SAM2_PREDICTOR):
        raise RuntimeError("SAM2 not initialized")

    from PIL import Image
    import numpy as np
    import cv2

    image = Image.open(image_path).convert("RGB")
    np_img = np.array(image)

    # 'predict_everything' returns a list of results with .mask attributes
    masks = SAM2_PREDICTOR.predict_everything(image, verbose=False)

    if masks:
        combined = np.zeros(np_img.shape[:2], dtype=np.uint8)
        for m in masks:
            combined = np.logical_or(combined, getattr(m, "mask", None))
        mask_final = (combined.astype(np.uint8) * 255)
    else:
        mask_final = np.zeros(np_img.shape[:2], dtype=np.uint8)

    mask_path = image_path.rsplit(".", 1)[0] + "_mask.png"
    cv2.imwrite(mask_path, mask_final)
