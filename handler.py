import os
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
MATANYONE_API = None

SAM2_AVAILABLE = False
SAM2_PREDICTOR = None  # SAM2ImagePredictor

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
    run_cmd(f'ffmpeg -y -i "{src_mp4}" -frames:v 1 "{out_png}"')
    return out_png

def sam2_init_heavy() -> None:
    """
    Initialize SAM2 on CUDA if available, weights are pre-fetched in /models/hf-cache.
    """
    global SAM2_AVAILABLE, SAM2_PREDICTOR
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load from HF snapshot; from_pretrained will hit $HF_HOME
        SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large",
            device=device
        )
        SAM2_AVAILABLE = True
        log.info("✅ SAM2 loaded (device=%s).", device)
    except Exception as e:
        SAM2_AVAILABLE = False
        SAM2_PREDICTOR = None
        log.warning("SAM2 init failed (ignored): %s", e)

def sam2_mask(image_path: str) -> str:
    """
    Use SAM2 Automatic Mask Generator on a single image and save one-person mask (uint8 PNG, 0/255).
    Heuristic: pick largest plausible mask by area.
    """
    if not (SAM2_AVAILABLE and SAM2_PREDICTOR):
        raise RuntimeError("SAM2 not initialized")

    from PIL import Image
    import numpy as np
    import cv2

    # AMG lives in sam2; class name per docs
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as e:
        raise RuntimeError(f"SAM2 AMG not available: {e}")

    image = np.array(Image.open(image_path).convert("RGB"))
    amg = SAM2AutomaticMaskGenerator(SAM2_PREDICTOR)
    masks = amg.generate(image) or []

    if len(masks) == 0:
        # No masks -> return all zeros
        mask_final = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        # Choose the largest reasonable area (1%..80%) as the "person"
        H, W = image.shape[:2]
        total = H * W
        best = None
        best_area = 0
        for m in masks:
            seg = m.get("segmentation", None)
            if seg is None:
                continue
            area = int(m.get("area", int(seg.sum())))
            area_ratio = area / float(total)
            if 0.01 <= area_ratio <= 0.80 and area > best_area:
                best = seg
                best_area = area
        if best is None:
            # Fallback: just take the largest
            best = max(masks, key=lambda x: x.get("area", 0)).get("segmentation")
        mask_final = (best.astype(np.uint8) * 255)

        # Light morphology for cleaner edges
        kernel = np.ones((3, 3), np.uint8)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask_path = image_path.rsplit(".", 1)[0] + "_mask.png"
    cv2.imwrite(mask_path, mask_final)
    return mask_path

# ---------------- Cold start ----------------
def cold_start():
    """
    Load MatAnyone and SAM2 from local baked paths (no network),
    enable CUDA perf tweaks, and stay fault-tolerant.
    """
    global MODEL_READY, MATANYONE_AVAILABLE, MATANYONE_API
    if MODEL_READY:
        return

    # Torch/CUDA perf
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("CUDA available: %s", torch.cuda.get_device_name(0))
    except Exception as e:
        log.warning("Torch perf setup skipped: %s", e)

    # ---- MatAnyone from HF snapshot baked in Docker (/models/matanyone) ----
    try:
        from matanyone import InferenceCore
        # InferenceCore accepts an HF repo ID OR a local snapshot path.
        MATANYONE_API = InferenceCore("/models/matanyone")
        MATANYONE_AVAILABLE = True
        log.info("✅ MatAnyone loaded from /models/matanyone")
    except Exception as e:
        MATANYONE_AVAILABLE = False
        MATANYONE_API = None
        log.warning("MatAnyone init failed (will continue): %s", e)

    # ---- SAM2 ----
    sam2_init_heavy()

    MODEL_READY = True

# ---------------- MatAnyone wrappers ----------------
def run_matanyone_python(input_mp4: str, out_dir: str, max_size: int,
                         mask_path: Optional[str]) -> Tuple[str, str]:
    if MATANYONE_API is None:
        raise RuntimeError("MatAnyone API not initialized")

    kwargs = dict(
        input_path=input_mp4,
        output_path=out_dir,
        max_size=max_size,
        save_frames=False
    )
    if mask_path:
        kwargs["mask_path"] = mask_path  # external SAM2 mask

    result = MATANYONE_API.process_video(**kwargs)

    if isinstance(result, dict):
        fg = result.get("foreground_path")
        alpha = result.get("alpha_path")
    elif isinstance(result, (list, tuple)) and len(result) >= 2:
        fg, alpha = result[0], result[1]
    else:
        fg = alpha = None

    if not (fg and alpha and os.path.exists(fg) and os.path.exists(alpha)):
        raise RuntimeError("MatAnyone API did not return expected outputs.")
    return fg, alpha

def run_matanyone_cli(input_mp4: str, out_dir: str, max_size: int,
                      mask_path: Optional[str]) -> Tuple[str, str]:
    ensure_dir(out_dir)
    base = f'--video "{input_mp4}" --out_dir "{out_dir}" --max_size {max_size}'
    if mask_path:
        base += f' --mask_path "{mask_path}"'

    candidates = [
        f'python -m matanyone.infer {base}',
        f'python -m matanyone.demo {base}',
    ]
    last_err = None
    for cmd in candidates:
        try:
            run_cmd(cmd)
            fg = os.path.join(out_dir, "foreground.mp4")
            alpha = os.path.join(out_dir, "alpha.mp4")
            if os.path.exists(fg) and os.path.exists(alpha):
                return fg, alpha
        except Exception as e:
            last_err = e
    raise RuntimeError(f"MatAnyone CLI failed: {last_err}")

# ---------------- Compose (ffmpeg; NVENC when available) ----------------
def compose_video(fg_mp4: str, alpha_mp4: str, bg_img: str,
                  duration: float, out_mp4: str, src_video_for_audio: str,
                  use_nvenc: bool) -> str:
    dur = max(0.1, duration)
    tmp_bg = os.path.join(os.path.dirname(out_mp4), "bg_loop.mp4")
    run_cmd(
        f'ffmpeg -y -loop 1 -i "{bg_img}" -t {dur:.3f} -r 30 '
        f'-pix_fmt yuv420p -c:v libx264 -preset veryfast "{tmp_bg}"'
    )

    have_nvenc = detect_nvenc() if use_nvenc else False
    vcodec = "h264_nvenc" if have_nvenc else "libx264"
    preset = "p6" if have_nvenc else "veryfast"
    crf_or_cq = "-cq 22" if have_nvenc else "-crf 18"

    cmd = (
        f'ffmpeg -y -i "{fg_mp4}" -i "{alpha_mp4}" -i "{tmp_bg}" -i "{src_video_for_audio}" '
        f'-filter_complex "[1:v]format=gray[alpha];[0:v][alpha]alphamerge[fgrgba];'
        f'[2:v][fgrgba]overlay=shortest=1[outv]" '
        f'-map "[outv]" -map 3:a? -c:v {vcodec} -preset {preset} {crf_or_cq} '
        f'-c:a aac -shortest "{out_mp4}"'
    )
    run_cmd(cmd)
    return out_mp4

# ---------------- Main processing ----------------
def process_video_with_background(video_path: str, bg_path: str, out_path: str,
                                  run_sam2_mode: str, max_size: int,
                                  use_nvenc: bool) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(out_path))
    duration, width, height, fps = probe_video(video_path)
    workdir = Path(tempfile.mkdtemp(prefix="ma_sam2_"))
    out_dir = str(workdir / "ma_out")
    ensure_dir(out_dir)

    # --- SAM2: first-frame mask (cheap & stable) ---
    mask_path = None
    used_sam2 = False
    if run_sam2_mode != "none" and SAM2_AVAILABLE:
        try:
            first_frame_png = str(workdir / "first_frame.png")
            extract_first_frame(video_path, first_frame_png)
            mask_path = sam2_mask(first_frame_png)  # uint8 PNG, 0/255
            used_sam2 = True
            log.info("SAM2 mask generated: %s", mask_path)
        except Exception as e:
            log.warning("SAM2 mask generation failed; proceeding without mask. Err: %s", e)
            mask_path = None

    # --- MatAnyone (Python API -> CLI fallback) ---
    try:
        if MATANYONE_AVAILABLE:
            fg_mp4, alpha_mp4 = run_matanyone_python(
                input_mp4=video_path, out_dir=out_dir, max_size=max_size, mask_path=mask_path
            )
            ma_mode = "python_api"
        else:
            fg_mp4, alpha_mp4 = run_matanyone_cli(
                input_mp4=video_path, out_dir=out_dir, max_size=max_size, mask_path=mask_path
            )
            ma_mode = "cli"
    except Exception as e:
        raise RuntimeError(f"MatAnyone failed: {e}")

    # --- Compose ---
    final_path = compose_video(fg_mp4, alpha_mp4, bg_path, duration, out_path, video_path, use_nvenc)

    return {
        "duration_sec": duration,
        "resolution": [width, height],
        "fps": fps,
        "matanyone_mode": ma_mode,
        "used_sam2": used_sam2,
        "outputs": {
            "foreground_mp4": fg_mp4,
            "alpha_mp4": alpha_mp4,
            "final_mp4": final_path
        },
        "workdir": str(workdir)
    }

# ---------------- RunPod handler ----------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
    {
      "input": {
        "video_url": "...",
        "background_url": "...",
        "output_name": "result.mp4",
        "run_sam2": "none|first",   // default: "first"
        "max_size": 1080,
        "use_nvenc": true
      }
    }
    """
    cold_start()
    data = event.get("input") or {}
    video_url = data.get("video_url")
    bg_url = data.get("background_url")
    out_name = data.get("output_name", "output.mp4")
    run_sam2_mode = (data.get("run_sam2") or "first").lower()
    max_size = int(data.get("max_size") or 1080)
    use_nvenc = bool(data.get("use_nvenc") if data.get("use_nvenc") is not None else True)

    if not video_url or not bg_url:
        return {"status": "error", "message": "Both 'video_url' and 'background_url' are required."}
    if run_sam2_mode not in {"none", "first"}:
        run_sam2_mode = "first"

    try:
        in_video = download_to_tmp(video_url, ".mp4")
        in_bg = download_to_tmp(bg_url, ".jpg")
        ensure_dir("/tmp/outputs")
        out_path = f"/tmp/outputs/{out_name}"

        diag = process_video_with_background(
            in_video, in_bg, out_path,
            run_sam2_mode=run_sam2_mode,
            max_size=max_size,
            use_nvenc=use_nvenc
        )

        return {
            "status": "success",
            "output_path": out_path,
            "matanyone_loaded": MATANYONE_AVAILABLE,
            "sam2_loaded": SAM2_AVAILABLE,
            "details": diag
        }
    except Exception as e:
        log.exception("Processing failed")
        return {"status": "error", "message": str(e)}

# Start serverless
runpod.serverless.start({"handler": handler})
