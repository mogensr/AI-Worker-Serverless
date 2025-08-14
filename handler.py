import os
import tempfile
import runpod
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import cv2
from PIL import Image

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matanyone-serverless")

# ---------------- Global model flags/state ----------------
MODEL_READY = False
SAM2_AVAILABLE = False
MATANYONE_AVAILABLE = False

# Placeholders for real model objects (når vi integrerer dem)
SAM2_PREDICTOR = None
MATANYONE_PROCESSOR = None

# ---------------- Utils ----------------
def download_to_tmp(url: str, suffix: str) -> str:
    if not url:
        return None
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))

# ---------------- Cold start ----------------
def cold_start():
    """
    Prøv at loade SAM2 og MatAnyone, men vær fejltolerant.
    """
    global MODEL_READY, SAM2_AVAILABLE, MATANYONE_AVAILABLE
    global SAM2_PREDICTOR, MATANYONE_PROCESSOR

    if MODEL_READY:
        return

    # --- SAM2 (BEST EFFORT) ---
    try:
        # TODO: Indsæt den rigtige import + load-måde når bekræftet.
        # Eksempel (pseudo):
        #   from sam2.predictor import Sam2Predictor
        #   SAM2_PREDICTOR = Sam2Predictor.from_pretrained(checkpoint_path=...)
        #   SAM2_AVAILABLE = True
        # For nu kører vi uden hård afhængighed:
        SAM2_AVAILABLE = False
        SAM2_PREDICTOR = None
        logger.info("SAM2 not initialized (placeholder).")
    except Exception as e:
        logger.warning(f"SAM2 load failed (ignored): {e}")
        SAM2_AVAILABLE = False
        SAM2_PREDICTOR = None

    # --- MatAnyone (BEST EFFORT) ---
    try:
        # TODO: Indsæt korrekt API/CLI-kald og checkpoints.
        #   fx: from matanyone.api import MatAnyoneRunner
        #       MATANYONE_PROCESSOR = MatAnyoneRunner(weights=...)
        MATANYONE_AVAILABLE = False
        MATANYONE_PROCESSOR = None
        logger.info("MatAnyone not initialized (placeholder).")
    except Exception as e:
        logger.warning(f"MatAnyone load failed (ignored): {e}")
        MATANYONE_AVAILABLE = False
        MATANYONE_PROCESSOR = None

    MODEL_READY = True
    logger.info("Cold start done. SAM2=%s, MatAnyone=%s", SAM2_AVAILABLE, MATANYONE_AVAILABLE)

# ---------------- Basic compositing helpers ----------------
def composite_foreground_over_background(fg_rgb: np.ndarray, alpha01: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """
    fg_rgb: HxWx3 (uint8)
    alpha01: HxW (float32, 0..1)
    bg_rgb: HxWx3 (uint8) (resizes if needed)
    """
    h, w = fg_rgb.shape[:2]
    if bg_rgb.shape[:2] != (h, w):
        bg_rgb = cv2.resize(bg_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha3 = np.dstack([alpha01]*3)
    out = fg_rgb.astype(np.float32) * alpha3 + bg_rgb.astype(np.float32) * (1.0 - alpha3)
    return np.clip(out, 0, 255).astype(np.uint8)

def grabcut_person_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Meget simpel fallback-maske via GrabCut med en central ROI.
    Returnerer binær maske (uint8 0/255).
    """
    h, w = rgb.shape[:2]
    # central ROI
    x0 = int(w * 0.15)
    y0 = int(h * 0.10)
    x1 = int(w * 0.85)
    y1 = int(h * 0.90)
    rect = (x0, y0, x1 - x0, y1 - y0)

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # 0=BG, 2=PR_BG -> 0 | 2 = BG; 1=FG, 3=PR_FG -> 1 | 3 = FG
    mask_bin = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # let smoothing
    mask_bin = cv2.medianBlur(mask_bin, 5)
    return mask_bin

# ---------------- Main pipeline ----------------
def process_background_replacement(image_path: str, bg_path: Optional[str], out_path: str) -> Dict[str, Any]:
    """
    End-to-end: læs input, lav maske (SAM2/MatAnyone hvis muligt; ellers GrabCut),
    komponer over baggrund, og gem output.
    Returnerer diagnostic-info til response.
    """
    diag = {
        "used_sam2": False,
        "used_matanyone": False,
        "fallback_grabcut": False
    }

    # load image
    image = Image.open(image_path).convert("RGB")
    rgb = pil_to_np(image)

    # --- 1) Prøv SAM2 (TODO: indsæt rigtig inferens når klar) ---
    mask_bin = None
    if SAM2_AVAILABLE and SAM2_PREDICTOR is not None:
        try:
            # TODO: rigtig SAM2-kald; eksempel (pseudo):
            # SAM2_PREDICTOR.set_image(rgb)  # afhænger af API
            # masks = SAM2_PREDICTOR.predict_everything()  # eller tilsvarende
            # mask_bin = aggregate_masks(masks)
            raise NotImplementedError("SAM2 call not implemented yet.")
        except Exception as e:
            logger.warning(f"SAM2 inference failed, falling back. Err: {e}")
            mask_bin = None
    else:
        logger.info("SAM2 unavailable; using fallback.")
        mask_bin = None

    # --- 2) Prøv MatAnyone (kun relevant for video—springes over i dette image-flow) ---
    # NOTE: Din tidligere kode forsøgte at lave 1-frame video -> matting.
    # Når du har et fungerende MatAnyone API/CLI, kan vi:
    #  - konvertere billedet til 1-frame video
    #  - køre MatAnyone for alpha matte
    #  - læse alpha ud og composite
    # For nu bruges det ikke for single image:
    # diag["used_matanyone"] = True  # (når integreret)

    # --- 3) Fallback: GrabCut ---
    if mask_bin is None:
        mask_bin = grabcut_person_mask(rgb)
        diag["fallback_grabcut"] = True

    # alpha fra binær maske
    alpha01 = (mask_bin.astype(np.float32) / 255.0)

    # hvis ingen baggrund -> behold original (eller læg hvid baggrund)
    if not bg_path:
        logger.info("No background provided; returning original image.")
        out_img = rgb
    else:
        bg_img = Image.open(bg_path).convert("RGB")
        bg_rgb = pil_to_np(bg_img)
        out_img = composite_foreground_over_background(rgb, alpha01, bg_rgb)

    # gem
    ensure_dir(os.path.dirname(out_path))
    np_to_pil(out_img).save(out_path, format="JPEG", quality=95)

    return diag

# ---------------- RunPod handler ----------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
    {
      "input": {
        "image_url": "https://.../input.jpg",
        "background_url": "https://.../bg.jpg",  # optional
        "output_name": "result.jpg"               # optional
      }
    }
    """
    cold_start()

    data = event.get("input") or {}
    image_url = data.get("image_url")
    bg_url = data.get("background_url")
    out_name = data.get("output_name", "output.jpg")

    if not image_url:
        return {"status": "error", "message": "Missing 'image_url' in input"}

    try:
        in_image = download_to_tmp(image_url, suffix=".jpg")
        in_bg = download_to_tmp(bg_url, suffix=".jpg") if bg_url else None

        ensure_dir("/tmp/outputs")
        out_path = f"/tmp/outputs/{out_name}"

        diag = process_background_replacement(in_image, in_bg, out_path)

        return {
            "status": "success",
            "output_path": out_path,
            "models_loaded": {
                "sam2": SAM2_AVAILABLE,
                "matanyone": MATANYONE_AVAILABLE
            },
            "pipeline": diag
        }
    except Exception as e:
        logger.exception("Processing failed")
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
