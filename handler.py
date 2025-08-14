import os
import tempfile
import runpod
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Cold start: load models ----------
SAM2_AVAILABLE = False
SAM2_PREDICTOR = None
MATANYONE_AVAILABLE = False
MATANYONE_PROCESSOR = None
MODEL_READY = False

def cold_start():
    global MODEL_READY, SAM2_AVAILABLE, SAM2_PREDICTOR, MATANYONE_AVAILABLE, MATANYONE_PROCESSOR
    if MODEL_READY:
        return
    
    # Load SAM2
    try:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            from sam_2.sam2_image_predictor import SAM2ImagePredictor
        
        SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        SAM2_AVAILABLE = True
        logger.info("✅ SAM2 (Large Model) loaded successfully")
    except Exception as e:
        logger.error(f"🚨 Error loading SAM2: {e}")
        SAM2_AVAILABLE = False

    # Load MatAnyone
    try:
        from matanyone import InferenceCore
        MATANYONE_PROCESSOR = InferenceCore("PeiqingYang/MatAnyone")
        MATANYONE_AVAILABLE = True
        logger.info("✅ MatAnyone processor loaded successfully")
    except Exception as e:
        logger.error(f"🚨 Error loading MatAnyone: {e}")
        MATANYONE_AVAILABLE = False
    
    MODEL_READY = True

# ---------- Utils ----------
def download_to_tmp(url: str, suffix: str) -> str:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------- SAM2 + MatAnyone Pipeline ----------
def process_background_replacement(image_path: str, bg_path: Optional[str], out_path: str) -> None:
    """Complete SAM2 + MatAnyone pipeline"""
    from PIL import Image
    import numpy as np
    import cv2
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Step 1: SAM2 segmentation
    if SAM2_AVAILABLE and SAM2_PREDICTOR:
        logger.info("🎯 Starting SAM2 person segmentation...")
        everything_results = SAM2_PREDICTOR.predict_everything(image, verbose=False)
        
        if everything_results:
            full_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            for mask_result in everything_results:
                full_mask = np.logical_or(full_mask, mask_result.mask)
            mask_np = full_mask.astype(np.uint8) * 255
        else:
            mask_np = np.zeros(image_np.shape[:2], dtype=np.uint8)
    else:
        raise Exception("SAM2 not available")
    
    # Step 2: MatAnyone background replacement
    if MATANYONE_AVAILABLE and MATANYONE_PROCESSOR and bg_path:
        logger.info("🎨 Starting MatAnyone background replacement...")
        
        # Create temporary video for MatAnyone
        temp_dir = Path(tempfile.mkdtemp())
        temp_video = temp_dir / "input.mp4"
        h, w = image_np.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, 1.0, (w, h))
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        out.release()
        
        # Save mask
        mask_path = temp_dir / "mask.png"
        cv2.imwrite(str(mask_path), mask_np)
        
        # Process with MatAnyone
        foreground_path, alpha_path = MATANYONE_PROCESSOR.process_video(
            input_path=str(temp_video),
            mask_path=str(mask_path),
            output_path=str(temp_dir),
            max_size=1080,
            save_frames=False
        )
        
        # Read results and composite with background
        bg_image = Image.open(bg_path).convert("RGB")
        bg_np = np.array(bg_image)
        bg_resized = cv2.resize(bg_np, (w, h))
        
        # Read alpha and foreground
        alpha_cap = cv2.VideoCapture(alpha_path)
        ret, alpha_frame = alpha_cap.read()
        alpha_cap.release()
        
        fg_cap = cv2.VideoCapture(foreground_path)
        ret, fg_frame = fg_cap.read()
        fg_cap.release()
        
        if ret:
            fg_frame = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB)
            alpha_matte = cv2.cvtColor(alpha_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            alpha_3ch = np.stack([alpha_matte] * 3, axis=-1)
            result = (fg_frame * alpha_3ch + bg_resized * (1 - alpha_3ch)).astype(np.uint8)
        else:
            result = image_np
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Just return original if no background or MatAnyone unavailable
        result = image_np
    
    # Save result
    result_image = Image.fromarray(result)
    result_image.save(out_path, format='JPEG', quality=95)

# ---------- Handler ----------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected input:
    {
      "input": {
        "image_url": "https://.../input.jpg",
        "background_url": "https://.../bg.jpg",    # optional
        "output_name": "result.jpg"                # optional
      }
    }
    """
    cold_start()

    data = event.get("input", {}) or {}
    image_url = data.get("image_url")
    bg_url = data.get("background_url")
    out_name = data.get("output_name", "output.jpg")

    if not image_url:
        return {"error": "Missing 'image_url' in input"}

    try:
        # Download files
        in_image = download_to_tmp(image_url, suffix=".jpg")
        in_bg = download_to_tmp(bg_url, suffix=".jpg") if bg_url else None

        # Output location
        ensure_dir("/tmp/outputs")
        out_path = f"/tmp/outputs/{out_name}"

        # Process with SAM2 + MatAnyone
        process_background_replacement(in_image, in_bg, out_path)

        return {
            "status": "success",
            "output_path": out_path,
            "models_loaded": {
                "sam2": SAM2_AVAILABLE,
                "matanyone": MATANYONE_AVAILABLE
            }
        }
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return {"status": "error", "message": str(e)}

# Start serverless
runpod.serverless.start({"handler": handler})
