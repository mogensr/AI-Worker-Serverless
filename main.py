from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from PIL import Image
import numpy as np
import logging
from io import BytesIO
import torch
import cv2
import tempfile
from pathlib import Path
from typing import Optional

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Model Loading ---
SAM2_AVAILABLE = False
SAM2_PREDICTOR = None
MATANYONE_AVAILABLE = False
MATANYONE_PROCESSOR = None

# Load SAM2 for person segmentation
try:
    # Try both possible import names for SAM2
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        from sam_2.sam2_image_predictor import SAM2ImagePredictor
    
    SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    SAM2_AVAILABLE = True
    logger.info("✅ SAM2 (Large Model) loaded successfully")
except ImportError:
    logger.warning("⚠️ SAM2 not available. Please ensure it's installed via requirements.txt.")
    SAM2_AVAILABLE = False
    SAM2_PREDICTOR = None
except Exception as e:
    logger.error(f"🚨 Error loading SAM2 model: {e}")
    SAM2_AVAILABLE = False
    SAM2_PREDICTOR = None

# Load MatAnyone for background replacement
try:
    from matanyone import InferenceCore
    MATANYONE_PROCESSOR = InferenceCore("PeiqingYang/MatAnyone")
    MATANYONE_AVAILABLE = True
    logger.info("✅ MatAnyone processor loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ MatAnyone not available: {e}")
except Exception as e:
    logger.error(f"🚨 Error loading MatAnyone processor: {e}")

app = FastAPI()

def segment_person_sam2(image_np, predictor):
    """Use SAM2 to create person segmentation mask"""
    if not SAM2_AVAILABLE or predictor is None:
        raise HTTPException(status_code=503, detail="SAM2 model is not available.")
    
    input_image = Image.fromarray(image_np)
    everything_results = predictor.predict_everything(input_image, verbose=False)
    
    if not everything_results:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    full_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for mask_result in everything_results:
        full_mask = np.logical_or(full_mask, mask_result.mask)
        
    return full_mask.astype(np.uint8) * 255

def replace_background_matanyone(image_np, mask_np, background_image_np=None):
    """Use MatAnyone for sophisticated background replacement based on SAM2 mask"""
    if not MATANYONE_AVAILABLE or MATANYONE_PROCESSOR is None:
        raise HTTPException(status_code=503, detail="MatAnyone processor is not available.")
    
    try:
        # Create temporary directory for MatAnyone processing
        temp_dir = Path(tempfile.mkdtemp())
        
        # Save input frame as temporary video (MatAnyone expects video)
        temp_video = temp_dir / "input.mp4"
        h, w = image_np.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, 1.0, (w, h))
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        out.release()
        
        # Save SAM2 mask
        mask_path = temp_dir / "mask.png"
        cv2.imwrite(str(mask_path), mask_np)
        
        logger.info("🎬 Processing with MatAnyone...")
        
        # Process with MatAnyone
        foreground_path, alpha_path = MATANYONE_PROCESSOR.process_video(
            input_path=str(temp_video),
            mask_path=str(mask_path),
            output_path=str(temp_dir),
            max_size=1080,
            save_frames=False
        )
        
        # Read the alpha matte result
        alpha_cap = cv2.VideoCapture(alpha_path)
        ret, alpha_frame = alpha_cap.read()
        alpha_cap.release()
        
        if not ret:
            raise ValueError("Failed to read MatAnyone alpha result")
        
        # Read the foreground result
        fg_cap = cv2.VideoCapture(foreground_path)
        ret, fg_frame = fg_cap.read()
        fg_cap.release()
        
        if not ret:
            raise ValueError("Failed to read MatAnyone foreground result")
        
        # Convert back to RGB
        fg_frame = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB)
        
        # Convert alpha to single channel if needed
        if len(alpha_frame.shape) == 3:
            alpha_matte = cv2.cvtColor(alpha_frame, cv2.COLOR_BGR2GRAY)
        else:
            alpha_matte = alpha_frame
        
        # Normalize alpha to 0-1 range
        alpha_matte = alpha_matte.astype(np.float32) / 255.0
        
        # If background provided, composite with it
        if background_image_np is not None:
            # Resize background to match frame size
            bg_resized = cv2.resize(background_image_np, (w, h))
            
            # Alpha composite: result = fg * alpha + bg * (1 - alpha)
            alpha_3ch = np.stack([alpha_matte] * 3, axis=-1)
            result = (fg_frame * alpha_3ch + bg_resized * (1 - alpha_3ch)).astype(np.uint8)
        else:
            # Return foreground with alpha channel
            result = fg_frame
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass  # Ignore cleanup errors
        
        logger.info("✅ MatAnyone background replacement complete")
        return result
        
    except Exception as e:
        logger.error(f"❌ MatAnyone processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"MatAnyone processing failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "AI Worker is running"}

@app.post("/segment/", response_class=Response)
async def segment_image(file: UploadFile = File(...)):
    """Legacy endpoint - returns only the segmentation mask"""
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    mask_np = segment_person_sam2(image_np, SAM2_PREDICTOR)

    mask_image = Image.fromarray(mask_np)
    buf = BytesIO()
    mask_image.save(buf, format='PNG')
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/replace_background/", response_class=Response)
async def replace_background(
    file: UploadFile = File(...),
    background_file: Optional[UploadFile] = File(None)
):
    """Complete SAM2 + MatAnyone pipeline for background replacement"""
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Load background image if provided
    background_np = None
    if background_file:
        bg_contents = await background_file.read()
        try:
            bg_image = Image.open(BytesIO(bg_contents)).convert("RGB")
            background_np = np.array(bg_image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid background image file")

    # Step 1: Use SAM2 to segment person
    logger.info("🎯 Starting SAM2 person segmentation...")
    mask_np = segment_person_sam2(image_np, SAM2_PREDICTOR)
    
    # Step 2: Use MatAnyone for sophisticated background replacement
    logger.info("🎨 Starting MatAnyone background replacement...")
    result_np = replace_background_matanyone(image_np, mask_np, background_np)
    
    # Return the final result
    result_image = Image.fromarray(result_np)
    buf = BytesIO()
    result_image.save(buf, format='JPEG', quality=95)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/jpeg")
