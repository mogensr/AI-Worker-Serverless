from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import numpy as np
import logging
from io import BytesIO

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Model Loading ---
SAM2_AVAILABLE = False
SAM2_PREDICTOR = None

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    # This will be slow on startup, which is expected for a worker.
    SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    SAM2_AVAILABLE = True
    logger.info("✅ SAM2 (Large Model) loaded successfully")
except ImportError:
    logger.warning("⚠️ SAM2 not available. Please ensure it's installed via requirements.txt.")
except Exception as e:
    logger.error(f"🚨 Error loading SAM2 model: {e}")

app = FastAPI()

def segment_person_sam2(image_np, predictor):
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

@app.get("/")
def read_root():
    return {"message": "AI Worker is running"}

@app.post("/segment/", response_class=Response)
async def segment_image(file: UploadFile = File(...)):
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
