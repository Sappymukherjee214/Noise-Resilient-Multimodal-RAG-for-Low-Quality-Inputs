from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add project root to sys.path for core research modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.noise_engine import apply_multimodal_noise
from src.fusion_engine import AdaptiveMultimodalFusion
from src.retrieval import RobustRetriever
from src.data_loader import FashionDataLoader
from src.mcl import MetaCognitiveLoop, LatentDenoisingBridge

import threading

app = FastAPI(title="NR-M-RAG Sentinel Dashboard")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Research Components (Lazy Loaded)
_system_state = {
    "loader": FashionDataLoader(),
    "fusion": None,
    "retriever": None,
    "mcl": None,
    "bridge": None
}

def get_state():
    if _system_state["fusion"] is None:
        print("[System]: Lazy-loading research components...")
        _system_state["fusion"] = AdaptiveMultimodalFusion()
        _system_state["retriever"] = RobustRetriever()
        _system_state["mcl"] = MetaCognitiveLoop()
        _system_state["bridge"] = LatentDenoisingBridge()
    return _system_state

# ASYNC CLOUD LOADING
def background_load():
    print("[System]: Beginning Asynchronous Cloud Initialization...")
    _system_state["loader"].download_and_init()
    print("[System]: Cloud Initialization Complete.")

threading.Thread(target=background_load, daemon=True).start()

@app.get("/")
async def get_index():
    return FileResponse("web/frontend/index.html")

@app.get("/images/{image_id}")
async def get_dataset_image(image_id: str):
    base_path = _system_state["loader"].get_image_dir()
    if not base_path: return {"error": "Dataset not loaded"}
    img_path = os.path.join(base_path, f"{image_id}.jpg")
    if os.path.exists(img_path):
        return FileResponse(img_path)
    return {"error": "Image not found"}

app.mount("/frontend", StaticFiles(directory="web/frontend"), name="frontend")

@app.post("/process")
async def process_multimodal_query(
    text: str = Form(...),
    sigma: float = Form(0.2),
    image: UploadFile = File(None)
):
    import torch
    state = get_state()
    
    # 1. Load Image
    if image:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        pil_img = Image.new('RGB', (224, 224), color = (73, 109, 137))

    # 2. Inject Controlled Noise
    noisy_text, noisy_img = apply_multimodal_noise(text, pil_img, sigma=sigma)
    
    # 3. Simulate Embeddings
    t_emb = torch.randn(512) 
    v_emb = torch.randn(512)
    
    # 4. Adaptive Fusion
    fused_emb = state["fusion"].fuse_embeddings(t_emb, v_emb, noisy_text, noisy_img)
    
    # 5. Retrieval & MCL
    top_results = state["retriever"].search(fused_emb, top_k=5)
    is_safe = state["mcl"].evaluate_retrieval_safety(top_results)
    intervention = state["mcl"].generate_intervention_strategy(is_safe)
    
    # 6. Encode Image
    buffered = io.BytesIO()
    noisy_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "status": "success",
        "noisy_text_preview": noisy_text,
        "noisy_image_base64": img_str,
        "is_safe": is_safe,
        "intervention": intervention,
        "top_result": top_results[0] if top_results else None,
        "all_scores": top_results,
        "diagnostics": {
            "noise_level": sigma,
            "fusion_tau": state["fusion"].calculate_meta_tau([0.5, 0.5])
        }
    }

@app.get("/samples")
def get_dataset_samples():
    """Returns a few real product items to the UI for testing."""
    df = _system_state["loader"].get_metadata()
    if df is None: return []
    # Filter for items that actually have images
    samples = df.sample(min(len(df), 24))
    return samples[['id', 'productDisplayName', 'baseColour', 'usage']].to_dict('records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
