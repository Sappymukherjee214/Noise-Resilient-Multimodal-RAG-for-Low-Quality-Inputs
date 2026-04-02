from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add project root to sys.path for core research modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from noise_engine import apply_multimodal_noise
from fusion_engine import AdaptiveMultimodalFusion
from retrieval import RobustRetriever
from data_loader import FashionDataLoader
from mcl import MetaCognitiveLoop, LatentDenoisingBridge

app = FastAPI(title="NR-M-RAG Sentinel Dashboard")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Research Components
loader = FashionDataLoader()
loader.download_and_init()
fusion = AdaptiveMultimodalFusion()
retriever = RobustRetriever()
mcl = MetaCognitiveLoop()
bridge = LatentDenoisingBridge()

@app.get("/")
async def get_index():
    return FileResponse("web/frontend/index.html")

@app.get("/images/{image_id}")
async def get_dataset_image(image_id: str):
    """Serves raw images from the Kaggle dataset cache for UI preview."""
    # The loader stores the base path
    base_path = loader.get_image_dir()
    img_path = os.path.join(base_path, f"{image_id}.jpg")
    if os.path.exists(img_path):
        return FileResponse(img_path)
    return {"error": "Image not found"}

app.mount("/frontend", StaticFiles(directory="web/frontend"), name="frontend")

@app.post("/process")
async def process_multimodal_query(
    text: str = Form(...),
    sigma: float = Form(0.2), # Noise level slider
    image: UploadFile = File(None)
):
    """Processes a multimodal query and returns diagnostic RAG results."""
    
    # 1. Load Image
    if image:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        # Fallback to a neutral placeholder logic if needed
        pil_img = Image.new('RGB', (224, 224), color = (73, 109, 137))

    # 2. Inject Controlled Noise
    noisy_text, noisy_img = apply_multimodal_noise(text, pil_img, sigma=sigma)
    
    # 3. Simulate Embeddings (Research Proxy)
    t_emb = torch.randn(512) 
    v_emb = torch.randn(512)
    
    # 4. Adaptive Fusion (ENG + MAC)
    fused_emb = fusion.fuse_embeddings(t_emb, v_emb, noisy_text, noisy_img)
    
    # 5. Retrieval & MCL
    top_results = retriever.search(fused_emb, top_k=5)
    is_safe = mcl.evaluate_retrieval_safety(top_results)
    intervention = mcl.generate_intervention_strategy(is_safe)
    
    # 6. Encode Image for UI preview
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
            "fusion_tau": fusion.calculate_meta_tau([0.5, 0.5]) # Proxy
        }
    }

@app.get("/samples")
def get_dataset_samples():
    """Returns a few real product items to the UI for testing."""
    df = loader.get_metadata()
    # Filter for items that actually have images
    samples = df.sample(min(len(df), 24))
    return samples[['id', 'productDisplayName', 'baseColour', 'usage']].to_dict('records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
