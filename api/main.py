from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import uuid
import logging
import io
import os
from PIL import Image

from config.settings import settings
from ingestion.dataset_manager import DatasetManager
from preprocessing.cleaner import DataCleaner
from retrieval.engine import RobustRetrievalEngine
from embeddings.manager import EmbeddingManager
from vector_store.store import VectorStore
from generation.generator import RAGGenerator
from evaluation.metrics import EvaluationSuite

# --- Configuration & Logging ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    handlers=[logging.FileHandler("logs/api_json.log"), logging.StreamHandler()]
)
logger = logging.getLogger("NR-M-RAG-API")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade Noise-Resilient Multimodal RAG API",
    version="3.0.0"
)

# --- Rate Limiting (Simple) ---
class RateLimiter:
    def __init__(self, limit: int = 100, window: int = 60):
        self.limit = limit
        self.window = window
        self.requests = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = [now]
            return True
        
        # Clean old requests
        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < self.window]
        
        if len(self.requests[client_id]) < self.limit:
            self.requests[client_id].append(now)
            return True
        return False

limiter = RateLimiter()

# --- Schemas ---
class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    confidence: float
    retrieved_items: int
    processing_time: float
    metrics: Dict[str, Any]

# --- Dependency Injection & Initialization ---
# In a full-scale app, use FastAPI Depends
class SystemState:
    def __init__(self):
        self.dm = DatasetManager()
        self.cleaner = DataCleaner()
        self.em = EmbeddingManager()
        self.vs = VectorStore()
        self.retriever = RobustRetrievalEngine(self.vs, self.em)
        self.generator = RAGGenerator()
        self.evaluator = EvaluationSuite()

state = SystemState()

# --- Middleware ---
@app.middleware("http")
async def security_and_tracking(request: Request, call_next):
    client_ip = request.client.host
    if not limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    response.headers["X-Response-Time"] = f"{duration:.4f}s"
    return response

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "operational", "version": "3.0.0", "engine": "Noise-Resilient-Multimodal-RAG"}

@app.post("/query", response_model=QueryResponse)
async def process_query(
    text: str = Form(..., description="The user query text"),
    image: Optional[UploadFile] = File(None),
    denoise: bool = Form(True),
    prompt_version: str = Form("dynamic")
):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 1. Cleaning
        noise_score = 0.0 # Placeholder for actual noise assessment
        clean_text = state.cleaner.clean_text(text) if denoise else text
        
        pil_img = None
        if image:
            img_bytes = await image.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if denoise:
                pil_img = state.cleaner.denoise_image(pil_img)

        # 2. Retrieval
        search_results = state.retriever.retrieve(clean_text, pil_img, top_k=3)
        
        # 3. Generation
        gen_result = state.generator.generate_response(
            query=clean_text, 
            context_results=search_results, 
            prompt_version=prompt_version,
            noise_score=noise_score
        )
        
        # 4. Evaluation (Real-time)
        context_str = state.generator.optimizer.compress_context(search_results)
        eval_metrics = state.evaluator.evaluate_response(context_str, gen_result["text"])
        
        process_time = time.time() - start_time
        
        logger.info(f"Query processed: id={request_id} lat={process_time:.2f}s score={eval_metrics['faithfulness']:.2f}")

        return QueryResponse(
            request_id=request_id,
            query=text,
            answer=gen_result["text"],
            confidence=1.0 - eval_metrics["hallucination_score"],
            retrieved_items=len(search_results),
            processing_time=process_time,
            metrics=eval_metrics
        )

    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)

