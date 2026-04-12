import sys
import os
from PIL import Image

# Add project root to path
sys.path.append(os.getcwd())

from ingestion.dataset_manager import DatasetManager
from embeddings.manager import EmbeddingManager
from vector_store.store import VectorStore
from tqdm import tqdm

def initialize():
    print("[Init] Starting Database Initialization...")
    
    dm = DatasetManager()
    dm.download_and_init()
    
    em = EmbeddingManager()
    vs = VectorStore()
    
    # Ingesting 1 item for test
    limit = 1
    print(f"[Init] Ingesting {limit} items into vector store...")
    
    for i in tqdm(range(limit)):
        try:
            sample = dm.get_sample(i)
            text = sample['text']
            img_path = sample['image_path']
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                # Get multimodal embedding
                emb = em.get_multimodal_embedding(text, img)
                
                # Add to store
                vs.add(emb, {
                    "productDisplayName": text,
                    "id": sample['id'],
                    "base_colour": sample['metadata']['base_colour'],
                    "category": sample['metadata']['category']
                })
        except Exception as e:
            continue
            
    vs.save()
    print(f"[Init] Done. Vector store saved at {vs.index_path}")

if __name__ == "__main__":
    initialize()
