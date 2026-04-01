import torch
from PIL import Image
from noise_engine import apply_multimodal_noise
from fusion_engine import AdaptiveMultimodalFusion
from retrieval import RobustRetriever
from data_loader import FashionDataLoader
import os

def run_prototype():
    print("--- [NR-M-RAG Prototype | Fashion Dataset Edition] ---")
    
    # Initialize Data Loader
    loader = FashionDataLoader()
    try:
        loader.download_and_init()
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
        
    # Pick a random product from the Kaggle dataset
    clean_text, image, metadata = loader.get_random_sample()
    
    print(f"\n[Selected Product]: ID {metadata['id']} | {metadata['category']}")
    print(f"[Reference Colour]: {metadata['base_colour']}")
    
    # 1. Simulate High Noise Condition (Low-Quality Input)
    noise_level = 0.6
    noisy_text, noisy_image = apply_multimodal_noise(clean_text, image, noise_level=noise_level)
    
    print(f"\n[Input Clean]: '{clean_text}'")
    print(f"[Input Noisy]: '{noisy_text}' (Noise Level: {noise_level})")
    
    # Save noisy version for inspection
    noisy_image.save("noisy_sample.png")
    print("[System]: Noisy image saved as 'noisy_sample.png' for visual debugging.")
    
    # 2. Adaptive Multimodal Fusion
    # In a real system, we'd use CLIP/BLIP here. Using dummy embeddings for 512-dim.
    t_emb = torch.randn(512)
    v_emb = torch.randn(512)
    
    fusion = AdaptiveMultimodalFusion()
    fused_emb = fusion.fuse_embeddings(t_emb, v_emb, noisy_text, noisy_image)
    
    # 3. Robust Retrieval
    retriever = RobustRetriever(embedding_dim=512)
    results = retriever.search(fused_emb, top_k=3)
    
    print("\n--- [System Observations] ---")
    # Access DQE via fusion for display
    t_conf = fusion.dqe.estimate_text_reliability(noisy_text)
    v_conf = fusion.dqe.estimate_image_reliability(noisy_image)
    
    print(f"Text Reliability Score (0-1): {t_conf:.2f}")
    print(f"Image Reliability Score (0-1): {v_conf:.2f}")
    
    # Calculate relative weights (simplified)
    # The actual weights are from a softmax in the fusion class.
    total = t_conf + v_conf
    print(f"System Decision: Dynamic Priority shifted towards '{'Text' if t_conf > v_conf else 'Vision'}'")
    
    print("\n[Retrieval Results (Candidates)]:")
    for doc, score in results:
        print(f" - {doc} (Similarity: {score:.4f})")

if __name__ == "__main__":
    run_prototype()
