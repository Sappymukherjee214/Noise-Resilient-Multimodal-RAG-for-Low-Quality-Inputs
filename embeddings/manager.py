import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from PIL import Image
from typing import List, Union
import numpy as np
from config.settings import settings

class EmbeddingManager:
    """Manages text and image embeddings using CLIP and Transformer models."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EmbeddingManager] Using device: {self.device}")
        
        # Load CLIP for Multimodal
        self.clip_model = CLIPModel.from_pretrained(settings.IMAGE_EMBEDDING_MODEL).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(settings.IMAGE_EMBEDDING_MODEL)
        
        # Load Text-only model for reconstruction if needed
        self.text_tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_EMBEDDING_MODEL)
        self.text_model = AutoModel.from_pretrained(settings.TEXT_EMBEDDING_MODEL).to(self.device)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generates embedding for a single text query."""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return text_features.cpu().numpy().flatten()

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generates embedding for a single image."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten()

    def get_multimodal_embedding(self, text: str, image: Image.Image, alpha: float = 0.5) -> np.ndarray:
        """Fuses text and image embeddings using a weighted average."""
        t_emb = self.get_text_embedding(text)
        i_emb = self.get_image_embedding(image)
        
        # Normalize
        t_emb = t_emb / np.linalg.norm(t_emb)
        i_emb = i_emb / np.linalg.norm(i_emb)
        
        fused = (alpha * t_emb) + ((1 - alpha) * i_emb)
        return fused / np.linalg.norm(fused)

if __name__ == "__main__":
    em = EmbeddingManager()
    # Test would go here
