import numpy as np
import json
import os
from typing import List, Dict, Any
from scipy.spatial.distance import cosine

class VectorStore:
    """A clean, modular vector store implementation for semantic retrieval."""

    def __init__(self, index_path: str = "vector_store/index.json"):
        self.index_path = index_path
        self.embeddings = []
        self.metadata = []
        self.load()

    def add(self, vector: np.ndarray, meta: Dict[str, Any]):
        """Adds a vector and its associated metadata to the store."""
        self.embeddings.append(vector.tolist())
        self.metadata.append(meta)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs cosine similarity search."""
        if not self.embeddings:
            return []
            
        embeddings_np = np.array(self.embeddings)
        # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
        # For normalized vectors, it's just the dot product
        similarities = np.dot(embeddings_np, query_vector)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        return results

    def save(self):
        """Persists the index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f)

    def load(self):
        """Loads the index from disk."""
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                data = json.load(f)
                self.embeddings = data.get("embeddings", [])
                self.metadata = data.get("metadata", [])
            print(f"[VectorStore] Loaded {len(self.embeddings)} entries.")

if __name__ == "__main__":
    vs = VectorStore()
    # Test would go here
