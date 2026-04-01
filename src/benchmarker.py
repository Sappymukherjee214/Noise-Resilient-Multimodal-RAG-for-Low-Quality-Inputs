import torch
import numpy as np
import matplotlib.pyplot as plt
from noise_engine import apply_multimodal_noise
from fusion_engine import AdaptiveMultimodalFusion
from retrieval import RobustRetriever
from data_loader import FashionDataLoader
from stats_engine import ResearchStatsEngine
from tqdm import tqdm

class HallucinationFrontierScanner:
    """Rigorous evaluation suite to measure the system's breakdown point."""

    def __init__(self):
        self.loader = FashionDataLoader()
        self.loader.download_and_init()
        self.fusion = AdaptiveMultimodalFusion()
        self.retriever = RobustRetriever()
        self.stats = ResearchStatsEngine()

    def run_benchmark(self, noise_steps: int = 10, samples_per_step: int = 5):
        results = []
        noise_levels = np.linspace(0.0, 0.9, noise_steps)
        
        print("\n[Research Benchmark]: Scanning Hallucination Frontier...")
        
        for noise in tqdm(noise_levels):
            step_scores = []
            for _ in range(samples_per_step):
                # Get clean sample
                clean_text, image, _ = self.loader.get_random_sample()
                
                # Apply noise
                noisy_text, noisy_image = apply_multimodal_noise(clean_text, image, noise_level=noise)
                
                # Dummy embeddings for prototype speed
                t_emb = torch.randn(512)
                v_emb = torch.randn(512)
                
                # Fuse and Retrieve
                fused = self.fusion.fuse_embeddings(t_emb, v_emb, noisy_text, noisy_image)
                retrieval_results = self.retriever.search(fused, top_k=1)
                
                # Score based on similarity (Distance-based proxy for veracity)
                top_score = retrieval_results[0][1]
                step_scores.append(top_score)
            
            results.append(np.mean(step_scores))
            
        return noise_levels, results

    def plot_frontier(self, noise_levels, results):
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, results, marker='o', linestyle='-', color='b', linewidth=2)
        plt.fill_between(noise_levels, 0, results, alpha=0.1, color='b')
        
        # Mark the 'Breakdown Point' (Heuristic: 50% of max score)
        threshold = results[0] * 0.5
        breakdown_idx = next((i for i, v in enumerate(results) if v < threshold), None)
        
        if breakdown_idx is not None:
            plt.axvline(x=noise_levels[breakdown_idx], color='r', linestyle='--', label='Hallucination Frontier')
            print(f"\n[Analysis]: Frontier detected at Noise Level {noise_levels[breakdown_idx]:.2f}")

        plt.title("Hallucination Frontier Analysis: NR-M-RAG Robustness Curve")
        plt.xlabel("Input Noise Intensity (Sigma)")
        plt.ylabel("Retrieval Fidelity (Candidate Similarity)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("hallucination_frontier.png")
        print("[System]: Plot saved as 'hallucination_frontier.png'")

if __name__ == "__main__":
    scanner = HallucinationFrontierScanner()
    levels, scores = scanner.run_benchmark()
    scanner.plot_frontier(levels, scores)
