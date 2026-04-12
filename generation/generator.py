from typing import List, Dict, Any, Optional
import logging
from generation.llm_client import LLMFactory
from generation.prompt_manager import PromptManager
from generation.optimizer import PromptOptimizer
from retrieval.cache import CacheManager

logger = logging.getLogger(__name__)

class RAGGenerator:
    """Manages production-grade LLM generation with optimization and tracking."""

    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.llm_client = LLMFactory.get_client(provider, model=model)
        self.prompt_manager = PromptManager()
        self.optimizer = PromptOptimizer()
        self.cache = CacheManager()

    def generate_response(
        self, 
        query: str, 
        context_results: List[Dict[str, Any]], 
        prompt_version: str = "v1",
        noise_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Formats context and generates an LLM response with optimization.
        """
        # 1. Caching Check
        cache_key = f"{query}_{prompt_version}_{noise_score}_{len(context_results)}"
        cached_val = self.cache.get(cache_key)
        if cached_val:
            logger.info("Persistent cache hit")
            return cached_val

        # 2. Context Compression
        context_str = self.optimizer.compress_context(context_results)

        # 3. Prompt Selection & Adaptation
        if prompt_version == "dynamic":
            prompts = self.prompt_manager.adapt_prompt(query, noise_score)
            # Replace placeholder if present
            if "{context}" in prompts["user"]:
                prompts["user"] = prompts["user"].replace("{context}", context_str)
            else:
                prompts["user"] += f"\n\nCONTEXT:\n{context_str}"
        else:
            prompts = self.prompt_manager.get_prompt(prompt_version, query=query, context=context_str)

        # 4. LLM Call with Fallback
        try:
            result = self.llm_client.generate(
                system_prompt=prompts["system"], 
                user_prompt=prompts["user"]
            )
            
            # Add metadata
            result["prompt_version"] = prompt_version
            result["context_count"] = len(context_results)
            result["noise_score"] = noise_score
            
            # 5. Store in cache
            self.cache.set(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "text": self.optimizer.fallback_strategy(str(e)),
                "error": str(e)
            }


if __name__ == "__main__":
    gen = RAGGenerator()
    results = gen.generate_response("Blue shirts", [], prompt_version="v1")
    print(f"Answer: {results['text']}")
    print(f"Latency: {results.get('latency', 0):.4f}s")

