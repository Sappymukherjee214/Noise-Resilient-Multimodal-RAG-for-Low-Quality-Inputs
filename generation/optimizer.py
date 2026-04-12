from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """Optimizes prompts for latency and cost reduction."""

    @staticmethod
    def compress_context(context_results: List[Dict[str, Any]], max_items: int = 3, max_chars: int = 1000) -> str:
        """
        Compresses retrieval results by selecting top-K and truncating descriptions.
        """
        if not context_results:
            return "No relevant context found."

        compressed_lines = []
        char_count = 0
        
        for i, res in enumerate(context_results[:max_items]):
            meta = res.get('metadata', {})
            name = meta.get('productDisplayName', 'Unknown Item')
            color = meta.get('base_colour', 'N/A')
            
            line = f"[{i+1}] {name} ({color})"
            
            if char_count + len(line) > max_chars:
                break
                
            compressed_lines.append(line)
            char_count += len(line)

        return "\n".join(compressed_lines)

    @staticmethod
    def fallback_strategy(error: str) -> str:
        """Lightweight fallback response when LLM fails."""
        return "I encountered a technical issue. However, I found matching items in our catalog. Please check the results below."
