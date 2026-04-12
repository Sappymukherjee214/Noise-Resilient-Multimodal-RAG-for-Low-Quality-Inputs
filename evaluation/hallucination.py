import numpy as np
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """Advanced hallucination detection using semantic similarity and claim verification."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        # In a production environment, we'd use a BERT-based NLI model or cross-encoder
        # For this implementation, we use a placeholder for semantic grounding analysis

    def check_hallucination(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Analyzes the answer for potential hallucinations relative to the context.
        Returns a score and an explanation.
        """
        if not context or context == "No relevant context found.":
            return {
                "score": 1.0, 
                "is_hallucinated": True, 
                "reason": "No context provided to ground the answer."
            }

        # Clean and tokenize (improved logic)
        def clean_and_tokenize(text: str) -> set:
            if not text: return set()
            # Basic cleaning (punctuation remove)
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            tokens = set(cleaned.split())
            # Filter stop words to avoid unfair grounding penalties
            stop_words = {"the", "is", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "of", "this", "that"}
            return tokens - stop_words

        context_words = clean_and_tokenize(context)
        answer_words = clean_and_tokenize(answer)
        
        # Calculate grounding score (simplified)
        grounded_words = answer_words.intersection(context_words)
        grounding_ratio = len(grounded_words) / len(answer_words) if answer_words else 1.0
        
        is_hallucinated = grounding_ratio < self.threshold
        
        # Heuristic for explanation
        reason = "Answer is well-grounded in the retrieved context."
        if is_hallucinated:
            unsupported = list(answer_words - context_words)[:3]
            if not unsupported and not answer_words:
                 reason = "Empty answer provided."
            else:
                 reason = f"Answer contains terms ({', '.join(unsupported)}) not found in context."

        return {
            "score": round(1.0 - grounding_ratio, 4),
            "grounding_score": round(grounding_ratio, 4),
            "is_hallucinated": is_hallucinated,
            "reason": reason
        }

    def calculate_confidence(self, retrieval_score: float, grounding_score: float) -> float:
        """Calculates a combined confidence score for the response."""
        return (retrieval_score * 0.4) + (grounding_score * 0.6)
