from typing import Dict, Any, Optional
import json
import os

class PromptManager:
    """Manages versioned prompt templates and dynamic adaptation."""
    
    def __init__(self, templates_path: str = "config/prompts/"):
        self.templates_path = templates_path
        self.prompts = {
            "v1": {
                "system": (
                    "You are an AI Fashion Assistant. Base your answer ONLY on the provided context. "
                    "If the context is empty, politely inform the user."
                ),
                "user": "USER QUERY: {query}\n\nRETRIEVED CONTEXT:\n{context}\n\nResponse:"
            },
            "v2_robust": {
                "system": (
                    "You are a specialized Noise-Resilient Fashion Assistant. "
                    "Handle noisy or ambiguous queries with grace. Acknowledge potential mismatches."
                ),
                "user": (
                    "POTENTIALLY NOISY QUERY: {query}\n"
                    "INTENT CONFIDENCE: {confidence}\n\n"
                    "CANDIDATE PRODUCTS:\n{context}\n\n"
                    "Synthesize a robust response:"
                )
            }
        }
        
    def get_prompt(self, version: str, **kwargs) -> Dict[str, str]:
        """Returns a formatted prompt based on version and input data."""
        template = self.prompts.get(version, self.prompts["v1"])
        
        system = template["system"]
        user = template["user"]
        
        # Safely format the user prompt
        # Fallback for missing keys: keep them or use 'N/A'
        for key, value in kwargs.items():
            user = user.replace(f"{{{key}}}", str(value))
            
        return {
            "system": system,
            "user": user
        }

    def adapt_prompt(self, query: str, noise_score: float) -> Dict[str, str]:
        """Dynamically chooses a prompt based on input noise level."""
        if noise_score > 0.5:
            # Use robust prompt for high-noise inputs
            return self.get_prompt("v2_robust", query=query, confidence="Low", context="{context}")
        else:
            # Use standard prompt for clean inputs
            return self.get_prompt("v1", query=query, context="{context}")
