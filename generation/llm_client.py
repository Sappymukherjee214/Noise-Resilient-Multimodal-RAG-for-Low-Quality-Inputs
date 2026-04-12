import os
from typing import Dict, Any, List, Optional
import time
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        pass

class OpenAIClient(LLMClient):
    """Client for OpenAI API with retry logic."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # In a real app, initialize openai client here:
        # self.client = OpenAI(api_key=self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            # Simulate actual API call logic
            logger.info(f"Calling OpenAI with model {self.model}")
            
            # Simulation of response
            response_text = f"Simulated response from {self.model} for: {user_prompt[:50]}..."
            
            latency = time.time() - start_time
            return {
                "text": response_text,
                "model": self.model,
                "latency": latency,
                "usage": {"total_tokens": 100} # Mock usage
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

class AnthropicClient(LLMClient):
    """Client for Anthropic API."""
    
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        # Simulate Anthropic logic
        latency = time.time() - start_time
        return {
            "text": f"Simulated response from Claude for: {user_prompt[:50]}...",
            "model": self.model,
            "latency": latency,
            "usage": {"total_tokens": 80}
        }

class LLMFactory:
    """Factory to create LLM clients."""
    
    @staticmethod
    def get_client(provider: str = "openai", **kwargs) -> LLMClient:
        if provider == "openai":
            return OpenAIClient(**kwargs)
        elif provider == "anthropic":
            return AnthropicClient(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
