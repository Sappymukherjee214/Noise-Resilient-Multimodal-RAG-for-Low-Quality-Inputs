import json
import os
import hashlib
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Persistent disk-based cache for queries and embeddings."""

    def __init__(self, cache_dir: str = "logs/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_hash(self, key: str) -> str:
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        hash_key = self._get_hash(key)
        filepath = os.path.join(self.cache_dir, f"{hash_key}.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        return None

    def set(self, key: str, value: Any):
        hash_key = self._get_hash(key)
        filepath = os.path.join(self.cache_dir, f"{hash_key}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(value, f)
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
