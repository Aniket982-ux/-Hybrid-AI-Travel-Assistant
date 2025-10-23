import time
import requests
from typing import List
import config

HF_API_URL = f"https://api-inference.huggingface.co/models/{config.HF_MODEL_NAME}"
HF_HEADERS = {"Authorization": f"Bearer {config.HF_API_KEY}"}

#LRU cache for embeddings for 128 most recent texts
_embedding_cache = {}
_cache_order = []
MAX_CACHE_SIZE = 128

def embed_text(text: str, retries=3) -> List[float]:
    """Get embedding from Hugging Face with LRU caching."""
    #for o(1) lookup
    if text in _embedding_cache:
        return _embedding_cache[text]
    
    for attempt in range(retries):
        try:
            response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=30)
            response.raise_for_status()
            data = response.json()
            embedding = data[0] if isinstance(data, list) and isinstance(data[0], list) else data
            
            _embedding_cache[text] = embedding
            _cache_order.append(text)
            if len(_cache_order) > MAX_CACHE_SIZE:
                oldest = _cache_order.pop(0)
                _embedding_cache.pop(oldest, None)
            
            return embedding
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            if attempt == retries - 1:
                print(f"Error: Embedding API failed after {retries} attempts.")
                raise
            time.sleep(2 ** attempt) # Exponential backoff
