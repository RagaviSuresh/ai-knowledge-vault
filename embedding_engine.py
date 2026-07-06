import os
import pickle
import threading
from typing import List, Dict

_model_lock = threading.Lock()
_model = None

def get_sentence_transformer():
    """Initializes and returns the sentence transformer model thread-safely."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Use the exact same model BAAI/bge-small-en-v1.5
        _model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _model

class EmbeddingEngine:
    def __init__(self, cache_file_name: str = "chunk_embeddings.pkl"):
        # Locate the cache file in the backend directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_path = os.path.join(backend_dir, cache_file_name)
        self.cache: Dict[str, List[float]] = {}
        self._load_cache()

    def _load_cache(self):
        """Loads precomputed chunk embeddings from the pickle cache."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"EmbeddingEngine: Loaded {len(self.cache)} cached chunk embeddings from {self.cache_path}")
            except Exception as e:
                print(f"EmbeddingEngine: Failed to load cached chunk embeddings: {e}")
        else:
            print(f"EmbeddingEngine: Cache file not found at {self.cache_path}, starting with empty cache.")

    def get_query_embedding(self, query: str) -> List[float]:
        """Generates embedding for a query with standard search prefix."""
        model = get_sentence_transformer()
        query_to_encode = f"Represent this sentence for searching relevant passages: {query}"
        with _model_lock:
            # Return list of floats
            return model.encode(query_to_encode).tolist()

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Retrieves embeddings for a list of texts, checking the cache first
        and generating any missing ones in a single batch.
        """
        if not texts:
            return []
            
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []
        
        # Check cache
        for idx, txt in enumerate(texts):
            if txt in self.cache:
                results[idx] = self.cache[txt]
            else:
                missing_indices.append(idx)
                missing_texts.append(txt)
                
        # Batch encode missing texts
        if missing_texts:
            try:
                model = get_sentence_transformer()
                with _model_lock:
                    # Double check cache inside lock to prevent race conditions
                    still_missing_indices = []
                    still_missing_texts = []
                    for idx, txt in zip(missing_indices, missing_texts):
                        if txt in self.cache:
                            results[idx] = self.cache[txt]
                        else:
                            still_missing_indices.append(idx)
                            still_missing_texts.append(txt)
                            
                    if still_missing_texts:
                        encoded = model.encode(still_missing_texts, batch_size=64).tolist()
                        for idx, txt, emb in zip(still_missing_indices, still_missing_texts, encoded):
                            self.cache[txt] = emb
                            results[idx] = emb
            except Exception as e:
                print(f"EmbeddingEngine: Error batch encoding texts: {e}")
                # Fallback to zeros on error to prevent pipeline crash
                fallback_emb = [0.0] * 384
                for idx in missing_indices:
                    if results[idx] is None:
                        results[idx] = fallback_emb
                        
        return results
