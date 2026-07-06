import time
import logging
from typing import List, Dict, Any, Optional
from .document_representation import from_db_model, DocumentRepresentation
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStore
from .hybrid_search import HybridSearch
from .reranker import Reranker
from .confidence_filter import ConfidenceFilter

logger = logging.getLogger("retrieval_pipeline")
logger.setLevel(logging.INFO)

class RetrievalPipeline:
    """
    RetrievalPipeline manages the indexing of document lists and coordinates
    hybrid semantic search, diversity reranking, and confidence filtering.
    """
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.hybrid_search = HybridSearch(self.vector_store, self.embedding_engine)
        self.reranker = Reranker()
        self.confidence_filter = ConfidenceFilter(min_score=60.0, margin=20.0, max_docs=5)

    def search_pipeline(
        self,
        query: str,
        doc_list: List[Any],
        top_n_chunks: int = 6
    ) -> List[Dict[str, Any]]:
        # 1. Map database models to representation models
        doc_representations = []
        for doc in doc_list:
            doc_rep = from_db_model(doc)
            doc_representations.append(doc_rep)
            
        # 3. Perform Hybrid Search
        # This will automatically index documents and generate embeddings in the VectorStore if needed.
        raw_scored_chunks = self.hybrid_search.search(query, doc_representations, top_n_chunks)
        if not raw_scored_chunks:
            return []
            
        # 4. Rerank (Deduplicate pages, apply diversity rules)
        reranked_chunks = self.reranker.rerank(raw_scored_chunks, query, top_n_chunks)
        
        # 5. Format results back to original structure expected by backend
        # Ensure the chunk contains a dict with keys: text, page, and the original SQLAlchemy document instance
        formatted_chunks = []
        for sc in reranked_chunks:
            doc_rep = sc["doc"]
            db_doc = doc_rep.db_doc_instance
            
            # Map back to the dict layout expected by query_vault in ai.py
            formatted_chunks.append({
                "chunk": {
                    "text": sc["chunk"].text,
                    "page": sc["chunk"].page,
                    "doc": db_doc
                },
                "score": sc["score"],
                "semantic_sim": sc["semantic_sim"],
                "metadata_score": sc["metadata_score"],
                "keyword_overlap": sc["keyword_overlap"],
                "semantic_score_exposed": sc["semantic_score_exposed"],
                "metadata_score_exposed": sc["metadata_score_exposed"],
                "keyword_score_exposed": sc["keyword_score_exposed"],
                "intent_bonus": sc["intent_bonus"],
                "exact_keyword_bonus": sc["exact_keyword_bonus"]
            })
            
        return formatted_chunks

# Global pipeline instance
_pipeline = None
_pipeline_lock = threading_lock = time.sleep # we can just use threading.Lock inside the lock function

import threading
_init_lock = threading.Lock()

def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    with _init_lock:
        if _pipeline is None:
            _pipeline = RetrievalPipeline()
    return _pipeline

def retrieve_relevant_chunks_v2(
    query: str,
    doc_list: List[Any],
    top_n_chunks: int = 6
) -> List[Dict[str, Any]]:
    """
    Unified entry point for the new retrieval engine. Matches the interface
    of retrieve_relevant_chunks in the parent package.
    """
    pipeline = get_pipeline()
    return pipeline.search_pipeline(query, doc_list, top_n_chunks)
