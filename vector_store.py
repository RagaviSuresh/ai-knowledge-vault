import threading
from typing import List, Dict, Any, Tuple, Optional
from .hnsw_index import HNSWIndex
from .document_representation import DocumentRepresentation, ChunkRepresentation

class VectorStore:
    """
    VectorStore holds the document metadata and uses HNSWIndex to perform
    fast semantic nearest-neighbor search over text chunks.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = HNSWIndex(dim=dim)
        # Thread-safe locks for modifications
        self.lock = threading.Lock()
        
        # Maps doc_id -> DocumentRepresentation
        self.documents: Dict[int, DocumentRepresentation] = {}
        # Stores flat lists of chunks and their vectors for backup searches
        self.all_chunks: List[ChunkRepresentation] = []
        self.all_vectors: List[List[float]] = []

    def add_document(self, doc_rep: DocumentRepresentation, embeddings: List[List[float]]):
        """Indexes a DocumentRepresentation and its chunk embeddings in the vector store."""
        with self.lock:
            # Double-check inside lock to prevent concurrent threads from double-indexing
            if doc_rep.doc_id in self.documents:
                return
                
            if len(doc_rep.chunks) != len(embeddings):
                raise ValueError("Mismatch between document chunks count and embeddings list size.")
                
            self.documents[doc_rep.doc_id] = doc_rep
            
            for chunk, emb in zip(doc_rep.chunks, embeddings):
                self.all_chunks.append(chunk)
                self.all_vectors.append(emb)
                
                # Unique node identifier for HNSW index
                node_id = f"doc_{doc_rep.doc_id}_page_{chunk.page}_idx_{len(self.all_chunks)-1}"
                
                # Add to HNSW index
                self.index.add_item(
                    node_id=node_id,
                    vector=emb,
                    metadata={
                        "doc_id": doc_rep.doc_id,
                        "page": chunk.page,
                        "text": chunk.text
                    }
                )

    def search(self, query_vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the vector store using HNSW index search.
        Falls back to flat cosine similarity scanning if the index has no enter node.
        """
        hits = self.index.search(query_vector, k=k)
        
        # Fallback to flat scan if HNSW returns no hits but vectors are present
        if not hits and self.all_vectors:
            from .hnsw_index import cosine_similarity
            flat_hits = []
            for idx, vec in enumerate(self.all_vectors):
                sim = cosine_similarity(query_vector, vec)
                flat_hits.append((sim, idx))
            # Sort by similarity descending
            flat_hits.sort(key=lambda x: -x[0])
            
            results = []
            for sim, idx in flat_hits[:k]:
                chunk = self.all_chunks[idx]
                doc_rep = self.documents.get(chunk.doc_id)
                results.append({
                    "chunk": chunk,
                    "score": sim,
                    "doc": doc_rep
                })
            return results

        # Process HNSW hits
        results = []
        for sim, node_id in hits:
            meta = self.index.node_metadata.get(node_id, {})
            doc_id = meta.get("doc_id")
            page = meta.get("page", 1)
            text = meta.get("text", "")
            
            doc_rep = self.documents.get(doc_id)
            if doc_rep:
                # Find matching chunk representation object
                chunk_obj = None
                for c in doc_rep.chunks:
                    if c.page == page and c.text == text:
                        chunk_obj = c
                        break
                if not chunk_obj:
                    chunk_obj = ChunkRepresentation(text=text, page=page, doc_id=doc_id)
                    
                results.append({
                    "chunk": chunk_obj,
                    "score": sim,
                    "doc": doc_rep
                })
        return results

    def get_document(self, doc_id: int) -> Optional[DocumentRepresentation]:
        """Retrieves an indexed document representation by ID."""
        return self.documents.get(doc_id)

    def clear(self):
        """Clears all vectors and indexed documents."""
        with self.lock:
            self.index = HNSWIndex(dim=self.dim)
            self.documents.clear()
            self.all_chunks.clear()
            self.all_vectors.clear()
