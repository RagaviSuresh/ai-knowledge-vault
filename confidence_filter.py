from typing import List, Dict, Any

class ConfidenceFilter:
    """
    ConfidenceFilter applies threshold checks on scored chunks.
    Ensures that only chunks close to the top-scoring candidate are returned,
    and caps the total number of unique documents returned.
    """
    def __init__(self, min_score: float = 60.0, margin: float = 20.0, max_docs: int = 5):
        self.min_score = min_score
        self.margin = margin
        self.max_docs = max_docs

    def filter(self, top_chunks_scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            if not top_chunks_scored:
                return []
                
            top_score = max(sc["score"] for sc in top_chunks_scored if sc is not None and "score" in sc)
            
            filtered_chunks = []
            for sc in top_chunks_scored:
                if sc is None or "score" not in sc or "chunk" not in sc:
                    continue
                # Score must exceed minimum threshold and lie within 'margin' of top score
                if sc["score"] >= self.min_score and sc["score"] >= (top_score - self.margin):
                    filtered_chunks.append(sc)
                    
            # Track unique documents to limit to max_docs
            seen_docs = set()
            allowed_doc_ids = set()
            for sc in filtered_chunks:
                chunk = sc.get("chunk")
                if chunk is None:
                    continue
                doc_id = getattr(chunk, "doc_id", None)
                if doc_id is None:
                    # Check if it has doc key (for compat)
                    doc_obj = sc.get("doc")
                    if doc_obj:
                        doc_id = getattr(doc_obj, "doc_id", None)
                if doc_id is not None and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    if len(allowed_doc_ids) < self.max_docs:
                        allowed_doc_ids.add(doc_id)
                        
            final_chunks = []
            for sc in filtered_chunks:
                chunk = sc.get("chunk")
                doc_id = getattr(chunk, "doc_id", None)
                if doc_id is None:
                    doc_obj = sc.get("doc")
                    if doc_obj:
                        doc_id = getattr(doc_obj, "doc_id", None)
                if doc_id in allowed_doc_ids:
                    final_chunks.append(sc)
                    
            return final_chunks
        except Exception as e:
            print(f"ConfidenceFilter: Error in document filtering: {e}")
            # Fallback to Top max_docs documents in case of failure
            fallback_chunks = []
            seen_docs = set()
            try:
                safe_chunks_scored = top_chunks_scored if top_chunks_scored is not None else []
                for sc in safe_chunks_scored:
                    if sc is None or "chunk" not in sc:
                        continue
                    chunk = sc["chunk"]
                    doc_id = getattr(chunk, "doc_id", None)
                    if doc_id is None:
                        doc_obj = sc.get("doc")
                        if doc_obj:
                            doc_id = getattr(doc_obj, "doc_id", None)
                            
                    if doc_id is not None and doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        if len(seen_docs) <= self.max_docs:
                            fallback_chunks.append(sc)
                    else:
                        if doc_id in seen_docs:
                            fallback_chunks.append(sc)
            except Exception:
                pass
            return fallback_chunks
