import re
from typing import List, Dict, Any, Set

def is_redundant(words: Set[str], selected_word_sets: List[Set[str]]) -> bool:
    """
    Checks Jaccard word similarity between a word set and previously selected word sets.
    Filters out near-duplicate context (overlap > 65%) to optimize context window space.
    """
    if not words or not selected_word_sets:
        return False
    for sel_words in selected_word_sets:
        if not sel_words:
            continue
        intersection = words.intersection(sel_words)
        union = words.union(sel_words)
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard > 0.65:
            return True
    return False

class Reranker:
    """
    Reranker sorts, deduplicates, and diversifies retrieved chunks.
    Ensures that context pages are unique and multi-document queries return diverse documents.
    """
    def __init__(self):
        pass

    def rerank(self, scored_chunks: List[Dict[str, Any]], query: str, top_n_chunks: int = 6) -> List[Dict[str, Any]]:
        if not scored_chunks:
            return []
            
        # 1. Sort by relevance score, using semantic similarity as secondary and document ID / page as tie-breakers
        scored_chunks.sort(key=lambda x: (
            -x["score"],
            -x["semantic_sim"],
            x["chunk"].doc_id,
            x["chunk"].page
        ))

        # 2. Duplicate Result Reduction: only keep the highest-scoring chunk from the same document and page.
        deduplicated_scored_chunks = []
        seen_pages = set()
        for sc in scored_chunks:
            doc_id = sc["chunk"].doc_id
            page = sc["chunk"].page
            if (doc_id, page) in seen_pages:
                continue
            seen_pages.add((doc_id, page))
            deduplicated_scored_chunks.append(sc)

        # 3. Better Result Diversity
        unique_scored_chunks = []
        seen_texts = set()
        selected_word_sets = []
        
        # Pass 1: Choose at most 1 chunk per unique document for highly relevant documents
        doc_counts_pass1 = {}
        remaining_chunks = []
        
        max_score = deduplicated_scored_chunks[0]["score"] if deduplicated_scored_chunks else 0.0
        
        for sc in deduplicated_scored_chunks:
            doc_id = sc["chunk"].doc_id
            clean_txt = sc["chunk"].text.strip().lower()
            
            # Check text duplication and Jaccard redundancy
            if clean_txt in seen_texts:
                continue
            words = set(w.lower() for w in re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_txt).split() if len(w) > 3)
            if is_redundant(words, selected_word_sets):
                continue
                
            # Highly relevant: score >= 40.0 OR within 30.0 of the maximum score
            is_high = (sc["score"] >= 40.0) or (sc["score"] >= (max_score - 30.0))
            
            if doc_id not in doc_counts_pass1 and is_high:
                doc_counts_pass1[doc_id] = 1
                seen_texts.add(clean_txt)
                selected_word_sets.append(words)
                unique_scored_chunks.append(sc)
            else:
                remaining_chunks.append((sc, clean_txt, words))
                
        # Pass 2: Select additional chunks from remaining_chunks to fill slots up to top_n_chunks
        doc_chunk_counts = {doc_id: 1 for doc_id in doc_counts_pass1}
        
        for sc, clean_txt, words in remaining_chunks:
            if len(unique_scored_chunks) >= top_n_chunks:
                break
                
            doc_id = sc["chunk"].doc_id
            # Limit to at most 1 chunk per document for certificate queries, otherwise 3 chunks
            limit = 1 if "certificate" in query.lower() else 3
            if doc_chunk_counts.get(doc_id, 0) >= limit:
                continue
                
            # Re-check text duplication and Jaccard redundancy for Pass 2 chunks
            if clean_txt in seen_texts:
                continue
            if is_redundant(words, selected_word_sets):
                continue
                
            seen_texts.add(clean_txt)
            selected_word_sets.append(words)
            unique_scored_chunks.append(sc)
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
            
        return unique_scored_chunks
