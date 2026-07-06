import random
import threading
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) index for fast approximate nearest neighbor search.
    Implements a multi-layered graph where upper layers have sparser connections and lower layers
    have dense connections.
    """
    def __init__(self, dim: int = 384, M: int = 16, ef_construction: int = 64, ef_search: int = 32):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.level_mult = 1 / np.log(M)
        
        self.enter_node = None
        self.max_level = -1
        
        self.node_levels = {}
        self.node_vectors = {}
        self.node_metadata = {}
        # Graphs mapping level -> node_id -> list of neighbor node_ids
        self.graphs = {}
        
        # Thread safety lock for graphs updates
        self.lock = threading.Lock()

    def _get_random_level(self) -> int:
        """Returns a random layer level using an exponential decay distribution."""
        # Ensure we always get a level >= 0
        r = random.random()
        if r == 0:
            r = 0.0001
        return int(-np.log(r) * self.level_mult)

    def add_item(self, node_id: Any, vector: List[float], metadata: Dict[str, Any] = None):
        """Adds a new item to the HNSW index, building links on appropriate levels."""
        with self.lock:
            # Ensure vector dimensions match dim
            if len(vector) != self.dim:
                if len(vector) < self.dim:
                    vector = vector + [0.0] * (self.dim - len(vector))
                else:
                    vector = vector[:self.dim]
                    
            self.node_vectors[node_id] = vector
            self.node_metadata[node_id] = metadata or {}
            
            # Select random insertion level
            level = self._get_random_level()
            self.node_levels[node_id] = level
            
            # First element initialization
            if self.enter_node is None:
                self.enter_node = node_id
                self.max_level = level
                for l in range(level + 1):
                    self.graphs[l] = {node_id: []}
                return
                
            curr_node = self.enter_node
            query_vec = vector
            
            # 1. Find entry point for the insertion level by searching upper layers
            for l in range(self.max_level, level, -1):
                candidates = self._search_layer(query_vec, curr_node, 1, l)
                if candidates:
                    curr_node = candidates[0][1]
                    
            # 2. Insert and connect at level down to 0
            for l in range(min(level, self.max_level), -1, -1):
                if l not in self.graphs:
                    self.graphs[l] = {}
                self.graphs[l][node_id] = []
                
                # Find construction-phase candidates at this layer
                candidates = self._search_layer(query_vec, curr_node, self.ef_construction, l)
                if not candidates:
                    continue
                    
                # Select M nearest neighbors
                neighbors = candidates[:self.M]
                for dist, neighbor_id in neighbors:
                    # Form bidirectional links
                    self.graphs[l][node_id].append(neighbor_id)
                    self.graphs[l][neighbor_id].append(node_id)
                    
                    # Shrink connections of the neighbor if connection limit exceeded
                    max_conn = self.M0 if l == 0 else self.M
                    if len(self.graphs[l][neighbor_id]) > max_conn:
                        nb_vec = self.node_vectors[neighbor_id]
                        nb_neighbors = self.graphs[l][neighbor_id]
                        # Sort by cosine similarity descending
                        nb_neighbors.sort(key=lambda x: -cosine_similarity(nb_vec, self.node_vectors[x]))
                        self.graphs[l][neighbor_id] = nb_neighbors[:max_conn]
                        
                curr_node = neighbors[0][1]
                
            # 3. Handle level higher than max_level
            if level > self.max_level:
                for l in range(self.max_level + 1, level + 1):
                    self.graphs[l] = {node_id: []}
                    if self.enter_node is not None:
                        self.graphs[l][self.enter_node] = []
                        self.graphs[l][node_id].append(self.enter_node)
                        self.graphs[l][self.enter_node].append(node_id)
                self.enter_node = node_id
                self.max_level = level

    def _search_layer(self, query_vec: List[float], enter_node: Any, ef: int, level: int) -> List[Tuple[float, Any]]:
        """Searches a single layer graph for ef nearest neighbors to the query vector."""
        if enter_node not in self.node_vectors:
            return []
            
        # Keep track of visited nodes to avoid cyclic loops
        visited = {enter_node}
        
        # Calculate entry node similarity
        entry_sim = cosine_similarity(query_vec, self.node_vectors[enter_node])
        candidates = [(entry_sim, enter_node)]
        results = [(entry_sim, enter_node)]
        
        while candidates:
            # Sort candidates by similarity descending to examine the closest neighbor
            candidates.sort(key=lambda x: -x[0])
            curr_sim, curr_node = candidates.pop(0)
            
            # Find the worst result (lowest similarity) currently stored
            results.sort(key=lambda x: -x[0])
            worst_result_sim = results[-1][0]
            
            # Prune search early if current candidate is worse than worst stored result
            if curr_sim < worst_result_sim and len(results) >= ef:
                break
                
            # Travel along neighbors in graph layer
            neighbors = self.graphs.get(level, {}).get(curr_node, [])
            for neighbor in neighbors:
                if neighbor not in self.node_vectors:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    neighbor_sim = cosine_similarity(query_vec, self.node_vectors[neighbor])
                    results.sort(key=lambda x: -x[0])
                    worst_result_sim = results[-1][0]
                    
                    if neighbor_sim > worst_result_sim or len(results) < ef:
                        candidates.append((neighbor_sim, neighbor))
                        results.append((neighbor_sim, neighbor))
                        
                        # Keep size of results within ef
                        if len(results) > ef:
                            results.sort(key=lambda x: -x[0])
                            results.pop()
                            
        results.sort(key=lambda x: -x[0])
        return results

    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[float, Any]]:
        """Queries the HNSW index for the top k closest vectors using cosine similarity."""
        with self.lock:
            if self.enter_node is None:
                return []
                
            curr_node = self.enter_node
            # Search down to layer 1 greedily
            for l in range(self.max_level, 0, -1):
                res = self._search_layer(query_vec, curr_node, 1, l)
                if res:
                    curr_node = res[0][1]
                    
            # Search layer 0 with full search depth (ef_search)
            results = self._search_layer(query_vec, curr_node, max(self.ef_search, k), 0)
            return results[:k]
