import time
import numpy as np

class GalleryShard:
    def __init__(self, shard_id, max_size, dim):
        self.shard_id = shard_id
        self.max_size = max_size
        self.dim = dim
        self.embeddings = np.zeros((max_size, dim), dtype=np.float32)
        self.global_ids = np.zeros(max_size, dtype=np.int64)
        self.timestamps = np.zeros(max_size, dtype=np.float64)
        self.size = 0
        self.max_timestamp = 0
        self.min_timestamp = float('inf')

    def add(self, global_id, embedding, timestamp):
        if self.size >= self.max_size:
            return False # Shard full
        self.embeddings[self.size] = embedding
        self.global_ids[self.size] = global_id
        self.timestamps[self.size] = timestamp
        self.max_timestamp = max(self.max_timestamp, timestamp)
        self.min_timestamp = min(self.min_timestamp, timestamp)
        self.size += 1
        return True

class ScalableGalleryService:
    def __init__(self, dim=256, shard_capacity=10000, retention_seconds=86400):
        self.dim = dim
        self.shard_capacity = shard_capacity
        self.retention_seconds = retention_seconds
        
        self.shards = []
        self.next_shard_id = 1
        self._current_shard = self._create_new_shard()
        self.shards.append(self._current_shard)
        
    def _create_new_shard(self):
        shard = GalleryShard(shard_id=self.next_shard_id, 
                             max_size=self.shard_capacity, 
                             dim=self.dim)
        self.next_shard_id += 1
        return shard

    def enrol(self, global_id, embedding, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        if not self._current_shard.add(global_id, embedding, timestamp):
            # Shard is full, rotate to a new shard
            self._current_shard = self._create_new_shard()
            self.shards.append(self._current_shard)
            self._current_shard.add(global_id, embedding, timestamp)

    def query(self, query_embedding, top_k=5, probe_fraction=1.0, rng=None):
        """
        probe_fraction < 1.0 turns this into an ANN-style approximate search:
        instead of scanning every shard (exact, probe_fraction=1.0), only a
        random sample of shards is scanned. This is the honest, minimal way
        to get a real recall/latency/memory tradeoff out of a sharded index
        without depending on an external ANN library (HNSW/FAISS) that isn't
        installed in this environment -- exact search has no recall to trade
        away, so this parameter exists specifically to produce a measurable
        tradeoff for benchmark.py's recall-vs-latency-vs-size sweep.
        """
        query_embedding = np.array(query_embedding).flatten().astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding /= norm

        shards_to_search = self.shards
        if probe_fraction < 1.0 and len(self.shards) > 1:
            rng = rng or np.random
            n_probe = max(1, int(round(len(self.shards) * probe_fraction)))
            shards_to_search = list(rng.choice(self.shards, size=n_probe, replace=False))

        best_scores = []
        best_ids = []

        # Distribute search across the probed shards (embarrassingly parallelizable)
        for shard in shards_to_search:
            if shard.size == 0:
                continue
            
            scores = np.dot(shard.embeddings[:shard.size], query_embedding)
            
            if shard.size < top_k:
                top_idx = np.argsort(scores)[::-1]
            else:
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                
            best_scores.extend(scores[top_idx])
            best_ids.extend(shard.global_ids[top_idx])
            
        if not best_scores:
            return []
            
        best_scores = np.array(best_scores)
        best_ids = np.array(best_ids)
        
        if len(best_scores) <= top_k:
            final_idx = np.argsort(best_scores)[::-1]
        else:
            final_idx = np.argpartition(best_scores, -top_k)[-top_k:]
            final_idx = final_idx[np.argsort(best_scores[final_idx])[::-1]]
            
        results = []
        for idx in final_idx:
            results.append({
                "global_id": int(best_ids[idx]),
                "similarity": float(best_scores[idx])
            })
            
        return results

    def evict(self, current_time=None):
        """Evicts entire shards if their maximum timestamp is older than retention period. O(1) row deletions!"""
        if current_time is None:
            current_time = time.time()
            
        cutoff = current_time - self.retention_seconds
        active_shards = []
        evicted_count = 0
        evicted_records = 0
        
        for shard in self.shards:
            if shard == self._current_shard:
                active_shards.append(shard)
                continue
                
            if shard.max_timestamp < cutoff:
                evicted_count += 1
                evicted_records += shard.size
            else:
                active_shards.append(shard)
                
        self.shards = active_shards
        return evicted_count, evicted_records

    def get_stats(self):
        total_records = sum(s.size for s in self.shards)
        return {
            "num_shards": len(self.shards),
            "total_records": total_records,
            "memory_mb": (total_records * self.dim * 4) / (1024 * 1024)
        }
