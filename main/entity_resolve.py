from datetime import datetime, timezone
from loguru import logger
import threading
from typing import Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from graph.memgraph import MemGraphStore
from redisclient import SyncRedisClient


class EntityResolver:

    def __init__(self, embedding_model='dunzhang/stella_en_1.5B_v5'):

 
        self.redis_client = SyncRedisClient().get_client()
        
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True, device='cpu')
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')

        self.embedding_dim = 1024
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map = faiss.IndexIDMap2(self.faiss_index)
        self.entity_profiles = {}
        self._name_to_id = {}
        self._lock = threading.RLock()
    
        self._hydrate_aliases()
        self.store = MemGraphStore()
        self._hydrate_vectors()

    def _hydrate_aliases(self):
        try:
            stored_mentions = self.redis_client.hgetall("entity_mentions")
            with self._lock:
                if stored_mentions:
                    self._name_to_id = {k.decode(): int(v) for k, v in stored_mentions.items()}
                    logger.info(f"Hydrated EntityResolver with {len(self._name_to_id)} aliases from Redis.")
                else:
                    logger.warning("Redis alias cache is empty. Attempting Cold Sync from Memgraph...")
                    
                    if not hasattr(self, 'store'):
                        self.store = MemGraphStore()

                    full_mapping = self.store.get_all_aliases_map()
                    
                    if full_mapping:
                        self._name_to_id = full_mapping
                        
                        try:
                            self.redis_client.hset("entity_mentions", mapping=full_mapping)
                            logger.info(f"Cold Sync Successful: Restored {len(full_mapping)} aliases from Memgraph to Redis.")
                        except Exception as e:
                            logger.error(f"Failed to refill Redis after Cold Sync: {e}")
                    else:
                        logger.info("Memgraph is also empty. Starting with fresh Identity Map.")
                          
        except Exception as e:
            logger.critical(f"Failed to hydrate from Redis: {e}")
    
    def _hydrate_vectors(self):
        try:
            logger.info("Hydrating FAISS vectors from Memgraph...")
            embeddings_map = self.store.get_all_embeddings()
            
            if not embeddings_map:
                logger.info("No vectors found in Memgraph.")
                return

            ids = []
            vectors = []
            
            for eid, vec in embeddings_map.items():
                if len(vec) == self.embedding_dim:
                    ids.append(eid)
                    vectors.append(vec)
            
            if ids:
                self.index_id_map.add_with_ids(
                    np.array(vectors, dtype=np.float32),
                    np.array(ids, dtype=np.int64)
                )
            logger.info(f"Restored {len(ids)} vectors to FAISS.")
            
        except Exception as e:
            logger.error(f"Failed to hydrate vectors: {e}")

    def set_mentions(self, mentions: Dict[str, int]):
        """Set _name_to_id from external source (Redis)."""
        with self._lock:
            self._name_to_id.update(mentions)
    
    def get_mentions(self) -> Dict[str, int]:
        """Get copy of _name_to_id for persistence."""
        with self._lock:
            return self._name_to_id.copy()
    
    def get_id(self, name: str) -> Optional[int]:
        if name in self._name_to_id:
            return self._name_to_id[name]
        
        redis_id = self.redis_client.hget("entity_mentions", name)
        if redis_id:
            entity_id = int(redis_id)
            with self._lock:
                self._name_to_id[name] = entity_id
            return entity_id
            
        return None
    
    def get_mentions_for_id(self, entity_id: int) -> List[str]:
        return [mention for mention, eid in self._name_to_id.items() if eid == entity_id]
    
    def get_embedding_for_id(self, entity_id: int) -> List[float]:
        """Retrieve embedding from FAISS by ID."""
        with self._lock:
            try:
                embedding = self.index_id_map.reconstruct(entity_id)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Could not retrieve embedding for {entity_id}: {e}")
                return []
    
    def validate_existing(self, canonical_name: str, mentions: List[str]) -> Optional[int]:
        """
        Check if canonical_name exists. If yes, register mention aliases and return ID.
        If no, return None (caller handles demotion).
        """
        with self._lock:
            entity_id = self.get_id(canonical_name)
            logger.debug(f"validate_existing: '{canonical_name}' -> id={entity_id}")
            if entity_id is None:
                return None
            
            new_aliases = {}
            for mention in mentions:
                if mention not in self._name_to_id:
                    self._name_to_id[mention] = entity_id
                    new_aliases[mention] = entity_id
            
            if new_aliases:
                try:
                    self.redis_client.hset("entity_mentions", mapping=new_aliases)
                    logger.debug(f"Sent {len(new_aliases)} to Redis store for entity {entity_id}")
                except Exception as e:
                    logger.error(f"Failed to persist new aliases: {e}")

            return entity_id
    
    def register_entity(
        self, 
        entity_id: int, 
        canonical_name: str, 
        mentions: List[str], 
        entity_type: str, 
        topic: str
    ) -> List[float]:
        """
        Register new entity: update all indexes and return embedding.
        """
        profile = {
            "canonical_name": canonical_name,
            "type": entity_type,
            "topic": topic,
            "summary": ""
        }
        
        embedding = self.add_entity(entity_id, profile)
        
        with self._lock:
            self._name_to_id[canonical_name] = entity_id
            for mention in mentions:
                self._name_to_id[mention] = entity_id

            mapping_update = {m: entity_id for m in mentions}
            mapping_update[canonical_name] = entity_id

            try:
                self.redis_client.hset("entity_mentions", mapping=mapping_update)
                logger.debug(f"Persisted {len(mapping_update)} aliases for {canonical_name} to Redis.")
            except Exception as e:
                logger.error(f"CRITICAL: Failed to persist entity mappings to Redis: {e}")

        return embedding

    def add_entity(self, entity_id: int, profile: Dict) -> List[float]:

        canonical_name = profile.get("canonical_name", "")
        summary = profile.get("summary", "") or ""

        resolution_text = f"{canonical_name}. {summary[:200]}"
        embedding_np = self.embedding_model.encode([resolution_text])[0]
        faiss.normalize_L2(embedding_np.reshape(1, -1))


        with self._lock:
            logger.info(f"Adding entity {entity_id}-{profile["canonical_name"]} to resolver indexes.")

            profile.setdefault("topic", "General")
            profile.setdefault("first_seen", datetime.now(timezone.utc).isoformat())
            profile["last_seen"] = datetime.now(timezone.utc).isoformat()
            
            self.index_id_map.add_with_ids(
                np.array([embedding_np]), 
                np.array([entity_id], dtype=np.int64)
            )

            self.entity_profiles[entity_id] = profile
        
        return embedding_np.tolist()
    

    def update_profile_summary(self, entity_id: int, new_summary: str) -> List[float]:
        """
        Update entity summary and recompute embedding.
        Returns new embedding.
        """
        with self._lock:
            profile = self.entity_profiles.get(entity_id)
            if not profile:
                logger.warning(f"Cannot update profile for unknown entity {entity_id}")
                return []
            
            canonical_name = profile.get("canonical_name", "")
            profile["summary"] = new_summary
            profile["last_seen"] = datetime.now(timezone.utc).isoformat()
            
            resolution_text = f"{canonical_name}. {new_summary[:200]}"
            embedding_np = self.embedding_model.encode([resolution_text])[0]
            faiss.normalize_L2(embedding_np.reshape(1, -1))

            self.index_id_map.remove_ids(np.array([entity_id], dtype=np.int64))
            self.index_id_map.add_with_ids(
                np.array([embedding_np]),
                np.array([entity_id], dtype=np.int64)
            )
            
            return embedding_np.tolist()

    def detect_merge_candidates(self) -> list:
        """Detect potential entity merges based on summary similarity."""

        logger.info(f"Merge detection started, {len(self.entity_profiles)} entities to scan")
        
        candidates = []
        seen_pairs = set()
        
        for entity_id, profile in self.entity_profiles.items():
            summary = profile.get("summary", "")
            entity_type = profile.get("type")
            canonical_name = profile.get("canonical_name", "Unknown")
            if not summary:
                continue
            
            import re
            match = re.match(r'^(.+?[.!?])(?:\s+[A-Z]|$)', summary)
            first_sentence = match.group(1).strip() if match else summary[:200].strip()
            
            query_text = f"{canonical_name}. {first_sentence}"
            embedding = self.embedding_model.encode([query_text])
            faiss.normalize_L2(embedding)
            
            with self._lock:
                if self.index_id_map.ntotal == 0:
                    continue
                
                scores, indices = self.index_id_map.search(embedding, k=5)
            
            for match_id, faiss_score in zip(indices[0], scores[0]):
                match_id = int(match_id)
                
                if match_id == -1 or match_id == entity_id:
                    continue
                
                match_profile = self.entity_profiles.get(match_id)
                if not match_profile:
                    continue

                match_type = match_profile.get("type")
                match_name = match_profile.get("canonical_name", "Unknown")
        
                if entity_type != match_type:
                    logger.debug(f"Type mismatch: {canonical_name} ({entity_type}) vs {match_name} ({match_type}), skipping")
                    continue

                if canonical_name == match_name:
                    pair = tuple(sorted([entity_id, match_id]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    
                    logger.info(f"Merge verification: ({entity_id}, {match_id}) {canonical_name} <- {match_name} | "
                                f"Exact name match Decision=APPROVED")
                    primary_id, secondary_id = pair
                    candidates.append({
                        "primary_id": primary_id,
                        "secondary_id": secondary_id,
                        "primary_name": canonical_name,
                        "secondary_name": match_name,
                        "faiss_score": float(faiss_score),
                        "cross_score": 1.0  # Exact match
                    })
                    continue

                if faiss_score < 0.50:
                    continue

                primary_text = f"{canonical_name}. {profile.get('summary', '')[:150]}"
                secondary_text = f"{match_name}. {match_profile.get('summary', '')[:150]}"
                cross_score = float(self.cross_encoder.predict([[primary_text, secondary_text]])[0])
                
                pair = tuple(sorted([entity_id, match_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                decision = "APPROVED" if cross_score >= 0.65 else "REJECTED"
                logger.info(
                    f"Merge verification: ({entity_id}, {match_id}) {canonical_name} <- {match_name} | "
                    f"FAISS={faiss_score:.3f} CrossEncoder={cross_score:.3f} Decision={decision}"
                )
                
                if cross_score >= 0.65:
                    primary_id, secondary_id = pair
                    candidates.append({
                        "primary_id": primary_id,
                        "secondary_id": secondary_id,
                        "primary_name": self.entity_profiles[primary_id].get("canonical_name", "Unknown"),
                        "secondary_name": self.entity_profiles[secondary_id].get("canonical_name", "Unknown"),
                        "faiss_score": float(faiss_score),
                        "cross_score": cross_score
                    })
        
        logger.info(f"Merge detection complete: {len(candidates)} candidates found")
        
        return candidates