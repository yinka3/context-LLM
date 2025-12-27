from datetime import datetime, timezone
from loguru import logger
import threading
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from db.memgraph import MemGraphStore



class EntityResolver:

    def __init__(self, store: 'MemGraphStore', embedding_model='dunzhang/stella_en_1.5B_v5'):
        self.store = store
        
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True, device='cpu')

        self.embedding_dim = 1024
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map = faiss.IndexIDMap2(self.faiss_index)
        self.entity_profiles = {}
        self._name_to_id = {}
        self._lock = threading.RLock()
    
        self._hydrate_from_store()

    def _hydrate_from_store(self):
        """Populate all resolver structures from Memgraph."""
        try:
            entities = self.store.get_all_entities_for_hydration()
            
            if not entities:
                logger.info("No entities in Memgraph. Starting fresh.")
                return
            
            ids = []
            vectors = []
            
            with self._lock:
                for ent in entities:
                    ent_id = ent["id"]
                    canonical = ent["canonical_name"]
                    aliases = ent["aliases"] or []
                    embedding = ent["embedding"]
                    
                    self._name_to_id[canonical.lower()] = ent_id
                    for alias in aliases:
                        self._name_to_id[alias.lower()] = ent_id
                    
                    self.entity_profiles[ent_id] = {
                        "canonical_name": canonical,
                        "type": ent["type"],
                        "topic": ent["topic"] or "General",
                        "summary": ent["summary"] or ""
                    }
                    
                    if embedding and len(embedding) == self.embedding_dim:
                        ids.append(ent_id)
                        vectors.append(embedding)
                
                if ids:
                    self.index_id_map.add_with_ids(
                        np.array(vectors, dtype=np.float32),
                        np.array(ids, dtype=np.int64)
                    )
            
            logger.info(f"Hydrated {len(self.entity_profiles)} entities, {len(ids)} vectors from Memgraph")
            
        except Exception as e:
            logger.error(f"Hydration failed: {e}")
            raise

    def get_mentions(self) -> Dict[str, int]:
        """Get copy of _name_to_id for persistence."""
        with self._lock:
            return self._name_to_id.copy()
    
    def get_id(self, name: str) -> Optional[int]:
        return self._name_to_id.get(name.lower())
    
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
    
    def validate_existing(self, canonical_name: str, mentions: List[str]) -> Tuple[Optional[int], bool]:
        """
        Check if canonical_name exists. If yes, register mention aliases and return ID.
        If no, return None (caller handles demotion).
        """
        with self._lock:
            entity_id = self.get_id(canonical_name)
            logger.debug(f"validate_existing: '{canonical_name}' -> id={entity_id}")
            if entity_id is None:
                return None, False
            
            new_aliases = {}
            for mention in mentions:
                if mention.lower() not in self._name_to_id:
                    self._name_to_id[mention.lower()] = entity_id
                    new_aliases[mention] = entity_id

            return entity_id, len(new_aliases) > 0
    
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
            self._name_to_id[canonical_name.lower()] = entity_id
            for mention in mentions:
                self._name_to_id[mention.lower()] = entity_id

        return embedding

    def add_entity(self, entity_id: int, profile: Dict) -> List[float]:

        canonical_name = profile.get("canonical_name", "")
        summary = profile.get("summary", "") or ""

        resolution_text = f"{canonical_name}. {summary}"
        embedding_np = self.embedding_model.encode([resolution_text])[0]
        faiss.normalize_L2(embedding_np.reshape(1, -1))


        with self._lock:

            #TODO: eventually need to make a better LRU system
            if len(self.entity_profiles) >= 10000:
                oldest_id = next(iter(self.entity_profiles))
                del self.entity_profiles[oldest_id]
                
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
        """Detect potential entity merges using name matching and summary similarity."""
        
        logger.info(f"Merge detection started, {len(self.entity_profiles)} entities to scan")
        
        candidates = []
        seen_pairs = set()
        aliases = list(self._name_to_id.keys())
        for i in range(len(aliases)):
            for j in range(i + 1, len(aliases)):
                id_i = self._name_to_id[aliases[i]]
                id_j = self._name_to_id[aliases[j]]
                if id_i == id_j:
                    continue
                score = fuzz.WRatio(aliases[i], aliases[j])
                if score >= 85:
                    seen_pairs.add(tuple(sorted([id_i, id_j])))
                    

        
        for id_a, id_b in seen_pairs:
            if self.store.has_direct_edge(id_a, id_b):
                logger.debug(f"Blocked ({id_a}, {id_b}) | Direct edge exists")
                continue
            
            neighbors_a = self.store.get_neighbor_ids(id_a)
            neighbors_b = self.store.get_neighbor_ids(id_b)
            neighbors_a.discard(1) # user id - hardcoded for now
            neighbors_b.discard(1)
            
            if neighbors_a & neighbors_b:
                logger.debug(f"Blocked ({id_a}, {id_b}) | Shared neighbors: {neighbors_a & neighbors_b}")
                continue
            
            # Passed filtering - add to candidates
            profile_a = self.entity_profiles.get(id_a, {})
            profile_b = self.entity_profiles.get(id_b, {})
            
            candidates.append({
                "primary_id": id_a,
                "secondary_id": id_b,
                "primary_name": profile_a.get("canonical_name", "Unknown"),
                "secondary_name": profile_b.get("canonical_name", "Unknown"),
                "profile_a": profile_a,
                "profile_b": profile_b,
            })
            
            logger.info(f"Candidate ({id_a}, {id_b}) {profile_a.get('canonical_name')} <-> {profile_b.get('canonical_name')} | Passed graph filtering")
    
        logger.info(f"Merge detection complete: {len(candidates)} candidates found")
        return candidates