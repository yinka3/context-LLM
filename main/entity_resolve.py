from datetime import datetime, timezone
from loguru import logger
import threading
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz
import faiss
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from db.memgraph import MemGraphStore



class EntityResolver:

    def __init__(self, store: 'MemGraphStore', embedding_model='dunzhang/stella_en_1.5B_v5'):
        self.store = store
        
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True, device='cpu')
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')

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

        resolution_text = f"{canonical_name}. {summary[:200]}"
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
        
        # phase 1 to gather candidate pairs
        name_to_ids: dict[str, list[int]] = {}
        for ent_id, profile in self.entity_profiles.items():
            canonical = profile.get("canonical_name", "")
            if canonical:
                key = canonical.lower().strip()
                name_to_ids.setdefault(key, []).append(ent_id)
        
        for ids in name_to_ids.values():
            if len(ids) < 2:
                continue
            for i, id_a in enumerate(ids):
                for id_b in ids[i + 1:]:
                    seen_pairs.add(tuple(sorted([id_a, id_b])))
        
        # Summary-based detection
        for entity_id, profile in self.entity_profiles.items():
            summary = profile.get("summary", "")
            if not summary:
                continue
            
            canonical_name = profile.get("canonical_name", "Unknown")
            match = re.match(r'^(.+?[.!?])(?:\s+[A-Z]|$)', summary)
            first_sentence = match.group(1).strip() if match else summary[:200].strip()
            
            query_text = f"{canonical_name}. {first_sentence}"
            embedding = self.embedding_model.encode([query_text])
            faiss.normalize_L2(embedding)
            
            with self._lock:
                if self.index_id_map.ntotal == 0:
                    continue
                scores, indices = self.index_id_map.search(embedding, k=5)
            
            for match_id, score in zip(indices[0], scores[0]):
                match_id = int(match_id)
                if match_id == -1 or match_id == entity_id:
                    continue
                if score >= 0.50:
                    seen_pairs.add(tuple(sorted([entity_id, match_id])))
        
        for id_a, id_b in seen_pairs:
            profile_a = self.entity_profiles.get(id_a, {})
            profile_b = self.entity_profiles.get(id_b, {})
            
            summary_a = profile_a.get("summary", "")
            summary_b = profile_b.get("summary", "")
            
            if not summary_a or not summary_b:
                logger.debug(f"Skipping ({id_a}, {id_b}): empty summary")
                continue
            
            name_a = profile_a.get("canonical_name", "Unknown")
            name_b = profile_b.get("canonical_name", "Unknown")
            
            text_a = f"{name_a}. {summary_a[:150]}"
            text_b = f"{name_b}. {summary_b[:150]}"
            cross_score = float(self.cross_encoder.predict([[text_a, text_b]])[0])
            
            if cross_score < 0.65:
                logger.info(f"Rejected ({id_a}, {id_b}) {name_a} <-> {name_b} | Cross={cross_score:.3f}")
                continue
            
            # Relationship check
            if self.store.has_direct_edge(id_a, id_b):
                logger.info(f"Blocked ({id_a}, {id_b}) {name_a} <-> {name_b} | Direct edge exists")
                continue
            
            if cross_score >= 0.93:
                logger.info(f"Candidate ({id_a}, {id_b}) {name_a} <-> {name_b} | Cross={cross_score:.3f} → AUTO")
            else:
                logger.info(f"Candidate ({id_a}, {id_b}) {name_a} <-> {name_b} | Cross={cross_score:.3f} → HITL")

            candidates.append({
                "primary_id": id_a,
                "secondary_id": id_b,
                "primary_name": name_a,
                "secondary_name": name_b,
                "cross_score": cross_score
            })
        
        logger.info(f"Merge detection complete: {len(candidates)} candidates found")
        return candidates