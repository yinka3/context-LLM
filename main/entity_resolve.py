from datetime import datetime, timezone
from loguru import logger
import threading
from typing import Dict, List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import process as fuzzy_process, fuzz
from graph.memgraph import MemGraphStore



class EntityResolver:

    def __init__(self, embedding_model_model='google/embeddinggemma-300m'):
        self.embedding_model = SentenceTransformer(embedding_model_model)
        self.embedding_dim = 768
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map = faiss.IndexIDMap2(self.faiss_index)
        
        self.fuzzy_choices: Dict[str, int] = {}
        self.entity_profiles: Dict[int, Dict] = {}

        self._lock = threading.RLock()
    
    def _init_from_db(self):
        logger.info("Hydrating Resolver from Memgraph...")

        temp_profiles = {}
        temp_fuzzy = {}
        vectors = []
        ids = []

        try:
            store = MemGraphStore()
            
            query = """
            MATCH (n:Entity)
            RETURN n.id, n.canonical_name, n.aliases, n.summary, n.type, n.embedding
            """
            
            with store.driver.session() as session:
                results = session.run(query)

                for r in results:
                    e_id = r["n.id"]
                    
                    temp_profiles[e_id] = {
                        "canonical_name": r["n.canonical_name"],
                        "aliases": r["n.aliases"] if r["n.aliases"] else [],
                        "summary": r["n.summary"],
                        "type": r["n.type"]
                    }
                    
                    names = [r["n.canonical_name"]] + (r["n.aliases"] or [])
                    for n in names:
                        temp_fuzzy[n] = e_id
                    
                    if r["n.embedding"]:
                        vectors.append(r["n.embedding"])
                        ids.append(e_id)
                
            with self._lock:
                self.entity_profiles.update(temp_profiles)
                self.fuzzy_choices.update(temp_fuzzy)

                if vectors:
                    logger.info(f"Loading {len(vectors)} vectors into FAISS.")
                    vec_np = np.array(vectors, dtype=np.float32)
                    ids_np = np.array(ids, dtype=np.int64)
                    faiss.normalize_L2(vec_np)
                    self.index_id_map.add_with_ids(vec_np, ids_np)

            return True
        except Exception as e:
            logger.error(f"Failed to hydrate from DB: {e}")
            return False

    def add_alias(self, entity_id: int, alias: str):
        with self._lock:
            if entity_id in self.entity_profiles:
                profile = self.entity_profiles[entity_id]
                if alias not in profile.get("aliases", []):
                    profile.setdefault("aliases", []).append(alias)
                    self.fuzzy_choices[alias] = entity_id

    def add_entity(self, entity_id: int, profile: Dict) -> List[float]:

        summary_text = profile.get("summary", "") or "No information available."
        embedding_np = self.embedding_model.encode([summary_text])[0]
        faiss.normalize_L2(embedding_np.reshape(1, -1))


        with self._lock:
            if entity_id in self.entity_profiles:
                logger.info(f"Entity {entity_id} already exists. Updating.")
                return self._update_entity_inner(entity_id, profile, embedding_np)
            
            logger.info(f"Adding entity {entity_id} to resolver indexes.")

            profile.setdefault("topic", "General")
            profile.setdefault("first_seen", datetime.now(timezone.utc).isoformat())
            profile["last_seen"] = datetime.now(timezone.utc).isoformat()
            
            self.index_id_map.add_with_ids(
                np.array([embedding_np]), 
                np.array([entity_id], dtype=np.int64)
            )

            self.entity_profiles[entity_id] = profile
            for name in [profile["canonical_name"]] + profile["aliases"]:
                self.fuzzy_choices[name] = entity_id
        
        return embedding_np.tolist()
    
    def update_entity(self, entity_id: int, new_profile: Dict) -> List[float]:

        summary_text = new_profile.get("summary", "") or "No information available."
        embedding_np = self.embedding_model.encode([summary_text])[0]
        faiss.normalize_L2(embedding_np.reshape(1, -1))

        with self._lock:
            return self._update_entity_inner(entity_id, new_profile, embedding_np)
    
    def _update_entity_inner(self, entity_id: int, new_profile: Dict, embedding_np) -> List[float]:
        """Assumes lock is already held."""
        old_profile = self.entity_profiles.get(entity_id, {})

        new_profile["first_seen"] = old_profile.get("first_seen", datetime.now(timezone.utc).isoformat())
        new_profile["last_seen"] = datetime.now(timezone.utc).isoformat()
        
        if old_profile:
            old_aliases = [old_profile.get("canonical_name")] + old_profile.get("aliases", [])
            for old_alias in old_aliases:
                if self.fuzzy_choices.get(old_alias) == entity_id:
                    del self.fuzzy_choices[old_alias]

        new_aliases = [new_profile["canonical_name"]] + new_profile.get("aliases", [])
        for alias in new_aliases:
            self.fuzzy_choices[alias] = entity_id
        
        self.entity_profiles[entity_id] = new_profile

        try:
            self.index_id_map.remove_ids(np.array([entity_id], dtype=np.int64))
        except Exception:
            pass
        
        self.index_id_map.add_with_ids(
            np.array([embedding_np]), 
            np.array([entity_id], dtype=np.int64)
        )

        return embedding_np.tolist()


    def resolve(self, text: str, context: str, top_k: int = 10, fuzzy_cutoff: int = 80):

        with self._lock:
            has_data = bool(self.fuzzy_choices) or self.index_id_map.ntotal > 0
        
        if not has_data:
            return []
        
        query_text = f"{text} mentioned in context of: {context}"
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)

        candidates_map: Dict[int, str | Dict] = {}

        with self._lock:
            if self.fuzzy_choices:
                fuzzy_matches = fuzzy_process.extract(query=text, choices=self.fuzzy_choices.keys(),
                                                    scorer=fuzz.WRatio, score_cutoff=fuzzy_cutoff, limit=10)
                for match_name, score, _ in fuzzy_matches:
                    entity_id = self.fuzzy_choices[match_name]
                    norm_score = score / 100.0

                    candidates_map[entity_id] = {
                        "id": entity_id,
                        "match_detail" : {
                            "source": "fuzzy",
                            "score": score,
                            "norm_score": norm_score,
                            "matched_aliases": match_name
                        },
                        "profile": self.entity_profiles.get(entity_id, {})
                    }
            
            if self.index_id_map.ntotal > 0:

                search_k = min(top_k, self.faiss_index.ntotal)
                scores, ann = self.index_id_map.search(query_embedding, k=search_k)

                for index_id, score in zip(ann[0], scores[0]):
                    if index_id != -1:
                        entity_id = int(index_id)
                        norm_score = float((score + 1) / 2)

                        exisiting = candidates_map.get(entity_id)

                        if exisiting:
                            exisiting["match_detail"]["source"] = "hybrid"
                            exisiting["match_detail"]["vector_store"] = float(score)

                            if norm_score > exisiting["match_detail"]["norm_score"]:
                                exisiting["match_detail"]["norm_score"] = norm_score
                        else:
                            candidates_map[entity_id] = {
                                "id": entity_id,
                                "match_detail": {
                                    "source": "faiss",
                                    "score": float(score),
                                    "norm_score": norm_score
                                },
                                "profile": self.entity_profiles[entity_id]
                            }

        candidates = list(candidates_map.values())
        candidates.sort(key=lambda x: x["match_detail"]["norm_score"], reverse=True)

        return candidates[:top_k]
