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
        self.faiss_index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map: faiss.IndexIDMap2 = faiss.IndexIDMap2(self.faiss_index)
        self.ref = {"me", "i", "myself", "user"}
        
        self.fuzzy_choices: Dict[str, int] = {}
        self.entity_profiles: Dict[int, Dict] = {}

        self._lock = threading.RLock()
    
    def _init_from_db(self):
        logger.info("Hydrating Resolver from Memgraph...")

        temp_profiles = {}
        temp_fuzzy = {}
        vectors = []
        ids = []
        self.ref = {"me", "i", "myself", "user"}
        try:
            store = MemGraphStore()
            
            query = """
            MATCH (n:Entity)
            RETURN n.id, n.canonical_name, n.aliases, n.summary, n.type, n.embedding, n.last_profiled_msg_id
            """
            
            with store.driver.session() as session:
                results = session.run(query)

                for r in results:
                    e_id = r["n.id"]
                    raw_last_msg = r.get("n.last_profiled_msg_id")
                    last_msg_id = raw_last_msg if raw_last_msg is not None else 0

                    
                    temp_profiles[e_id] = {
                        "canonical_name": r["n.canonical_name"],
                        "aliases": r["n.aliases"] if r["n.aliases"] else [],
                        "summary": r["n.summary"],
                        "type": r["n.type"],
                        "last_profiled_msg_id": last_msg_id
                    }
                    
                    names = [r["n.canonical_name"]] + (r["n.aliases"] or [])
                    for n in names:
                        temp_fuzzy[n] = e_id
                    
                    if r["n.embedding"]:
                        vectors.append(r["n.embedding"])
                        ids.append(e_id)
                
            with self._lock:
                self.entity_profiles = temp_profiles
                self.fuzzy_choices = temp_fuzzy

                self.index_id_map.reset()

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
    
    
    def resolve(self, text: str, context: str, fuzzy_cutoff: int = 80):

        with self._lock:
            has_data = bool(self.fuzzy_choices) or self.index_id_map.ntotal > 0
        
        if not has_data:
            return {"resolved": None, "ambiguous": [], "new": True, "mention": text}
        
        query_text = f"{text} mentioned in context of: {context}"
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        set_text = set(word.strip(".,!'?") for word in text.split())
        if any(word in self.ref for word in set_text):
            if "USER" in self.fuzzy_choices:
                _id = self.fuzzy_choices["USER"]
                return {
                    "resolved": {"id": _id, "profile": self.entity_profiles[_id]},
                    "ambiguous": [],
                    "new": False,
                    "mention": text
                }
            
        candidates_map: Dict[int, Dict] = {}
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
                scores, ann = self.index_id_map.search(query_embedding, k=min(10, self.index_id_map.ntotal))

                for index_id, score in zip(ann[0], scores[0]):
                    logger.debug(f"FAISS result: query='{text}' -> id={index_id}, score={score:.3f}")
                    if index_id != -1 and score >= 0.5:
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
        if not candidates_map:
            return {"resolved": None, "ambiguous": [], "new": True, "mention": text}
        
        sorted_candidates = sorted(
        candidates_map.values(),
        key=lambda x: x["match_detail"]["norm_score"],
        reverse=True)

        best_score = sorted_candidates[0]["match_detail"]["norm_score"]
    
        if best_score >= 0.85:
            return {"resolved": sorted_candidates[0], "ambiguous": [], "new": False, "mention": text}
        elif best_score >= 0.5:
            viable = [c for c in sorted_candidates if c["match_detail"]["norm_score"] >= 0.5]
            return {"resolved": None, "ambiguous": viable, "new": False, "mention": text}
        else:
            return {"resolved": None, "ambiguous": [], "new": True, "mention": text}
