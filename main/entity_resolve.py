from datetime import datetime, timezone
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import process as fuzzy_process, fuzz

from graph.memgraph import MemGraphStore

logger = logging.getLogger(__name__)


class EntityResolver:

    def __init__(self, embedding_model_model='google/embeddinggemma-300m', data_dir="./graph_data"):
        self.embedding_model = SentenceTransformer(embedding_model_model)
        self.embedding_dim = 768
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map = faiss.IndexIDMap2(self.faiss_index)
        
        self.fuzzy_choices: Dict[str, int] = {}
        self.entity_profiles: Dict[int, Dict] = {}
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def _init_from_db(self):
        logger.info("Hydrating Resolver from Memgraph...")
        try:
            store = MemGraphStore()
            
            query = """
            MATCH (n:Entity)
            RETURN n.id, n.canonical_name, n.aliases, n.summary, n.type, n.embedding
            """
            
            with store.driver.session() as session:
                results = session.run(query)
                
                vectors = []
                ids = []

                for r in results:
                    e_id = r["n.id"]
                    
                    self.entity_profiles[e_id] = {
                        "canonical_name": r["n.canonical_name"],
                        "aliases": r["n.aliases"] if r["n.aliases"] else [],
                        "summary": r["n.summary"],
                        "type": r["n.type"]
                    }
                    
                    names = [r["n.canonical_name"]] + (r["n.aliases"] or [])
                    for n in names:
                        self.fuzzy_choices[n] = e_id
                    
                    if r["n.embedding"]:
                        vectors.append(r["n.embedding"])
                        ids.append(e_id)
                
                if vectors:
                    logger.info(f"Loading {len(vectors)} vectors into FAISS.")
                    vec_np = np.array(vectors, dtype=np.float32)
                    ids_np = np.array(ids, dtype=np.int64)
                    faiss.normalize_L2(vec_np)
                    self.index_id_map.add_with_ids(vec_np, ids_np)

        except Exception as e:
            logger.error(f"Failed to hydrate from DB: {e}")


    def add_entity(self, entity_id: int, profile: Dict) -> List[float]:

        if entity_id in self.entity_profiles:
            print(f"Entity {entity_id} already exists. Updating.")
            self.update_entity(entity_id, profile)
            return
        
        logger.info(f"Adding entity {entity_id} to resolver indexes.")

        profile.setdefault("topic", "General")
        profile.setdefault("first_seen", datetime.now(timezone.utc).isoformat())
        profile["last_seen"] = datetime.now(timezone.utc).isoformat()


        summary_text = profile.get("summary", "") or "No information available."
        embedding_np = self.embedding_model.encode([summary_text])[0]
        
        faiss.normalize_L2(embedding_np.reshape(1, -1))
        
        self.index_id_map.add_with_ids(
            x=np.array([embedding_np]), 
            xids=np.array([entity_id], dtype=np.int64)
        )

        self.entity_profiles[entity_id] = profile
        for name in [profile["canonical_name"]] + profile["aliases"]:
            self.fuzzy_choices[name] = entity_id
        
        return embedding_np.tolist()
    
    def update_entity(self, entity_id: int, new_profile: Dict) -> List[float]:

        old_profile = self.entity_profiles.get(entity_id, {})

        new_profile["first_seen"] = old_profile.get("first_seen", datetime.now(timezone.utc).isoformat())
        new_profile["last_seen"] = datetime.now(timezone.utc).isoformat()
        
        if old_profile:
            old_aliases = [old_profile.get("canonical_name")] + old_profile.get("aliases", [])
            for old_alias in old_aliases:
                if old_alias in self.fuzzy_choices:
                    del self.fuzzy_choices[old_alias]

        new_aliases = [new_profile["canonical_name"]] + new_profile.get("aliases", [])
        for alias in new_aliases:
            self.fuzzy_choices[alias] = entity_id
        
        self.entity_profiles[entity_id] = new_profile

        try:
            self.index_id_map.remove_ids(np.array([entity_id], dtype=np.int64))
        except Exception:
            pass

        summary_text = new_profile.get("summary", "") or "No information available."
        embedding_np = self.embedding_model.encode([summary_text])[0]
        
        faiss.normalize_L2(embedding_np.reshape(1, -1))
        
        self.index_id_map.add_with_ids(
            x=np.array([embedding_np]), 
            xids=np.array([entity_id], dtype=np.int64)
        ) 

        return embedding_np.tolist()


    def resolve(self, text: str, context: str, top_k: int = 10, fuzzy_cutoff: int = 80):
        
        candidates_map: Dict[int, Any] = {}


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
            query_text = f"{text} mentioned in context of: {context}"
            query_embedding = self.embedding_model.encode([query_text])
            faiss.normalize_L2(query_embedding)

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
