import logging
from pathlib import Path
import pickle
from typing import Any, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import process as fuzzy_process, fuzz

logger = logging.getLogger(__name__)


class EntityResolver:

    def __init__(self, embedding_model_model='all-MiniLM-L6-v2', data_dir="./graph_data"):
        self.embedding_model = SentenceTransformer(embedding_model_model)
        self.embedding_dim = 384
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map = faiss.IndexIDMap2(self.faiss_index)
        
        self.fuzzy_choices: Dict[str, int] = {} # Alias -> Entity ID
        self.entity_profiles: Dict[int, Dict] = {} # Entity ID -> Profile
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    

    def save(self):
        try:
            faiss.write_index(self.index_id_map, str(self.data_dir / "resolver.index"))
            with open(self.data_dir / "resolver_data.pkl", "wb") as f:
                pickle.dump({
                    "profiles": self.entity_profiles,
                    "fuzzy": self.fuzzy_choices
                }, f)
        except Exception as e:
            logger.warning(f"Error saving to disk: {e}")
    
    def load(self):
        index_path = self.data_dir / "resolver.index"
        data_path = self.data_dir / "resolver_data.pkl"

        if index_path.exists() and data_path.exists():

            try:
                self.index_id_map = faiss.read_index(str(index_path))

                with open(data_path, "rb") as f:

                    data = pickle.load(f)
                    self.entity_profiles = data["fuzzy"]
                
                logger.info("Resolver state loaded")
            except Exception as e:
                logger.info(f"Failed to load resolver state: {e}")


    def add_entity(self, entity_id: int, profile: Dict):

        if entity_id in self.entity_profiles:
            print(f"Entity {entity_id} already exists. Updating.")
            self.update_entity(entity_id, profile)
            return
        
        logger.info(f"Adding entity {entity_id} to resolver indexes.")
        self.entity_profiles[entity_id] = profile
        

        for name in [profile["canonical_name"]] + profile["aliases"]:
            self.fuzzy_choices[name] = entity_id
        
        if "summary" in profile and profile["summary"]:
            profile_emd = self.embedding_model.encode([profile["summary"]])
            faiss.normalize_L2(profile_emd)
            self.index_id_map.add_with_ids(x=profile_emd, xids=np.array([entity_id], dtype=np.int64))    
        
    
    def update_entity(self, entity_id: str, new_profile: Dict):

        old_profile = self.entity_profiles.get(entity_id)
        
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

        if "summary" in new_profile and new_profile["summary"]:
            profile_emd = self.embedding_model.encode([new_profile["summary"]])
            faiss.normalize_L2(profile_emd)
            self.index_id_map.add_with_ids(x=profile_emd, xids=np.array([entity_id], dtype=np.int64))   


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
                    }
                }
        
        if self.faiss_index.ntotal > 0:
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
