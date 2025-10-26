from typing import Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from thefuzz import process as fuzzy_process

class EntityResolver:

    def __init__(self, embedding_model_model='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_model)
        self.embedding_dim = 384 
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map = faiss.IndexIDMap2(self.faiss_index)
        self.next_id = 0
        self.f_index_to_id: Dict[int, str] = {}
        self.fuzzy_choices: Dict[str, str] = {}
        self.entity_profiles: Dict[str, Dict] = {}
        self.id_to_f_index: Dict[str, int] = {}
    
    def get_next_id(self):
        self.next_id += 1
        return self.next_id

    def add_entity(self, entity_id: str, profile: Dict):
        
        print(f"Adding entity {entity_id} to resolver indexes.")
        self.entity_profiles[entity_id] = profile
        new_id = self.get_next_id()

        for name in [profile["canonical_name"]] + profile["aliases"]:
            self.fuzzy_choices[name] = entity_id
        
        self.f_index_to_id[new_id] = entity_id
        self.id_to_f_index[entity_id] = new_id
        
        profile_embedding = self.embedding_model.encode([profile["summary"]])
        faiss.normalize_L2(profile_embedding)
        self.index_id_map.add_with_ids(x=profile_embedding, xids=np.array([new_id], dtype=np.int64))    
        
    
    def update_entity(self, entity_id: str, new_profile: Dict):

        new_aliases = new_profile["aliases"]
        old_aliases = self.entity_profiles[entity_id]["aliases"]

        for old_alias in old_aliases:
            if old_alias in self.fuzzy_choices:
                del self.fuzzy_choices[old_alias]

        for alias in new_aliases:
            self.fuzzy_choices[alias] = entity_id
        
        self.entity_profiles[entity_id] = new_profile

        faiss_index = self.id_to_f_index[entity_id]
        self.index_id_map.remove_ids(np.array([faiss_index], dtype=np.int64))
        profile_embedding = self.embedding_model.encode([new_profile["summary"]])
        faiss.normalize_L2(profile_embedding)
        self.index_id_map.add_with_ids(x=profile_embedding, xids=np.array([faiss_index], dtype=np.int64))


    def resolve(self, text: str, context: str, top_k: int = 10, fuzzy_cutoff: int = 80):
        
        candidate_ids = set()

        if self.fuzzy_choices:
            fuzzy_matches = fuzzy_process.extractBests(query=text, choices=self.fuzzy_choices.keys(), 
                                                       score_cutoff=fuzzy_cutoff, limit=10)
            for match, score in fuzzy_matches:
                candidate_ids.add((self.fuzzy_choices[match], score, score / 100.00,"fuzzy"))
        
        if self.faiss_index.ntotal > 0:
            query_text = f"{text} mentioned in context of: {context}"
            query_embedding = self.embedding_model.encode([query_text])
            faiss.normalize_L2(query_embedding)
            scores, ann = self.index_id_map.search(query_embedding, k=min(top_k, self.faiss_index.ntotal))

            for index_pos, score in zip(ann[0], scores[0]):
                if index_pos != -1:
                    candidate_ids.add((self.f_index_to_id[index_pos], score, (score + 1) / 2),"faiss")
        
        candidates = []
        for entity_id, score, norm, type in candidate_ids:
            candidates.append({
                "id": entity_id,
                "match_detail": {
                    "source": type,
                    "score": score,
                    "norm_score": norm
                },
                "profile": self.entity_profiles[entity_id]
            })

        return candidates
