import faiss
from sentence_transformers import SentenceTransformer
from thefuzz import process as fuzzy_process

class EntityResolver:

    def __init__(self, embedding_model_model='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_model)
        self.embedding_dim = 384 
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.index_to_id = {}
        self.fuzzy_choices = {}
        self.entity_profiles = {}
    

    def add_entity(self, entity_id: str, names: list[str], profile: str):

        if entity_id in self.entity_profiles:
            # Logic to update an existing entity can be added here later
            return
        
        print(f"Adding entity {entity_id} to resolver indexes.")
        self.entity_profiles[entity_id] = profile
        

        self.fuzzy_choices[entity_id] = names
        
        profile_embedding = self.embedding_model.encode([profile])
        
        self.faiss_index.add(profile_embedding)
        
        new_index_position = self.faiss_index.ntotal - 1
        self.index_to_id[new_index_position] = entity_id
    

    def resolve(self, text: str, context: str, top_k: int = 10, fuzzy_cutoff: int = 80):
        
        candidate_ids = set()

        choices = {name: entity_id for entity_id, names in self.fuzzy_choices.items() for name in names}

        if choices:
            fuzzy_matches = fuzzy_process.extractBests(query=text, choices=choices.keys(), score_cutoff=fuzzy_cutoff, limit=10)
            for match, score in fuzzy_matches:
                candidate_ids.add((choices[match], score))
        
        if self.faiss_index.ntotal > 0:
            query_text = f"{text} mentioned in context of: {context}"
            query_embedding = self.embedding_model.encode([query_text])

            distances, indices = self.faiss_index.search(query_embedding, k=min(top_k, self.faiss_index.ntotal))

            for index_pos, distance in zip(indices[0], distances[0]):
                if index_pos != -1:
                    candidate_ids.add((self.index_to_id[index_pos], distance))
        
        candidates = []

        for entity_id, score in candidate_ids:
            candidates.append({
                "id": entity_id,
                "score": score,
                "profile": self.entity_profiles[entity_id]
            })

        return candidates
