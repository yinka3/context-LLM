from datetime import datetime
from typing import Dict, List
from functools import lru_cache
import numpy as np
from dtypes import EntityData
from entity import EntityResolver
from sentence_transformers import SentenceTransformer, util
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ERTransformer(EntityResolver):

    CONCEPT_WINDOW_SIZE = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.semantic_model = SentenceTransformer('all-MiniLM-L12-v2')
        self._get_cached_embedding = lru_cache(maxsize=1000)(self._compute_embedding)
        self.entity_cache = {}
        self.entity_history = {}
    
    def _compute_embedding(self, text: str):
        return self.semantic_model.encode([text])[0]
    
    def _handle_concept_drift(self, entity_id: int, new_context: str):
        """
        STUB: This function will eventually trigger a clarification question to the user.
        For now, it just logs a warning.
        """
        logger.warning(
            f"CONCEPT DRIFT DETECTED for entity {entity_id} "
            f"with new context: '{new_context}'. User clarification needed."
        )
        # In a real implementation, this would return a special flag or
        # add a clarification request to a response queue.
        pass

    def _check_candidate_ambiguity(self, top_candidates: List[EntityData], entity_data: Dict) -> bool:
        """
        Overrides the parent's empty hook to use semantic ambiguity detection.
        """
        is_ambiguous, avg_similarity = self.detect_mult_ent(top_candidates)
        logger.info(f"Ambiguity check for '{entity_data['text']}': {is_ambiguous} (avg similarity: {avg_similarity:.2f})")
        return is_ambiguous
    
    def track_entity_change(self, entity_id: int, new_context: str, timestamp: datetime):

        new_emb = self._get_cached_embedding(new_context)

        if entity_id not in self.entity_history:
            self.entity_history[entity_id] = deque(maxlen=self.CONCEPT_WINDOW_SIZE)

        history_window = self.entity_history[entity_id]


        if len(history_window) > 5:

            past_embs = [item['embedding'] for item in history_window]
            centroid = np.mean(past_embs, axis=0)

            new_sim_score = util.cos_sim(new_emb, centroid).item()

            internal_similarities = [util.cos_sim(emb, centroid).item() for emb in past_embs]
            mean_sim = np.mean(internal_similarities)
            std_sim = np.std(internal_similarities)

            drift_threshold = mean_sim - (2 * std_sim)

            if new_sim_score < drift_threshold:
                self._handle_concept_drift(entity_id, new_context)
            
        history_window.append({
            'embedding': new_emb,
            'timestamp': timestamp,
            'context': new_context
        })

    def detect_mult_ent(self, candidates: List[EntityData]):
        if len(candidates) < 2:
            return False, 0.0
        
        embeddings = self.semantic_model.encode([c.name for c in candidates])
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = util.cos_sim(embeddings[i], embeddings[j])
                similarities.append(sim.item())

        avg_similarity = np.mean(similarities)
        is_ambiguous = avg_similarity > 0.85
        
        return is_ambiguous, avg_similarity
    
    def _get_match_score(self, search_text: str, candidate: EntityData):
        """Multi-level scoring with strict thresholds"""

        parent_score = super()._get_match_score(search_text, candidate)

        if parent_score >= 0.9:
            return parent_score
        
        search_embedding = self._get_cached_embedding(search_text)
        candidate_embedding = self._get_cached_embedding(candidate.name.lower())
        
        similarity = util.cos_sim(search_embedding, candidate_embedding).item()

        if hasattr(candidate, 'type') and super()._are_types_compatible(
            search_text, candidate.type):
            similarity *= 1.1
        
        if parent_score > 0.7:
            return min(1.0, 0.6 * parent_score + 0.4 * similarity)
        else:
            return min(1.0, similarity)


    