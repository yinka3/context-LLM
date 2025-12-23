import json
from typing import List, Dict, Optional, TYPE_CHECKING

import faiss
import numpy as np
from rapidfuzz import process as fuzzy_process, fuzz
import redis
from main.entity_resolve import EntityResolver
from db.memgraph import MemGraphStore



class Tools:
    
    def __init__(self, user_name: str, store: MemGraphStore, ent_resolver: EntityResolver, redis_client: redis.Redis):
        self.store = store
        self.resolver = ent_resolver
        self.user_name = user_name
        self.redis = redis_client
    
    def _resolve_entity_name(self, entity: str) -> Optional[str]:
        """Resolve user input to canonical entity name via exact or fuzzy match."""
        
        
        entity_id = self.resolver._name_to_id.get(entity)
        if entity_id:
            profile = self.resolver.entity_profiles.get(entity_id)
            return profile["canonical_name"] if profile else entity
        
        if not self.resolver._name_to_id:
            return None
        
        result = fuzzy_process.extractOne(
            query=entity,
            choices=self.resolver._name_to_id.keys(),
            scorer=fuzz.WRatio,
            score_cutoff=85
        )
        
        if result:
            matched_name, _, _ = result
            entity_id = self.resolver._name_to_id[matched_name]
            profile = self.resolver.entity_profiles.get(entity_id)
            return profile["canonical_name"] if profile else matched_name
        
        return None

    
    def search_messages(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search past user messages by semantic similarity.
        Use when looking for what the user said about a topic, event, or person.
        Good starting point when the query references past conversations.
        
        Args:
            query: Keywords or phrase to search for
            limit: Max results (default 5)
        
        Returns: List of messages with content, timestamp, and relevance score.
        """
        content_key = f"message_content:{self.user_name}"
        all_messages = self.redis.hgetall(content_key)
        if not all_messages:
            return []
        
        msg_ids = []
        texts = []
        parsed_data = {}
        
        for msg_id, msg_data in all_messages.items():
            msg_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
            parsed = json.loads(msg_data)
            msg_ids.append(msg_id)
            texts.append(parsed["message"])
            parsed_data[msg_id] = parsed
        
        all_texts = [query] + texts
        embeddings = self.resolver.embedding_model.encode(all_texts)
        faiss.normalize_L2(embeddings)
        
        query_vec = embeddings[0]
        msg_vecs = embeddings[1:]
        
        scores = np.dot(msg_vecs, query_vec)
        top_indices = np.argsort(scores)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            msg_id = msg_ids[idx]
            results.append({
                "id": msg_id,
                "message": parsed_data[msg_id]["message"],
                "timestamp": parsed_data[msg_id]["timestamp"],
                "score": float(scores[idx])
            })
        
        return results


    def search_entities(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for entities by name or alias.
        Use when you need to find a person, place, or thing but aren't sure of exact name.
        Returns partial matches — use get_profile for full details after identifying.
        
        Args:
            query: Name or partial name to search
            limit: Max results to return (default 5)
        
        Returns: List of matching entities with id, name, summary snippet, type.
        """
        return self.store.search_entity(query, limit) or []

    def get_profile(self, entity_name: str) -> Optional[Dict]:
        """
        Get full profile for a specific entity.
        Use when you know the exact entity name and need complete information.
        
        Args:
            entity_name: Exact canonical name of the entity
        
        Returns: Full profile with summary, type, aliases, topic, last_mentioned.
        Returns None if entity not found.
        """
        canonical = self._resolve_entity_name(entity_name)
        if not canonical:
            return None
        
        entity_id = self.resolver.get_id(canonical)
        if entity_id:
            profile = self.resolver.entity_profiles.get(entity_id)
            if profile:
                return profile
            
        return self.store.get_entity_profile(canonical)

    def get_connections(self, entity_name: str, active_only: bool = True) -> List[Dict]:
        """
        Find all entities connected to a given entity.
        Use when asked about someone's relationships, network, or "who knows who".
        
        Args:
            entity_name: The entity to find connections for
            active_only: If True, exclude entities from inactive topics (default True)
        
        Returns: List of connections with target entity, connection strength, evidence message IDs.
        """
        canonical = self._resolve_entity_name(entity_name)
        if not canonical:
            return []
        return self.store.get_related_entities([canonical], active_only) or []

    def get_recent_activity(self, entity_name: str, hours: int = 24) -> List[Dict]:
        """
        Get recent interactions involving an entity within a time window.
        Use when asked "what happened with X recently" or "any updates on X".
        
        Args:
            entity_name: Entity to check activity for
            hours: How far back to look (default 24, use 168 for "this week")
        
        Returns: Recent interactions with timestamps and evidence message IDs.
        """
        canonical = self._resolve_entity_name(entity_name)
        if not canonical:
            return []
        return self.store.get_recent_activity(canonical, hours) or []

    def find_path(self, entity_a: str, entity_b: str) -> List[Dict]:
        """
        Find the shortest connection path between two entities.
        Use when asked "how is X connected to Y" or "what's the relationship between X and Y".
        Requires both entities to be known — use get_profile first if unsure.
        
        Args:
            entity_a: First entity name
            entity_b: Second entity name
        
        Returns: Step-by-step path showing each entity in the chain with evidence.
        Returns empty list if no connection found.
        """
        canonical_a = self._resolve_entity_name(entity_a)
        canonical_b = self._resolve_entity_name(entity_b)
        if not canonical_a or not canonical_b:
            return []
        return self.store.find_connection(canonical_a, canonical_b) or []

    def get_hot_topic_context(self, hot_topics: List[str]) -> Dict[str, List[Dict]]:
        """
        Retrieve pre-cached context for frequently accessed topics.
        Called automatically at start — you already have this data in hot_topic_context.
        Only call manually if hot topics changed mid-conversation.
        
        Args:
            hot_topics: List of topic names marked as "hot"
        
        Returns: Dict mapping topic name to list of top entities with summaries.
        """
        if not hot_topics:
            return {}
        return self.store.get_hot_topic_context(hot_topics)

    def web_search(self, query: str) -> List[Dict]:
        """
        Search the web for external information.
        Use ONLY for current events, external facts, or information not in the user's graph.
        This is a separate path — once you go web, you cannot use internal tools.
        
        Args:
            query: Search query
        
        Returns: List of web results with title, snippet, url.
        """
        # TODO: Implement web search
        return []