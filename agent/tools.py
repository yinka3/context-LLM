from typing import List, Dict, Optional

from graph.memgraph import MemGraphStore


class Tools:
    
    def __init__(self, store: Optional[MemGraphStore] = None):
        self.store = store or MemGraphStore()

    def search_messages(self, query: str) -> List[Dict]:
        """
        Search past user messages by semantic similarity.
        Use when looking for what the user said about a topic, event, or person.
        Good starting point when the query references past conversations.
        
        Args:
            query: Keywords or phrase to search for
        
        Returns: List of messages with content, timestamp, and relevance score.
        """
        return []

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
        return self.store.get_entity_profile(entity_name)

    def get_connections(self, entity_name: str, active_only: bool = True) -> List[Dict]:
        """
        Find all entities connected to a given entity.
        Use when asked about someone's relationships, network, or "who knows who".
        
        Args:
            entity_name: The entity to find connections for
            active_only: If True, exclude entities from inactive topics (default True)
        
        Returns: List of connections with target entity, connection strength, evidence message IDs.
        """
        return self.store.get_related_entities([entity_name], active_only) or []

    def get_recent_activity(self, entity_name: str, hours: int = 24) -> List[Dict]:
        """
        Get recent interactions involving an entity within a time window.
        Use when asked "what happened with X recently" or "any updates on X".
        
        Args:
            entity_name: Entity to check activity for
            hours: How far back to look (default 24, use 168 for "this week")
        
        Returns: Recent interactions with timestamps and evidence message IDs.
        """
        return self.store.get_recent_activity(entity_name, hours) or []

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
        return self.store.find_connection(entity_a, entity_b) or []

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