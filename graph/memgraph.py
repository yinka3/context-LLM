import time
from loguru import logger
from typing import Dict, List
from neo4j import GraphDatabase, Driver, ManagedTransaction



class MemGraphStore:
    def __init__(self, uri: str = "bolt://localhost:7687"):
        self.driver: Driver = GraphDatabase.driver(uri)
        self.verify_conn()
        self._setup_schema()
        logger.info("Graph store initialized")

    def close(self):
        if self.driver:
            self.driver.close()
    
    def verify_conn(self):
        self.driver.verify_connectivity()
    
    def _setup_schema(self):
        """
        Create indices and constraints to ensure performance and data integrity.
        """
        queries = [
            "CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE;",
            "CREATE CONSTRAINT ON (t:Topic) ASSERT t.name IS UNIQUE",
            "CREATE INDEX ON :DailyMood(date)",
            "CREATE INDEX ON :Entity(canonical_name);"
        ]
        
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.debug(f"Schema setup note: {e}")
        logger.info("Memgraph schema indices verified.")
    
    def write_batch(self, entities: List[Dict], relationships: List[Dict], is_user_message: bool = False):
        def _write(tx: 'ManagedTransaction'):
            for ent in entities:
                tx.run("""
                    MERGE (e:Entity {id: $id})
                    SET e.canonical_name = $name,
                        e.type = $type,
                        e.summary = $summary,
                        e.confidence = $confidence,
                        e.last_updated = timestamp(),
                        e.last_mentioned = CASE WHEN $is_user_message THEN timestamp() ELSE e.last_mentioned END,
                        e.embedding = $embedding,
                        e.aliases = $aliases
                    WITH e
                    FOREACH (_ IN CASE WHEN $topic IS NOT NULL AND $topic <> "" THEN [1] ELSE [] END |
                        MERGE (t:Topic {name: $topic})
                        MERGE (e)-[:BELONGS_TO]->(t)
                    )
                """, id=ent["id"], name=ent["canonical_name"], type=ent["type"],
                    summary=ent.get("summary", ""), confidence=ent.get("confidence", 1.0),
                    embedding=ent.get("embedding", []), aliases=ent.get("aliases", []),
                    topic=ent.get("topic"), is_user_message=is_user_message)

            for rel in relationships:
                tx.run("""
                    MATCH (a:Entity {canonical_name: $source_name})
                    MATCH (b:Entity {canonical_name: $target_name})
                    MERGE (a)-[r:RELATED_TO]-(b)
                    ON CREATE SET 
                        r.verb = $relation, r.weight = 1, r.confidence = $confidence,
                        r.last_seen = timestamp(), r.message_ids = [$message_id]
                    ON MATCH SET 
                        r.weight = r.weight + 1,
                        r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                        r.last_seen = timestamp(),
                        r.message_ids = CASE WHEN NOT ($message_id IN r.message_ids) 
                            THEN r.message_ids + $message_id ELSE r.message_ids END
                """, **rel)

        with self.driver.session() as session:
            session.execute_write(_write)
    

    def set_topic_status(self, topic_name: str, status: str):
        """Handles Topic State (active/inactive/hot)"""

        query = "MERGE (t:Topic {name: $name}) SET t.status = $status"
        with self.driver.session() as session:
            session.run(query, {"name": topic_name, "status": status}).consume()
    
    def log_daily_mood(self,
                       primary: str, primary_count: int, 
                       secondary: str, secondary_count: int, 
                       total: int):
        query = """
        MATCH (u:Entity {canonical_name: 'USER', type: 'PERSON'})
        CREATE (m:DailyMood {
            date: date(),
            timestamp: timestamp(),
            primary_emotion: $primary,
            primary_count: $primary_count,
            secondary_emotion: $secondary,
            secondary_count: $secondary_count,
            total_messages: $total
        })
        MERGE (u)-[:FELT]->(m)
        """
        with self.driver.session() as session:
            session.run(query, {
                "primary": primary,
                "primary_count": primary_count,
                "secondary": secondary,
                "secondary_count": secondary_count,
                "total": total
            }).consume()
    
    def get_hot_topic_context(self, hot_topic_names: List[str]):
        """
        Retrieves the top 3 most recently active entities for each Hot Topic.
        Used ONLY for the Chat LLM context window.
        """
        query = """
        MATCH (t:Topic) WHERE t.name IN $hot_topics
        MATCH (e:Entity)-[:BELONGS_TO]->(t)

        WITH t, e ORDER BY e.last_mentioned DESC 
        WITH t, collect(e)[..3] as top_entities
        UNWIND top_entities as e
        RETURN t.name as topic, e.canonical_name as name, e.summary as summary
        """
        
        with self.driver.session() as session:
            result = session.run(query, {"hot_topics": hot_topic_names})
            
            grouped = {}
            for record in result:
                topic = record["topic"]
                if topic not in grouped:
                    grouped[topic] = []
                grouped[topic].append({
                    "name": record["name"],
                    "summary": record["summary"]
                })
            
            return grouped
    
    def search_entity(self, query: str, limit: int = 5):
        """
        Search for entities by name or alias.
        Use this when the user mentions a person, place, or thing and you need to verify it exists or find the correct spelling.
        Returns: matching entities with their ID, name, summary, and type.
        """
        query_cypher = """
        MATCH (e:Entity)
        WHERE e.canonical_name CONTAINS $query 
        OR ANY(alias IN e.aliases WHERE alias CONTAINS $query)
        RETURN e.id as id, e.canonical_name as name, e.summary as summary, e.type as type
        ORDER BY e.last_mentioned DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query_cypher, {"query": query, "limit": limit})
            return [record.data() for record in result]
    
    def get_entity_profile(self, entity_name: str):
        """
        Get the full profile for a specific entity.
        Use this when the user asks "what do you know about X" or "tell me about X" for a single person, place, or thing.
        Returns: the entity's summary, type, aliases, topic, and when it was last mentioned.
        """

        query = """
        MATCH (e:Entity {canonical_name: $name})
        OPTIONAL MATCH (e)-[:BELONGS_TO]->(t:Topic)
        RETURN e.id as id, e.canonical_name as name, e.summary as summary, 
            e.type as type, e.aliases as aliases, t.name as topic,
            e.last_mentioned as last_mentioned
        """
        with self.driver.session() as session:
            result = session.run(query, {"name": entity_name})
            record = result.single()
            return record.data() if record else None

    def get_related_entities(self, entity_names: List[str], active_only: bool = True):
        """
        Find all entities connected to the given entities and how they relate.
        Use this when the user asks about someone's connections, relationships, network, or "who/what is related to X".
        Set active_only=False if the user wants to include entities from inactive topics.
        Returns: connected entities with relationship type, connection strength, and supporting message references.
        """

        query = """
        MATCH (source:Entity) WHERE source.canonical_name IN $names
        MATCH (source)-[r:RELATED_TO]-(target:Entity)
        OPTIONAL MATCH (target)-[:BELONGS_TO]->(t:Topic)
        WITH source, r, target, t
        WHERE
            ($active_only = false) OR
            (t IS NULL) OR
            (t.status <> 'inactive')
        RETURN
            source.canonical_name as source,
            target.canonical_name as target,
            target.summary as target_summary,
            r.verb as relation,
            r.weight as connection_strength,
            r.message_ids as evidence_ids,
            r.confidence as confidence,
            r.last_seen as last_seen
        ORDER BY r.weight DESC, r.last_seen DESC
        LIMIT 50
        """
        with self.driver.session() as session:
            res = session.run(query, {"names": entity_names, "active_only": active_only})
            return [record.data() for record in res]
        
    
    def get_recent_activity(self, entity_name: str, hours: int = 24):
        """
        Get recent interactions involving an entity within a time window.
        Use this when the user asks "what happened with X recently" or "any updates on X" or wants time-filtered information.
        Adjust hours parameter based on user's timeframe (24 for "today", 168 for "this week", etc).
        Returns: recent interactions with timestamps.
        """
        cutoff_ms = int((time.time() - (hours * 3600)) * 1000)
        query = """
        MATCH (e:Entity {canonical_name: $name})-[r:RELATED_TO]-(target:Entity)
        WHERE r.last_seen > $cutoff
        RETURN target.canonical_name as entity, r.verb as action, r.last_seen as time
        ORDER BY r.last_seen DESC
        """
        with self.driver.session() as session:
            result = session.run(query, {"name": entity_name, "cutoff": cutoff_ms})
            return [record.data() for record in result]
    

    def find_connection(self, start_name: str, end_name: str):
        """
        Find the shortest path connecting two entities.
        Use this when the user asks "how is X connected to Y" or "what's the relationship between X and Y".
        Returns: step-by-step path showing each entity and relationship in the chain, with message references as evidence.
        """
        query = """
        MATCH (start:Entity {canonical_name: $start_name}), (end:Entity {canonical_name: $end_name})
        MATCH p = shortestPath((start)-[:RELATED_TO*..4]-(end))
        RETURN [n in nodes(p) | n.canonical_name] as names, 
            [r in relationships(p) | r.message_ids] as evidence_ids,
            [r in relationships(p) | r.verb] as relations
        """
        with self.driver.session() as session:
            result = session.run(query, {"start_name": start_name, "end_name": end_name})
            record = result.single()
            if record:
                path_data = []
                names = record["names"]
                evidence = record["evidence_ids"]
                relations = record["relations"]
                for i in range(len(evidence)):
                    path_data.append({
                        "step": i,
                        "entity_a": names[i],
                        "entity_b": names[i+1],
                        "relation": relations[i],
                        "evidence_refs": evidence[i]
                    })
                return path_data
            return []