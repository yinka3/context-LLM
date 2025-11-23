import logging
import time
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)

class MemGraphStore:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(MemGraphStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("admin", "password")):

        if not hasattr(self, 'driver'):
            try:
                self.driver: Driver = GraphDatabase.driver(uri, auth=auth)
                self.verify_conn()
                self._setup_schema()
                logger.info("Graph store initialized")
            except Exception as e:
                logger.error(f"Failed to connect to Memgraph: {e}")
                raise

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
    
    def upsert_entity(self, entity_data: Dict[str, Any]):
        """
        Handles atomic Create/Update and Topic Linkage
        """

        query = """
        MERGE (e:Entity {id: $id})
        SET e.canonical_name = $name,
            e.type = $type,
            e.summary = $summary,
            e.confidence = $confidence,
            e.last_updated = timestamp()
        
        SET e.aliases = $aliases

        WITH e AS entity
        FOREACH (_ IN CASE WHEN $topic_name IS NOT NULL AND $topic_name <> "" THEN [1] ELSE [] END |
                MERGE (t:Topic {name: $topic_name})
                MERGE (e)-[:BELONGS_TO]->(t)
        )
        """

        params = {
            "id": entity_data["id"],
            "name": entity_data["name"],
            "type": entity_data["type"],
            "summary": entity_data.get("summary", "No Summary Provided Yet."),
            "confidence": entity_data.get("confidence", 1.0),
            "aliases": entity_data.get("aliases", []),
            "topic_name": entity_data.get("topic", None)
        }

        with self.driver.session() as session:
            session.run(query, params)
    

    def add_relationship(self, source_name: str, target_name: str, 
                        message_id: str, confidence: float = 1.0):
        
        """
        Handles relationship creation with generic head verb with specific verb property
        """

        query = """
        MATCH (a:Entity {canonical_name: $source_name})
        MATCH (b:Entity {canonical_name: $target_name})
        
        MERGE (a)-[r:RELATED_TO]-(b)
        
        ON CREATE SET 
            r.weight = 1,
            r.confidence = $confidence,
            r.last_seen = timestamp(),
            r.message_ids = [$msg_id]

        ON MATCH SET 
            r.weight = r.weight + 1, 
            r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
            r.last_seen = timestamp(),
            r.message_ids = CASE 
                WHEN NOT ($msg_id IN r.message_ids) THEN r.message_ids + $msg_id 
                ELSE r.message_ids 
            END
        """
        
        with self.driver.session() as session:
            session.run(query, {
                "source_name": source_name,
                "target_name": target_name,
                "msg_id": message_id,
                "confidence": confidence
            })
    

    def set_topic_status(self, topic_name: str, status: str):
        """Handles Topic State (active/inactive/hot)"""

        query = "MERGE (t:Topic {name: $name}) SET t.status = $status"
        with self.driver.session() as session:
            session.run(query, {"name": topic_name, "status": status})
    
    def log_daily_mood(self, user_name: str, emotion_data: Dict):
        """Handles Mood Time Series"""
        
        query = """
        MATCH (u:Entity {canonical_name: $user_name, type: 'PERSON'})
        CREATE (m:DailyMood {
            date: date(), 
            timestamp: timestamp(),
            dominant_emotion: $emotion,
            score: $score
        })
        MERGE (u)-[:FELT]->(m)
        """
        with self.driver.session() as session:
            session.run(query, {
                "user_name": user_name, 
                "emotion": emotion_data.get("label"), 
                "score": emotion_data.get("score")
            })
    

    def get_subgraph_context(self, entity_ids: List[int], active_only: bool = True):
        """
        Handles generic context retrieval
        """

        query = """
        MATCH (source:Entity) WHERE source.id IN $ids
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
            r.weight as connection_strength,
            r.message_ids as evidence_ids,
            r.confidence as confidence,
            r.last_seen as last_seen
        
        ORDER BY r.weight DESC, r.last_seen DESC
        LIMIT 50
        """

        with self.driver.session() as session:
            res = session.run(query, {
                "ids": entity_ids,
                "active_only": active_only
            })

            return [record.data() for record in res]
        
    
    def get_recent_interaction(self, entity_id: int, hours: int = 24):
        cutoff_ms = int((time.time() - (hours * 3600)) * 1000)
        
        query = """
        MATCH (e:Entity {id: $id})-[r:RELATED_TO]-(target:Entity)
        WHERE r.last_seen > $cutoff
        RETURN target.canonical_name as entity, r.verb as action, r.last_seen as time
        ORDER BY r.last_seen DESC
        """
        with self.driver.session() as session:
            result = session.run(query, {"id": entity_id, "cutoff": cutoff_ms})
            return [record.data() for record in result]
    

    def find_connection(self, start_id: int, end_id: int):
        """Handles retrieval for main reasoning"""

        query = """
        MATCH (start:Entity {id: $start_id}), (end:Entity {id: $end_id})
        // Shortest path based on fewer hops (unweighted breadth-first)
        // Future optimization: Use Dijkstra with 'weight' to find 'Strongest Path'
        MATCH p = shortestPath((start)-[:RELATED_TO*..4]-(end))
        
        RETURN [n in nodes(p) | n.canonical_name] as names, 
               [r in relationships(p) | r.message_ids] as evidence_ids
        """

        with self.driver.session() as session:
            result = session.run(query, {"start_id": start_id, "end_id": end_id})
            record = result.single()
            
            if record:
                path_data = []
                names = record["names"]
                evidence = record["evidence_ids"]
                
                for i in range(len(evidence)):
                    path_data.append({
                        "step": i,
                        "entity_a": names[i],
                        "entity_b": names[i+1],
                        # We return the IDs, Python fetches the text from Redis
                        "evidence_refs": evidence[i] 
                    })
                return path_data
            return []