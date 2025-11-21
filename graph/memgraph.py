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
                        relation: str, confidence: float = 1.0):
        
        """
        Handles relationship creation with generic head verb with specific verb property
        """

        query = """
        MATCH (a:Entity {canonical_name: $source_name})
        MATCH (b:Entity {canonical_name: $target_name})
        MERGE (a)-[r:RELATED_TO]->(b)
        SET r.verb = $verb,
            r.confidence = $confidence,
            r.last_seen = timestamp()
        """

        with self.driver.session() as session:
            session.run(query, 
                        {
                            "source_name": source_name,
                            "target_name": target_name,
                            "verb": relation,
                            "confidence": confidence
                        })
    
    def correct_relationship(self, source_name: str, target_name: str, new_verb: str):
        """
        Handles atomically deleting old edges and creating new ones
        """

        query = """
        MATCH (a:Entity {canonical_name: $source_name})-[r:RELATED_TO]-(b:Entity {canonical_name: $target_name})
        DELETE r
        WITH a, b
        MERGE (a)-[new_r:RELATED_TO]->(b)
        SET new_r.verb = $verb,
            new_r.confidence = 1.0,  // User truth
            new_r.last_seen = timestamp()
        """
        with self.driver.session() as session:
            session.run(query, {"source_name": source_name, "target_name": target_name, "verb": new_verb})
    

    def set_topic_status(self, topic_name: str, status: str):
        """Handles Topic State (active/inactive/hot)"""

        query = "MERGE (t:Topic {name: $name}) SET t.status = $status"
        with self.driver.session() as session:
            session.run(query, {"name": topic_name, "status": status})
    
    def log_daily_mood(self, user_name: str, emotion_data: Dict):
        """Part 2.5: Memory Management - Log Mood Time Series"""
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