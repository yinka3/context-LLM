import time
from loguru import logger
from typing import Dict, List, Optional
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
    
    def get_max_entity_id(self) -> int:
        """
        Returns the highest entity ID currently in the graph.
        Used on startup to sync Redis counters.
        """
        query = "MATCH (e:Entity) RETURN max(e.id) as max_id"
        with self.driver.session() as session:
            result = session.run(query).single()
            return result["max_id"] if result and result["max_id"] is not None else 0
    
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
                    ON CREATE SET
                        e.canonical_name = $canonical_name,
                        e.aliases = $aliases,
                        e.type = $type,
                        e.summary = $summary,
                        e.confidence = $confidence,
                        e.last_updated = timestamp(),
                        e.last_mentioned = timestamp(),
                        e.embedding = $embedding
                    ON MATCH SET 
                        e.canonical_name = $canonical_name,
                        e.confidence = $confidence,
                        e.last_updated = timestamp(),
                        e.last_mentioned = timestamp()

                    WITH e
                    UNWIND coalesce(e.aliases, []) + $aliases AS alias
                    WITH e, collect(DISTINCT alias) AS unique_aliases
                    SET e.aliases = unique_aliases

                    WITH e
                    FOREACH (_ IN CASE WHEN $topic IS NOT NULL AND $topic <> "" THEN [1] ELSE [] END |
                        MERGE (t:Topic {name: $topic})
                        MERGE (e)-[:BELONGS_TO]->(t)
                    )
                """, **ent, is_user_message=is_user_message)

            for rel in relationships:
                tx.run("""
                    MATCH (a:Entity {canonical_name: $entity_a})
                    MATCH (b:Entity {canonical_name: $entity_b})
                    MERGE (a)-[r:RELATED_TO]-(b)
                    
                    ON CREATE SET 
                        r.weight = 1, 
                        r.confidence = $confidence,
                        r.last_seen = timestamp(), 
                        r.message_ids = [$message_id]
                        
                    ON MATCH SET 
                        r.weight = r.weight + 1,
                        r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                        r.last_seen = timestamp()
                       
                    WITH r
                    UNWIND coalesce(r.message_ids, []) + [$message_id] AS mid
                    WITH r, collect(DISTINCT mid) AS unique_ids
                    SET r.message_ids = unique_ids
                """, **rel)

        with self.driver.session() as session:
            session.execute_write(_write)
    
    def get_all_entities_for_hydration(self) -> list[dict]:
        """
        Fetch all entity data needed to hydrate EntityResolver.
        Single query, single pass.
        """
        query = """
        MATCH (e:Entity)
        WHERE e.id IS NOT NULL
        OPTIONAL MATCH (e)-[:BELONGS_TO]->(t:Topic)
        WHERE t IS NULL OR t.status <> 'inactive'
        RETURN e.id AS id,
            e.canonical_name AS canonical_name,
            e.aliases AS aliases,
            e.type AS type,
            e.topic AS topic,
            e.summary AS summary,
            e.embedding AS embedding
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    

    def update_entity_profile(self, entity_id: int, canonical_name: str, 
                        summary: str, embedding: List[float], 
                        last_msg_id: int, topic: str = "General"):
        """
        Update an existing entity's profile without touching relationships.
        Called by GraphBuilder when processing PROFILE_UPDATE messages.
        """
        def _update(tx: 'ManagedTransaction'):
            tx.run("""
                MERGE (e:Entity {id: $id})
                
                ON CREATE SET
                    e.canonical_name = $canonical_name,
                    e.summary = $summary,
                    e.embedding = $embedding,
                    e.last_profiled_msg_id = $last_msg_id,
                    e.last_updated = timestamp(),
                    e.created_by = 'profile_stream'

                ON MATCH SET
                    e.canonical_name = $canonical_name,
                    e.summary = $summary,
                    e.embedding = $embedding,
                    e.last_updated = timestamp(),
                    e.last_profiled_msg_id = $last_msg_id
                
                WITH e
                FOREACH (_ IN CASE WHEN $topic IS NOT NULL AND $topic <> "" THEN [1] ELSE [] END |
                    MERGE (t:Topic {name: $topic})
                    MERGE (e)-[:BELONGS_TO]->(t)
                )
            """, 
            id=entity_id, 
            canonical_name=canonical_name, 
            summary=summary,
            embedding=embedding,
            last_msg_id=last_msg_id,
            topic=topic
            )
        
        with self.driver.session() as session:
            session.execute_write(_update)
            logger.info(f"Updated entity {entity_id} profile (checkpoint: msg_{last_msg_id})")

    def cleanup_null_entities(self) -> int:
        """Remove entities with null type and their relationships."""
        query = """
        MATCH (e:Entity)
        WHERE e.type IS NULL
        DETACH DELETE e
        RETURN count(e) as deleted
        """
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            deleted = record["deleted"] if record else 0
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} null-type entities")
            return deleted

    def set_topic_status(self, topic_name: str, status: str):
        """Handles Topic State (active/inactive/hot)"""

        query = "MERGE (t:Topic {name: $name}) SET t.status = $status"
        with self.driver.session() as session:
            session.run(query, {"name": topic_name, "status": status}).consume()
    
    def log_daily_mood(self,
                       user_name: str,
                       primary: str, primary_count: int, 
                       secondary: str, secondary_count: int, 
                       total: int):
        query = """
        MATCH (u:Entity {canonical_name: $user_name, type: 'PERSON'})
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
                "user_name": user_name,
                "primary": primary,
                "primary_count": primary_count,
                "secondary": secondary,
                "secondary_count": secondary_count,
                "total": total
            }).consume()
    
    
    def get_hot_topic_context(self, hot_topic_names: List[str]):
        """
        Retrieves the top 3 most recently active entities for each Hot Topic.
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
        """
        query_cypher = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[:BELONGS_TO]->(t:Topic)
        WHERE (e.canonical_name CONTAINS $query 
            OR ANY(alias IN e.aliases WHERE alias CONTAINS $query))
        AND (t IS NULL OR t.status <> 'inactive')
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
        """

        query = """
        MATCH (e:Entity {canonical_name: $name})
        OPTIONAL MATCH (e)-[:BELONGS_TO]->(t:Topic)
        WHERE t IS NULL OR t.status <> 'inactive'
        RETURN e.id as id,
            e.canonical_name as canonical_name,
            e.aliases as aliases,
            e.type as type,
            e.summary as summary,
            e.last_mentioned as last_mentioned,
            e.last_updated as last_updated,
            t.name as topic
        """
        with self.driver.session() as session:
            result = session.run(query, {"name": entity_name})
            record = result.single()
            return dict(record) if record else None

    def get_related_entities(self, entity_names: List[str], active_only: bool = True):
        """
        Find all entities connected to the given entities.
        Use this when the user asks about someone's connections, relationships, network, or "who/what is related to X".
        Set active_only=False if the user wants to include entities from inactive topics.
        Returns: connected entities with connection strength and supporting message references.
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
        """
        cutoff_ms = int((time.time() - (hours * 3600)) * 1000)
        query = """
        MATCH (e:Entity {canonical_name: $name})-[r:RELATED_TO]-(target:Entity)
        WHERE r.last_seen > $cutoff
        RETURN target.canonical_name as entity, r.message_ids as evidence_ids, r.last_seen as time
        ORDER BY r.last_seen DESC
        """
        with self.driver.session() as session:
            result = session.run(query, {"name": entity_name, "cutoff": cutoff_ms})
            return [record.data() for record in result]
    
    
    def _find_path_filtered(self, start_name: str, end_name: str, active_only: bool = True) -> List[Dict]:
        query = """
        MATCH (start:Entity {canonical_name: $start_name})
        MATCH (end:Entity {canonical_name: $end_name})
        MATCH p = shortestPath((start)-[:RELATED_TO*..4]-(end))
        WHERE ALL(n IN nodes(p) WHERE
            NOT EXISTS((n)-[:BELONGS_TO]->(:Topic {status: 'inactive'}))
            OR $active_only = false
        )
        RETURN [n in nodes(p) | n.canonical_name] as names,
            [r in relationships(p) | r.message_ids] as evidence_ids
        """
        with self.driver.session() as session:
            result = session.run(query, {
                "start_name": start_name, 
                "end_name": end_name,
                "active_only": active_only
            })
            record = result.single()
            if not record:
                return []
            
            path_data = []
            names = record["names"]
            evidence = record["evidence_ids"]
            for i in range(len(evidence)):
                path_data.append({
                    "step": i,
                    "entity_a": names[i],
                    "entity_b": names[i+1],
                    "evidence_refs": evidence[i]
                })
            return path_data
    

    def _fetch_entity(self, entity_id: int) -> Optional[Dict]:
        """Fetch entity properties by ID."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e.id as id,
            e.canonical_name as canonical_name,
            e.aliases as aliases,
            e.type as type,
            e.summary as summary,
            e.embedding as embedding,
            e.confidence as confidence,
            e.last_mentioned as last_mentioned,
            e.last_updated as last_updated
        """
        with self.driver.session() as session:
            result = session.run(query, {"entity_id": entity_id})
            record = result.single()
            return dict(record) if record else None
    
    def merge_entities(self, primary_id: int, secondary_id: int, merged_summary: str) -> bool:
        """
        Merge secondary entity into primary (single transaction).
        Primary survives with combined data, secondary is deleted.
        
        Args:
            primary_id: Entity that survives
            secondary_id: Entity that gets merged and deleted
            merged_summary: Pre-computed summary (from LLM or concat)
        """
        
        query = """
        MATCH (p:Entity {id: $primary_id})
        MATCH (s:Entity {id: $secondary_id})

        WITH p, s, coalesce(p.aliases, []) + coalesce(s.aliases, []) + [s.canonical_name] AS combined_aliases
        UNWIND combined_aliases AS alias
        WITH p, s, collect(DISTINCT alias) AS unique_aliases

        SET p.aliases = unique_aliases,
            p.summary = $summary,
            p.confidence = CASE WHEN coalesce(s.confidence, 0) > coalesce(p.confidence, 0) THEN s.confidence ELSE p.confidence END,
            p.last_mentioned = CASE WHEN coalesce(s.last_mentioned, 0) > coalesce(p.last_mentioned, 0) THEN s.last_mentioned ELSE p.last_mentioned END,
            p.last_updated = timestamp()

        WITH p, s

        OPTIONAL MATCH (s)-[r_source:RELATED_TO]-(target:Entity)
        WHERE target.id <> p.id

        WITH p, s, r_source, target
        WHERE r_source IS NOT NULL

        MERGE (p)-[r_target:RELATED_TO]-(target)
        ON CREATE SET 
            r_target.weight = r_source.weight,
            r_target.confidence = r_source.confidence,
            r_target.message_ids = r_source.message_ids,
            r_target.last_seen = r_source.last_seen
        ON MATCH SET
            r_target.weight = r_target.weight + r_source.weight,
            r_target.confidence = CASE WHEN r_source.confidence > r_target.confidence THEN r_source.confidence ELSE r_target.confidence END,
            r_target.last_seen = CASE WHEN r_source.last_seen > r_target.last_seen THEN r_source.last_seen ELSE r_target.last_seen END

        WITH p, s, r_target, r_source
        UNWIND coalesce(r_target.message_ids, []) + coalesce(r_source.message_ids, []) AS mid
        WITH p, s, r_target, collect(DISTINCT mid) AS unique_mids
        SET r_target.message_ids = unique_mids

        WITH DISTINCT s
        DETACH DELETE s
        RETURN count(s) as deleted
        """

        with self.driver.session() as session:
            try:
                result = session.run(query, {
                    "primary_id": primary_id, 
                    "secondary_id": secondary_id, 
                    "summary": merged_summary
                })
                record = result.single()
                if record and record["deleted"] > 0:
                    logger.info(f"Merged entity {secondary_id} into {primary_id}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Merge transaction failed: {e}")
                return False