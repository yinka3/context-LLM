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
                        e.aliases = apoc.coll.toSet(coalesce(e.aliases, []) + $aliases),
                        e.confidence = $confidence,
                        e.last_updated = timestamp(),
                        e.last_mentioned = timestamp()

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
                        r.last_seen = timestamp(),
                        r.message_ids = apoc.coll.toSet(r.message_ids + [$message_id])
                """, **rel)

        with self.driver.session() as session:
            session.execute_write(_write)
    

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
    
    def get_all_embeddings(self) -> Dict[int, List[float]]:
        """
        Fetch all entity embeddings to hydrate FAISS at startup.
        Returns: {entity_id: [float, float, ...]}
        """
        query = "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN e.id as id, e.embedding as vec"
        with self.driver.session() as session:
            result = session.run(query)
            return {record["id"]: record["vec"] for record in result}
    
    def get_all_aliases_map(self) -> Dict[str, int]:
        """
        Rebuild the Name->ID map from the database.
        Used for Cold Sync when Redis is empty.
        Returns: {'Destiny': 50, 'Des': 50, ...}
        """
        query = """
        MATCH (e:Entity) 
        RETURN e.id as id, e.canonical_name as name, e.aliases as aliases
        """
        mapping = {}
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                entity_id = record["id"]
                
                if record["name"]:
                    mapping[record["name"]] = entity_id
                
                if record["aliases"]:
                    for alias in record["aliases"]:
                        mapping[alias] = entity_id
                        
        return mapping
    
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
        Use this when the user asks "what happened with X recently" or "any updates on X" or wants time-filtered information.
        Adjust hours parameter based on user's timeframe (24 for "today", 168 for "this week", etc).
        Returns: recent interactions with timestamps and evidence.
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
    

    def find_connection(self, start_name: str, end_name: str):
        """
        Find the shortest path connecting two entities.
        Use this when the user asks "how is X connected to Y" or "what's the relationship between X and Y".
        Returns: step-by-step path showing each entity in the chain, with message references as evidence.
        """
        query = """
        MATCH (start:Entity {canonical_name: $start_name}), (end:Entity {canonical_name: $end_name})
        MATCH p = shortestPath((start)-[:RELATED_TO*..4]-(end))
        RETURN [n in nodes(p) | n.canonical_name] as names, 
            [r in relationships(p) | r.message_ids] as evidence_ids
        """
        with self.driver.session() as session:
            result = session.run(query, {"start_name": start_name, "end_name": end_name})
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
                        "evidence_refs": evidence[i]
                    })
                return path_data
            return []
    

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
        
        def _execute_merge(tx: ManagedTransaction) -> bool:
            primary = tx.run("""
                MATCH (e:Entity {id: $id})
                RETURN e.canonical_name as canonical_name,
                    e.aliases as aliases,
                    e.confidence as confidence,
                    e.last_mentioned as last_mentioned
            """, {"id": primary_id}).single()
            
            secondary = tx.run("""
                MATCH (e:Entity {id: $id})
                RETURN e.canonical_name as canonical_name,
                    e.aliases as aliases,
                    e.confidence as confidence,
                    e.last_mentioned as last_mentioned
            """, {"id": secondary_id}).single()
            
            if not primary or not secondary:
                return False
            
            primary_aliases = set(primary["aliases"] or [])
            secondary_aliases = set(secondary["aliases"] or [])
            secondary_aliases.add(secondary["canonical_name"])
            merged_aliases = list(primary_aliases | secondary_aliases)
            
            merged_confidence = max(
                primary["confidence"] or 0,
                secondary["confidence"] or 0
            )
            merged_last_mentioned = max(
                primary["last_mentioned"] or 0,
                secondary["last_mentioned"] or 0
            )
            
            tx.run("""
                MATCH (e:Entity {id: $id})
                SET e.aliases = $aliases,
                    e.summary = $summary,
                    e.confidence = $confidence,
                    e.last_mentioned = $last_mentioned,
                    e.last_updated = timestamp()
            """, {
                "id": primary_id,
                "aliases": merged_aliases,
                "summary": merged_summary,
                "confidence": merged_confidence,
                "last_mentioned": merged_last_mentioned
            })
            
            rels_result = tx.run("""
                MATCH (e:Entity {id: $id})-[r:RELATED_TO]-(target:Entity)
                RETURN target.id as target_id,
                    r.weight as weight,
                    r.confidence as confidence,
                    r.message_ids as message_ids,
                    r.last_seen as last_seen
            """, {"id": secondary_id})
            
            relationships = [dict(record) for record in rels_result]
            
            for rel in relationships:
                if rel["target_id"] == primary_id:
                    continue
                
                tx.run("""
                    MATCH (a:Entity {id: $primary_id})
                    MATCH (b:Entity {id: $target_id})
                    MERGE (a)-[r:RELATED_TO]-(b)
                    ON CREATE SET
                        r.weight = $weight,
                        r.confidence = $confidence,
                        r.message_ids = $message_ids,
                        r.last_seen = $last_seen
                    ON MATCH SET
                        r.weight = r.weight + $weight,
                        r.confidence = CASE WHEN $confidence > r.confidence 
                                            THEN $confidence ELSE r.confidence END,
                        r.message_ids = apoc.coll.toSet(
                            coalesce(r.message_ids, []) + $message_ids
                        ),
                        r.last_seen = CASE WHEN $last_seen > r.last_seen 
                                        THEN $last_seen ELSE r.last_seen END
                """, {
                    "primary_id": primary_id,
                    "target_id": rel["target_id"],
                    "weight": rel["weight"] or 1,
                    "confidence": rel["confidence"] or 0.5,
                    "message_ids": rel["message_ids"] or [],
                    "last_seen": rel["last_seen"] or 0
                })
            
            tx.run("""
                MATCH (e:Entity {id: $id})
                DETACH DELETE e
            """, {"id": secondary_id})
            
            return True
        
        with self.driver.session() as session:
            try:
                result = session.execute_write(_execute_merge)
                if result:
                    logger.info(f"Merged entity {secondary_id} into {primary_id}")
                return result
            except Exception as e:
                logger.error(f"Merge transaction failed: {e}")
                return False