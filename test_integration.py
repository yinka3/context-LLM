import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os
from dotenv import load_dotenv

from db.memgraph import MemGraphStore
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("ERROR: open router api env is not set")
    sys.exit(1)

from loguru import logger
from log.logging_setup import setup_logging
from neo4j import GraphDatabase
from main.context import Context
from schema.dtypes import MessageData
# from long_test_msgs import MAYA_MESSAGES_V2
from test_messages import ALEX_MESSAGES
setup_logging(log_level="INFO", log_file="test_integration.log")

TEST_MESSAGES = ALEX_MESSAGES

MEMGRAPH_URI = "bolt://localhost:7687"


WAIT_FOR_PROCESSING = 60 

def check_redis_connection():
    import redis
    try:
        client = redis.Redis(host='localhost', port=6379)
        client.ping()
        logger.info("✓ Redis connection OK")
        return True
    except redis.ConnectionError:
        logger.error("✗ Redis not reachable. Is docker compose up?")
        return False

def check_memgraph_connection():
    try:
        driver = GraphDatabase.driver(MEMGRAPH_URI)
        driver.verify_connectivity()
        driver.close()
        logger.info("✓ Memgraph connection OK")
        return True
    except Exception as e:
        logger.error(f"✗ Memgraph not reachable: {e}")
        return False

def query_entities():
    """Simulates the 'Read Path' - checking what data exists."""
    driver = GraphDatabase.driver(MEMGRAPH_URI)
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.id as id, e.canonical_name as name, e.type as type, 
                   e.summary as summary, e.last_profiled_msg_id as last_msg,
                   e.created_by as created_by
            ORDER BY e.id
        """)
        entities = [dict(record) for record in result]
    driver.close()
    return entities

def query_relationships():
    driver = GraphDatabase.driver(MEMGRAPH_URI)
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Entity)-[r:RELATED_TO]-(b:Entity)
            RETURN a.canonical_name as entity_a, b.canonical_name as entity_b, 
                   r.weight as weight, r.confidence as confidence,
                   r.message_ids as evidence_ids
        """)
        rels = [dict(record) for record in result]
    driver.close()
    return rels

async def wait_for_system_idle(ctx: Context):

    logger.info("  [TEST] Pausing: Waiting for system to fully digest batch...")
    
    while True:
        buffer_len = await ctx.redis_client.llen(f"buffer:{ctx.user_name}")
        
        is_processing = ctx._batch_processing_lock.locked()
        
        active_tasks = [t for t in ctx._background_tasks if not t.done()]
        
        if buffer_len == 0 and not is_processing and len(active_tasks) == 0:
            await asyncio.sleep(1)
            if not ctx._batch_processing_lock.locked() and len(ctx._background_tasks) == 0:
                logger.info("  [TEST] System is idle. State updated. Resuming...")
                break
        
        await asyncio.sleep(1)

async def run_test():
    logger.info("=" * 60)
    logger.info("VESTIGE WRITE PATH TEST")
    logger.info("=" * 60)

    if not check_redis_connection() or not check_memgraph_connection():
        return False

    user_name = "Alex"
    topics = [
        "Career & Professional Growth",
        "Post-Breakup & Dating Life",
        "Friendships & Social Circles",
        "Family Relationships",
        "Health & Fitness",
        "Daily Life & Routines",
        "Personal Growth & Self-Improvement",
    ]

    store = MemGraphStore(uri=MEMGRAPH_URI)
    executor = ThreadPoolExecutor(max_workers=4)

    try:
        logger.info("\n[Step 1] Initializing Application Context...")
        ctx = await Context.create(
            user_name=user_name,
            store=store,
            cpu_executor=executor,
            topics=topics
        )
        logger.info("✓ Application started.")
        
        logger.info("\n[Step 2] Receiving User Messages...")
        for i, msg in enumerate(TEST_MESSAGES, 1):
            logger.info(f"  User: {msg[:60]}...")
            msg_obj = MessageData(message=msg)
            await ctx.add(msg_obj)
            if i % 5 == 0:
                await wait_for_system_idle(ctx)
            else:
                await asyncio.sleep(0.1)
        
        logger.info("\n[Step 3] Application Shutdown...")
        await ctx.shutdown()

        logger.info("\n[Step 4] Auditing Database State...")
        
        entities = query_entities()
        relationships = query_relationships()

        logger.info(f"\n--- ENTITY AUDIT ({len(entities)} total) ---")
        
        if not entities:
            logger.error("No entities found. Did the messages contain any names?")
            return False

        for ent in entities:
            aliases = ent.get('aliases', []) or []
            logger.info(f"  [{ent['type']}] {ent['name']} (id: {ent['id']}, aliases: {aliases})")

        logger.info(f"\n--- RELATIONSHIP AUDIT ({len(relationships)} total) ---")
        for rel in relationships:
            logger.info(f"  {rel['entity_a']} <---> {rel['entity_b']} (weight: {rel['weight']}, evidence: {rel['evidence_ids']})")
        
        logger.info("\n--- VALIDATION ---")
        
        user_entity = next((e for e in entities if e['name'] == user_name), None)
        if not user_entity:
            logger.error(f"✗ User entity '{user_name}' not found")
            return False
        logger.info(f"✓ User entity exists")

        non_user_entities = [e for e in entities if e['name'] != user_name]
        if not non_user_entities:
            logger.error("✗ No entities extracted from messages")
            return False
        logger.info(f"✓ Extracted {len(non_user_entities)} entities from messages")

        if not relationships:
            logger.warning("⚠ No relationships found")
        else:
            logger.info(f"✓ Created {len(relationships)} relationships")

        logger.info("\nWRITE PATH TEST PASSED")
        return True

    except Exception as e:
        logger.exception(f"Test failed with error: {e}")
        return False
    finally:
        store.close()
        executor.shutdown(wait=True)

if __name__ == "__main__":
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)