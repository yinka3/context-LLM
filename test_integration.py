import asyncio
import sys
import os
import time
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("ERROR: open router api env is not set")
    sys.exit(1)

from loguru import logger
from logging_setup import setup_logging
from neo4j import GraphDatabase
from main.context import Context
from schema.dtypes import MessageData
from test_messages import MEDIUM_REFERENCE_MESSAGES, GAME_STUDIO_MESSAGES

setup_logging(log_level="INFO", log_file="test_integration.log")

TEST_MESSAGES = GAME_STUDIO_MESSAGES

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

async def run_test():
    logger.info("=" * 60)
    logger.info("VESTIGE PRODUCTION SIMULATION TEST")
    logger.info("=" * 60)

    # 1. Infrastructure Check
    if not check_redis_connection() or not check_memgraph_connection():
        return False

    try:
        # 2. Application Startup (Simulating Server Boot)
        logger.info("\n[Step 1] Initializing Application Context...")
        ctx = await Context.create(user_name="Elena", topics=["Work", "Creative", "Technical", "Business", "Projects"])
        logger.info("✓ Application started.")
        
        # 3. Ingestion Phase (Simulating User Chatting)
        logger.info("\n[Step 2] Receiving User Messages...")
        
        for i, msg in enumerate(TEST_MESSAGES, 1):
            logger.info(f"  User: {msg[:60]}...")
            msg_obj = MessageData(message=msg)
            await ctx.add(msg_obj)
            # Simulate natural typing delay (0.5s is realistic for fast typers)
            await asyncio.sleep(0.5) 
        
        # 4. Shutdown / Buffer Flush
        # In a real app, this runs when the server receives a SIGTERM signal.
        logger.info("\n[Step 3] Application Shutdown Triggered...")
        await ctx.shutdown()
        
        # 5. Verification Phase
        logger.info(f"\n[Step 4] Waiting {WAIT_FOR_PROCESSING}s for Eventual Consistency...")
        # This simulates a user coming back later to query the memory.
        await asyncio.sleep(WAIT_FOR_PROCESSING)

        logger.info("\n[Step 5] Auditing Database State...")
        
        entities = query_entities()
        relationships = query_relationships()

        logger.info(f"\n--- ENTITY AUDIT ---")
        failed_profiles = []
        
        if not entities:
            logger.warning("No entities found. Did the messages contain any names?")

        for ent in entities:
            summary = ent.get('summary', '')
            has_summary = len(summary) > 0
            
            status_icon = "✓" if has_summary else "⏳" # Hourglass means maybe profile stream is still working
            if ent['type'] == 'PERSON' and ent['name'] != 'USER' and not has_summary:
                status_icon = "✗"
                failed_profiles.append(ent['name'])

            logger.info(f"{status_icon} [{ent['type']}] {ent['name']} (Created by: {ent.get('created_by', 'structure')})")
            if has_summary:
                logger.info(f"    Summary: {summary[:80]}...")


        logger.info(f"\n--- RELATIONSHIP AUDIT ---")
        for rel in relationships:
            logger.info(f"  {rel['entity_a']} <---> {rel['entity_b']} (weight: {rel['weight']}, evidence: {rel['evidence_ids']})")
        
        # 6. Final Report
        success = True
        if failed_profiles:
            logger.error(f"\nFAILURE: The following profiles were created but never enriched: {failed_profiles}")
            logger.error("Possible causes: Profile Stream Lag, Throttling Logic Error, or LLM Failure.")
            success = False
        elif len(entities) > 0:
            logger.info("\nSUCCESS: All entities successfully ingested and enriched.")
        
        return success

    finally:
        logger.info("\nTest run complete.")

if __name__ == "__main__":
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)