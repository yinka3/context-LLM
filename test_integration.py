import asyncio
import subprocess
import time
import sys
import os
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("ERROR: open router api env is not set")
    sys.exit(1)

from loguru import logger
from neo4j import GraphDatabase
from main.context import Context
from graph.graphbuilder import GraphBuilder
from schema.dtypes import *

TEST_MESSAGES = [
    "I had coffee with Jake from Stripe today. He's working on their payments API.",
    "My sister Sarah called from Boston. She's thinking about visiting next month.",
]

MEMGRAPH_URI = "bolt://localhost:7687"
WAIT_FOR_PROCESSING = 60


def check_redis_connection():
    """Verify Redis is reachable."""
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
    """Verify Memgraph is reachable."""
    try:
        driver = GraphDatabase.driver(MEMGRAPH_URI)
        driver.verify_connectivity()
        driver.close()
        logger.info("✓ Memgraph connection OK")
        return True
    except Exception as e:
        logger.error(f"✗ Memgraph not reachable: {e}")
        return False


def clear_memgraph():
    """Wipe all data from Memgraph for a clean test."""
    driver = GraphDatabase.driver(MEMGRAPH_URI)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    logger.info("✓ Memgraph cleared")


def query_entities():
    """Fetch all entities from Memgraph."""
    driver = GraphDatabase.driver(MEMGRAPH_URI)
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.id as id, e.canonical_name as name, e.type as type, e.summary as summary
            ORDER BY e.id
        """)
        entities = [dict(record) for record in result]
    driver.close()
    return entities


def query_relationships():
    """Fetch all relationships from Memgraph."""
    driver = GraphDatabase.driver(MEMGRAPH_URI)
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Entity)-[r:RELATED_TO]-(b:Entity)
            RETURN a.canonical_name as source, b.canonical_name as target, 
                   r.verb as relation, r.confidence as confidence
        """)
        rels = [dict(record) for record in result]
    driver.close()
    return rels


async def run_test():
    logger.info("=" * 60)
    logger.info("VESTIGE INTEGRATION TEST")
    logger.info("=" * 60)

    logger.info("\n[1/6] Checking infrastructure...")
    if not check_redis_connection() or not check_memgraph_connection():
        logger.error("Infrastructure check failed. Exiting.")
        return False

    logger.info("\n[2/6] Clearing Memgraph for clean test...")
    clear_memgraph()

    logger.info("\n[3/6] Starting GraphBuilder consumer...")
    builder_process = subprocess.Popen(
        [sys.executable, "-c", """
import sys
sys.path.insert(0, ".")
from graph.graphbuilder import GraphBuilder
builder = GraphBuilder()
builder.start()
"""],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
    )

    time.sleep(2)

    if builder_process.poll() is not None:
        # something went wrong
        stdout, stderr = builder_process.communicate()
        logger.error(f"GraphBuilder failed to start: {stderr.decode()}")
        return False

    logger.info("✓ GraphBuilder started (PID: {})".format(builder_process.pid))
    
    try:
        logger.info("\n[4/6] Initializing Context (loading NLP models)...")
        ctx = await Context.create(user_name="TestUser", topics=["Work", "Family"])
        logger.info("✓ Context initialized")
        
        logger.info("\n[5/6] Sending test messages...")
        for i, msg in enumerate(TEST_MESSAGES, 1):
            logger.info(f"  Message {i}: {msg[:50]}...")
            msg = MessageData(message=msg)
            await ctx.add(msg)
            await asyncio.sleep(1)
        
        logger.info(f"\n    Waiting {WAIT_FOR_PROCESSING}s for GraphBuilder to process...")
        await asyncio.sleep(WAIT_FOR_PROCESSING)

        logger.info("\n[6/6] Verifying results in Memgraph...")
        
        entities = query_entities()
        relationships = query_relationships()

        logger.info(f"\n{'=' * 60}")
        logger.info("RESULTS")
        logger.info(f"{'=' * 60}")
        
        logger.info(f"\nEntities found: {len(entities)}")
        for ent in entities:
            logger.info(f"  - [{ent['type']}] {ent['name']}: {ent.get('summary', 'No summary')[:60]}")

        logger.info(f"\nRelationships found: {len(relationships)}")
        for rel in relationships:
            logger.info(f"  - {rel['source']} --[{rel['relation']}]--> {rel['target']}")
        
        entity_ids = [e['id'] for e in entities]
        success = True

        if len(entities) == 0:
            logger.error("✗ No entities created - something is wrong!")
            success = False
        else:
            logger.info("✓ Entities were created")

        if None in entity_ids:
            logger.error("✗ Some entities have None IDs")
            success = False
        elif 0 in entity_ids:
            logger.error("✗ Some entities have ID=0 (invalid)")
            success = False
        elif len(entity_ids) != len(set(entity_ids)):
            logger.error("✗ Duplicate entity IDs found")
            success = False
        else:
            logger.info(f"✓ All entity IDs valid and unique: {entity_ids}")

        entity_names = [e['name'].lower() for e in entities]
        expected = ['jake', 'stripe', 'sarah', 'boston']
        found = [name for name in expected if any(name in en for en in entity_names)]
        
        if found:
            logger.info(f"✓ Found expected entities: {found}")
        else:
            logger.warning(f"⚠ Did not find expected entities: {expected}")

        await ctx.shutdown()
        
        return success
    finally:
        logger.info("\nCleaning up...")
        builder_process.terminate()
        builder_process.wait(timeout=5)
        logger.info("✓ GraphBuilder terminated")


if __name__ == "__main__":
    success = asyncio.run(run_test())
    
    print("\n" + "=" * 60)
    if success:
        print("✓ INTEGRATION TEST PASSED")
    else:
        print("✗ INTEGRATION TEST FAILED")
    print("=" * 60)
    
    sys.exit(0 if success else 1)