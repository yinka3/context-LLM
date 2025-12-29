import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import time
from dotenv import load_dotenv

from db.memgraph import MemGraphStore
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

from loguru import logger
from log.logging_setup import setup_logging
from neo4j import GraphDatabase
from main.context import Context
from schema.dtypes import MessageData
from test_msgs import SARAH_MESSAGES_EXTENDED
from agent.loop import run as run_read_path

setup_logging(log_level="INFO", log_file="test_integration.log")

MEMGRAPH_URI = "bolt://localhost:7687"
SESSION_BREAK_SECONDS = 40

# ============================================================
# CONNECTION CHECKS
# ============================================================

def check_redis_connection():
    import redis
    try:
        client = redis.Redis(host='localhost', port=6379)
        client.ping()
        logger.info("✓ Redis connection OK")
        return True
    except redis.ConnectionError:
        logger.error("✗ Redis not reachable")
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

# ============================================================
# GRAPH QUERIES
# ============================================================

def query_entities():
    driver = GraphDatabase.driver(MEMGRAPH_URI)
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.id as id, e.canonical_name as name, e.type as type, 
                   e.summary as summary
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
                   r.weight as weight, r.message_ids as evidence_ids
        """)
        rels = [dict(record) for record in result]
    driver.close()
    return rels

# ============================================================
# WAIT HELPERS
# ============================================================

async def wait_for_system_idle(ctx: Context):
    logger.info("  [TEST] Waiting for system idle...")
    while True:
        buffer_len = await ctx.redis_client.llen(f"buffer:{ctx.user_name}")
        is_processing = ctx._batch_processing_lock.locked()
        active_tasks = [t for t in ctx._background_tasks if not t.done()]
        
        if buffer_len == 0 and not is_processing and len(active_tasks) == 0:
            await asyncio.sleep(1)
            if not ctx._batch_processing_lock.locked():
                logger.info("  [TEST] System idle")
                break
        await asyncio.sleep(1)

async def wait_for_dirty_clear(ctx: Context, timeout: int = 120):
    dirty_key = f"dirty_entities:{ctx.user_name}"
    start = time.time()
    while time.time() - start < timeout:
        count = await ctx.redis_client.scard(dirty_key)
        if count == 0:
            logger.info("  [TEST] Dirty entities cleared")
            return True
        await asyncio.sleep(2)
    logger.warning("  [TEST] Timeout waiting for dirty clear")
    return False

# ============================================================
# READ PATH TEST CASES
# ============================================================

READ_TEST_CASES = [
    {
        "query": "Who is James?",
        "description": "Simple entity lookup - work friend",
        "expected_tools": ["get_profile"],
    },
    {
        "query": "What's going on with mom's health?",
        "description": "Topic search - health arc",
        "expected_tools": ["search_messages"],
    },
    {
        "query": "What happened at work recently?",
        "description": "Broad topic search",
        "expected_tools": ["search_messages"],
    },
    {
        "query": "Who is Tom?",
        "description": "Core family member lookup",
        "expected_tools": ["get_profile"],
    },
    {
        "query": "What's been happening with Ben lately?",
        "description": "Recent activity for specific entity - swim lessons, daycare, etc.",
        "expected_tools": ["get_activity", "get_profile"],
    },
    {
        "query": "Where did we go for date night?",
        "description": "Entity lookup for place - Centro restaurant",
        "expected_tools": ["search_messages"],
    },
    {
        "query": "Where did we go for date night?",
        "description": "Entity lookup for place - Centro restaurant",
        "expected_tools": ["search_messages"],
    },
    {
        "query": "thinking about mom",
        "description": "Vague statement - should retrieve mom's health arc, PT, Dr. Hoffman",
        "expected_tools": ["search_messages", "get_profile"],
    },
    {
        "query": "how's everything going",
        "description": "Generic check-in - should synthesize recent state or clarify",
        "expected_tools": ["request_clarification"],
    },
    {
        "query": "between the kids and work and mom i dont even know anymore",
        "description": "Multi-topic dump - should pick dominant thread or clarify",
        "expected_tools": ["request_clarification"],
    },
        {
        "query": "tom's parents are a lot sometimes you know",
        "description": "In-laws context - Janet and Rick, screen time comments",
        "expected_tools": ["search_messages", "get_profile"],
    },
    {
        "query": "the whole promotion thing still feels weird",
        "description": "No names - could be Sarah's senior ops manager role or James being passed over",
        "expected_tools": ["search_messages"],
    },
    {
        "query": "what happened when ben got sick that time",
        "description": "References batch 1 - stomach bug arc, completely outside window",
        "expected_tools": ["search_messages"],
    },
    {
        "query": "remember that thai place james and i went to",
        "description": "Grove Street lunch after Q1 report - way outside window",
        "expected_tools": ["search_messages"],
    }
]

async def run_read_tests(ctx: Context):
    logger.info("\n" + "=" * 60)
    logger.info("READ PATH TESTS")
    logger.info("=" * 60)
    
    results = []
    
    for i, test in enumerate(READ_TEST_CASES, 1):
        logger.info(f"\n--- Test {i}/{len(READ_TEST_CASES)}: {test['description']} ---")
        logger.info(f"Query: {test['query']}")
        
        try:
            result = await run_read_path(
                user_query=test["query"],
                user_name=ctx.user_name,
                conversation_history=[],
                hot_topics=[],
                active_topics=ctx.active_topics,
                llm=ctx.llm,
                store=ctx.store,
                ent_resolver=ctx.ent_resolver,
                redis_client=ctx.redis_client
            )
            
            logger.info(f"Status: {result['status']}")
            logger.info(f"Tools used: {result['tools_used']}")
            logger.info(f"Response: {result['response'][:200]}..." if len(result.get('response', '')) > 200 else f"Response: {result.get('response', 'N/A')}")

            tools_match = any(t in result['tools_used'] for t in test["expected_tools"])

            results.append({
                "query": test["query"],
                "passed": result['status'] == "complete" and tools_match,
                "tools_used": result['tools_used'],
                "expected_tools": test["expected_tools"],
                "response_length": len(result.get('response', ''))
            })
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            results.append({
                "query": test["query"],
                "passed": False,
                "error": str(e)
            })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("READ PATH SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results if r["passed"])
    logger.info(f"Passed: {passed}/{len(results)}")
    
    for r in results:
        status = "✓" if r["passed"] else "✗"
        logger.info(f"  {status} {r['query'][:40]}...")
        if not r["passed"] and "error" in r:
            logger.info(f"      Error: {r['error']}")
        elif not r["passed"]:
            logger.info(f"      Expected: {r['expected_tools']}, Got: {r.get('tools_used', [])}")
    
    return passed == len(results)

# ============================================================
# MAIN TEST
# ============================================================

async def run_test():
    logger.info("=" * 60)
    logger.info("VESTIGE INTEGRATION TEST (WRITE + READ)")
    logger.info("=" * 60)

    if not check_redis_connection() or not check_memgraph_connection():
        return False

    user_name = "Sarah"
    topics = [
        "Career & Workplace Management",
        "Parenting & Family Dynamics",
        "Multigenerational Caregiving",
        "Midlife Identity & Aging",
        "Daily Stress & Household Logistics",
        "Friendships & Social Observations",
        "Health, Fitness & Self-Care",
    ]

    store = MemGraphStore(uri=MEMGRAPH_URI)
    executor = ThreadPoolExecutor(max_workers=4)

    try:
        # ========================================
        # PHASE 1: WRITE PATH
        # ========================================
        logger.info("\n[Phase 1] WRITE PATH")
        logger.info("=" * 40)
        
        logger.info("\n[Step 1.1] Initializing Context...")
        ctx = await Context.create(
            user_name=user_name,
            store=store,
            cpu_executor=executor,
            topics=topics
        )
        logger.info("✓ Context initialized")
        
        logger.info("\n[Step 1.2] Ingesting messages...")
        
        sessions = [
            SARAH_MESSAGES_EXTENDED[0:20],
            SARAH_MESSAGES_EXTENDED[20:50],
            SARAH_MESSAGES_EXTENDED[50:60],
            SARAH_MESSAGES_EXTENDED[60:80],
            SARAH_MESSAGES_EXTENDED[80:100]
        ]
        
        for session_num, session_msgs in enumerate(sessions, 1):
            logger.info(f"\n  --- Session {session_num} ({len(session_msgs)} messages) ---")
            
            for i, msg in enumerate(session_msgs, 1):
                preview = msg[:50].replace('\n', ' ')
                logger.info(f"  [{i}] {preview}...")
                await ctx.add(MessageData(message=msg))
                await asyncio.sleep(0.1)
                
                if i % 5 == 0:
                    await wait_for_system_idle(ctx)
            
            logger.info(f"  Session {session_num} complete")
            await wait_for_system_idle(ctx)
            
            if session_num < len(sessions):
                await wait_for_dirty_clear(ctx, timeout=90)
                await asyncio.sleep(SESSION_BREAK_SECONDS)
        
        logger.info("\n[Step 1.3] Waiting for background jobs...")
        await wait_for_dirty_clear(ctx, timeout=45)
        
        # Audit write results
        logger.info("\n[Step 1.4] Write Path Audit...")
        entities = query_entities()
        relationships = query_relationships()
        
        logger.info(f"  Entities: {len(entities)}")
        logger.info(f"  Relationships: {len(relationships)}")
        
        if len(entities) < 5:
            logger.error("✗ Too few entities extracted")
            return False
        logger.info("✓ Write path complete")
        
        # ========================================
        # PHASE 2: READ PATH
        # ========================================
        logger.info("\n[Phase 2] READ PATH")
        logger.info("=" * 40)
        
        read_success = await run_read_tests(ctx)
        
        # ========================================
        # CLEANUP
        # ========================================
        logger.info("\n[Phase 3] Shutdown...")
        await ctx.shutdown()
        
        # ========================================
        # FINAL RESULT
        # ========================================
        logger.info("\n" + "=" * 60)
        if read_success:
            logger.info("ALL TESTS PASSED")
        else:
            logger.info("SOME TESTS FAILED")
        logger.info("=" * 60)
        
        return read_success

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return False
    finally:
        store.close()
        executor.shutdown(wait=True)

if __name__ == "__main__":
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)