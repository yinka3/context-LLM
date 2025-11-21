import logging
import time
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)

class MemGraphStore:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("", "")):

        self.driver: Driver = GraphDatabase.driver(uri, auth=auth)
        self._setup_schema()
    

    def close(self):
        if self.driver:
            self.driver.close()
    

    def _setup_schema(self):
        """
        Create indices and constraints to ensure performance and data integrity.
        """
        queries = [
            "CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE;",
            "CREATE INDEX ON :Entity(canonical_name);"
        ]
        
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.debug(f"Schema setup note: {e}")
        logger.info("Memgraph schema indices verified.")