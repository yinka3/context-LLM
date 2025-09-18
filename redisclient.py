import json
import numpy as np
import redis
import logging

logger = logging.getLogger(__name__)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

class RedisClient:

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db_num=REDIS_DB):

        try:
            self.client = redis.Redis(host=host, port=port, db=db_num, decode_responses=True)
            self.client.ping()
            logger.info(f"Redis connected at {host}:{port}")
        except redis.exceptions.ConnectionError:
            logger.error(f"Redis not connected at {host}:{port}")
            self.client = None
    
    def store_entity_embedding(self, entity_id: int, embedding: np.ndarray, metadata: dict):
        """Store entity embedding for RAG retrieval"""
        self.client.hset(
            "entity_embeddings",
            f"ent_{entity_id}",
            embedding.tobytes()
        )
        self.client.hset(
            "entity_metadata", 
            f"ent_{entity_id}",
            json.dumps(metadata)
        )

    def get_all_embeddings(self):
        """Retrieve all embeddings for similarity search"""
        embeddings = self.client.hgetall("entity_embeddings")
        return {
            ent_id: np.frombuffer(emb_bytes, dtype=np.float32) 
            for ent_id, emb_bytes in embeddings.items()
        }
