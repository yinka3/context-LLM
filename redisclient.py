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
    
