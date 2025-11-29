import redis.asyncio as redis
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = 'localhost'
REDIS_PORT = 6379


class RedisClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            pool = redis.ConnectionPool.from_url(
                url=f"redis://{REDIS_HOST}:{REDIS_PORT}",
                decode_responses=True
            )
            cls._instance.client = redis.Redis(connection_pool=pool)
        return cls._instance

    def get_client(self):
        return self.client
