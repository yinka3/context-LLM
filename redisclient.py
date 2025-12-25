import redis
import redis.asyncio as async_redis

REDIS_HOST = 'localhost'
REDIS_PORT = 6379

class AsyncRedisClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            pool = async_redis.ConnectionPool.from_url(
                url=f"redis://{REDIS_HOST}:{REDIS_PORT}",
                decode_responses=True,
                max_connections=10
            )
            cls._instance.client = async_redis.Redis(connection_pool=pool)
        return cls._instance

    def get_client(self) -> async_redis.Redis:
        return self.client

class SyncRedisClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            pool = redis.ConnectionPool.from_url(
                url=f"redis://{REDIS_HOST}:{REDIS_PORT}",
                decode_responses=True,
                max_connections=5
            )
            cls._instance.client = redis.Redis(connection_pool=pool)
        return cls._instance

    def get_client(self) -> redis.Redis:
        return self.client