import os
import socket
import sys
import time
import signal
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from redis import exceptions
from redisclient import SyncRedisClient
from schema.common_pb2 import BatchMessage, MessageType
from graph.memgraph import MemGraphStore


STREAM_KEY = "stream:ai_response"
CONSUMER_GROUP = "group:graph_builders"
DEAD_QUEUE = 'stream:builder_dead_letters'
CONSUMER_NAME = f"builder-{socket.gethostname()}-{os.getpid()}"

class GraphBuilder:

    def __init__(self, max_workers: int = 2):
        self.redis_client = SyncRedisClient().get_client()
        
        self.store = MemGraphStore() 
        
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="graph-worker")
        self.processed_messages = 0
        self.failed_messages = 0
        
        self._ensure_consumer_group()

    def _ensure_consumer_group(self):
        try:
            logger.info(f"Ensuring consumer group '{CONSUMER_GROUP}' exists.")
            self.redis_client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id='$', mkstream=True)
        except exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group '{CONSUMER_GROUP}' already exists.")
            else:
                logger.critical(f"Failed to create consumer group: {e}")
                sys.exit(1)
    

    def start(self):
        logger.info("Starting GraphBuilder service...")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._message_loop()

    def stop(self):
        logger.info("Stopping GraphBuilder...")
        self.running = False
        self.store.close()
        self.executor.shutdown(wait=True)
        logger.info(f"Service stopped. Processed: {self.processed_messages}, Failed: {self.failed_messages}")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _message_loop(self):

        logger.info(f"Listening on {STREAM_KEY}")

        while self.running:
            try:
                response = self.redis_client.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {STREAM_KEY: '>'},
                    count=5,
                    block=1000
                )

                if not response: continue

                for _, messages in response:
                    for msg_id, msg_data in messages:
                        self.executor.submit(self._process_message, msg_id, msg_data)
            
            except exceptions.ConnectionError as e:
                logger.error(f"Redis connection lost: {e}. Retrying...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in loop: {e}", exc_info=True)
                time.sleep(1)
    
    def _process_message(self, stream_id, msg_data):

        try:
            batch_msg = BatchMessage()
            batch_msg.ParseFromString(msg_data[b'data'])

            if batch_msg.type == MessageType.USER_MESSAGE and batch_msg.message_id < 1:
                logger.warning(f"Skipping USER_MESSAGE {stream_id}: invalid message_id")
                self.redis_client.xack(STREAM_KEY, CONSUMER_GROUP, stream_id)
                return

            current_msg_ref = f"msg_{batch_msg.message_id}"

            entities = []
            relationships = []
            for entity in batch_msg.list_ents:

                if entity.id is None:
                    continue
                    
                entities.append({
                    "id": entity.id,
                    "canonical_name": entity.canonical_name,
                    "type": entity.type,
                    "confidence": entity.confidence,
                    "aliases": list(entity.aliases),
                    "summary": entity.summary, 
                    "topic": entity.topic,
                    "embedding": list(entity.embedding)
                })
            
            for rel in batch_msg.list_relations:

                relationships.append({
                "source_name": rel.source_text,
                "target_name": rel.target_text,
                "relation": rel.relation,
                "message_id": current_msg_ref,
                "confidence": rel.confidence
            })
            
            is_user_msg = (batch_msg.type == MessageType.USER_MESSAGE)
            self.store.write_batch(entities, relationships, is_user_message=is_user_msg)
            self.redis_client.xack(STREAM_KEY, CONSUMER_GROUP, stream_id)
            self.processed_messages += 1
        except Exception as e:
            logger.error(f"Failed to process message {stream_id}: {e}", exc_info=True)
            try:
                self.redis_client.xadd(DEAD_QUEUE, {'original_id': stream_id, 'data': msg_data[b'data']})
                self.redis_client.xack(STREAM_KEY, CONSUMER_GROUP, stream_id)
                self.failed_messages += 1
            except Exception as dlq_error:
                logger.critical(f"Failed to move to DLQ: {dlq_error}")

if __name__ == "__main__":
    builder = GraphBuilder(max_workers=2)
    builder.start()