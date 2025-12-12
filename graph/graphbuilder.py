import os
import socket
import sys
import time
import signal
import traceback
from loguru import logger
from redis import exceptions
from logging_setup import setup_logging
from redisclient import SyncRedisClient
from schema.common_pb2 import BatchMessage, MessageType
from graph.memgraph import MemGraphStore


STREAM_KEY_STRUCTURE = "stream:structure"
STREAM_KEY_PROFILE = "stream:profile"
CONSUMER_GROUP = "group:graph_builders"
DEAD_QUEUE = 'stream:builder_dead_letters'
CONSUMER_NAME = f"builder-{socket.gethostname()}-{os.getpid()}"
setup_logging(log_level="DEBUG", log_file="graph_builder.log")

class GraphBuilder:

    def __init__(self):
        self.redis_client = SyncRedisClient().get_client()
        
        self.store = MemGraphStore() 
        
        self.running = True
        self.processed_messages = 0
        self.failed_messages = 0
        
        self._ensure_consumer_group()

    def _ensure_consumer_group(self):
        try:
            logger.info(f"Ensuring consumer group '{CONSUMER_GROUP}' exists.")
            self.redis_client.xgroup_create(STREAM_KEY_STRUCTURE, CONSUMER_GROUP, id='0', mkstream=True)
            self.redis_client.xgroup_create(STREAM_KEY_PROFILE, CONSUMER_GROUP, id='0', mkstream=True)
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
        deleted = self.store.cleanup_null_entities()
        if deleted:
            logger.info(f"Final cleanup: removed {deleted} null-type entities")
        self.store.close()
        logger.info(f"Service stopped. Processed: {self.processed_messages}, Failed: {self.failed_messages}")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _message_loop(self):

        logger.info(f"Listening on {STREAM_KEY_STRUCTURE}, {STREAM_KEY_PROFILE}")

        while self.running:
            try:
                response = self.redis_client.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {
                        STREAM_KEY_STRUCTURE: '>',
                        STREAM_KEY_PROFILE: '>'
                    },
                    count=10,
                    block=1000
                )

                if not response: continue

                for stream_key, messages in response:
                    for msg_id, msg_data in messages:
                        self._process_message(stream_key, msg_id, msg_data)
            
            except exceptions.ConnectionError as e:
                logger.error(f"Redis connection lost: {e}. Retrying...")
                time.sleep(5)
            except exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    logger.warning("Consumer group lost, recreating...")
                    self._ensure_consumer_group()
                    logger.info("Catching up on missed messages...")
                    catchup = self.redis_client.xreadgroup(
                        CONSUMER_GROUP,
                        CONSUMER_NAME,
                        {
                        STREAM_KEY_STRUCTURE: '>',
                        STREAM_KEY_PROFILE: '>'
                        },
                        count=100
                    )
                    if catchup:
                        for stream_key, messages in catchup:
                            for msg_id, msg_data in messages:
                                self._process_message(stream_key, msg_id, msg_data)
                    logger.info("Consumer group recreated, resuming...")
                else:
                    logger.error(f"Redis error: {e}")
                    time.sleep(1)
            except Exception as e:
                import traceback
                logger.error(f"Unexpected error in loop: {e}\n{traceback.format_exc()}")
                time.sleep(1)
    
    def _process_message(self, stream_key, stream_id, msg_data):

        try:
            batch_msg = BatchMessage()
            batch_msg.ParseFromString(msg_data[b'data'])

            logger.debug(f"Processing {batch_msg.type} from {stream_key}")

            if batch_msg.type == MessageType.USER_MESSAGE:

                entities = []
                relationships = []

                
                for entity in batch_msg.list_ents:
                        
                    entities.append({
                        "id": entity.id,
                        "canonical_name": entity.canonical_name,
                        "type": entity.type,
                        "confidence": entity.confidence,
                        "summary": entity.summary, 
                        "topic": entity.topic,
                        "embedding": list(entity.embedding)
                    })
                
                for rel in batch_msg.list_relations:

                    relationships.append({
                        "entity_a": rel.entity_a,
                        "entity_b": rel.entity_b,
                        "message_id": f"msg_{rel.message_id}",
                        "confidence": rel.confidence
                    })
                
                self.store.write_batch(entities, relationships, is_user_message=True)
                self.redis_client.xack(stream_key, CONSUMER_GROUP, stream_id)
                self.processed_messages += 1

                self.store.cleanup_null_entities()

            elif batch_msg.type == MessageType.PROFILE_UPDATE:
                logger.info(f"Processing PROFILE_UPDATE message {stream_id}")
            
                for entity in batch_msg.list_ents:
                    if entity.id == 0:
                        logger.warning(f"Skipping PROFILE_UPDATE: entity has no id")
                        continue
                    
                    logger.info(f"PROFILE_UPDATE: id={entity.id}, name={entity.canonical_name}, summary_len={len(entity.summary)}, embedding_len={len(list(entity.embedding))}, embedding_type={type(list(entity.embedding))}")
                    self.store.update_entity_profile(
                        entity_id=entity.id,
                        canonical_name=entity.canonical_name,
                        summary=entity.summary,
                        embedding=list(entity.embedding),
                        last_msg_id=entity.last_profiled_msg_id
                    )
                
                self.redis_client.xack(stream_key, CONSUMER_GROUP, stream_id)
                self.processed_messages += 1
                logger.info(f"Profile update batch processed for {len(batch_msg.list_ents)} entities")
            elif batch_msg.type == MessageType.SYSTEM_ENTITY:
                logger.info(f"Processing SYSTEM_ENTITY message {stream_id}")
                
                entities = []
                for entity in batch_msg.list_ents:
                    entities.append({
                        "id": entity.id,
                        "canonical_name": entity.canonical_name,
                        "type": entity.type,
                        "confidence": entity.confidence,
                        "summary": entity.summary,
                        "topic": entity.topic,
                        "embedding": list(entity.embedding)
                    })
                
                self.store.write_batch(entities, [], is_user_message=False)
                self.redis_client.xack(stream_key, CONSUMER_GROUP, stream_id)
                self.processed_messages += 1
                logger.info(f"Created {len(entities)} system entities")
            else:
                logger.warning(f"Unknown message type: {batch_msg.type}")
                self.redis_client.xack(stream_key, CONSUMER_GROUP, stream_id)
        except Exception as e:
            logger.error("Failed to process message {}: {}", stream_id, str(e), exc_info=True)
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            try:
                self.redis_client.xadd(DEAD_QUEUE, {'original_id': stream_id, 'data': msg_data[b'data']})
                self.redis_client.xack(stream_key, CONSUMER_GROUP, stream_id)
                self.failed_messages += 1
            except Exception as dlq_error:
                logger.critical(f"Failed to move to DLQ: {dlq_error}")


if __name__ == "__main__":
    builder = GraphBuilder()
    builder.start()