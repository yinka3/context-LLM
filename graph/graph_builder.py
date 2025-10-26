import logging
import os
import socket
import threading
import time
import logging_setup
from redis import exceptions
from readerwriterlock import rwlock
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

from graph_driver import KnowGraph
from redisclient import RedisClient
from schema.common_pb2 import Entity, Relationship, BatchMessage

logging_setup.setup_logging(log_file="graph_builder_service.log")
logger = logging.getLogger(__name__)

STREAM_KEY = "stream:ai_response"
CONSUMER_GROUP = "group:parsers"
CONSUMER_NAME = f"builder-{socket.gethostname()}-{os.getpid()}"
DEAD_QUEUE = 'stream:parser_dead_letters'

class GraphBuilder:
    def __init__(self, max_workers: int = 4):
        self.driver: KnowGraph = KnowGraph()
        self.redis_client = RedisClient()
        self.pubsub = self.redis_client.client.pubsub()
        
        self.graph_lock = rwlock.RWLockFair()
        self.running = threading.Event()
        self.shutdown_event = threading.Event()
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="graph-worker")
        self.processed_messages = 0
        self.failed_messages = 0
        
        try:
            logger.info(f"Ensuring consumer group '{CONSUMER_GROUP}' exists for stream '{STREAM_KEY}'.")
            self.redis_client.client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id='$', mkstream=True)
        except exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.warning(f"Consumer group '{CONSUMER_GROUP}' already exists.")
            else:
                logger.error(f"Failed to create consumer group: {e}")
                sys.exit(1)

    
    def start(self):
        logger.info(f"Starting GraphBuilder service as consumer '{CONSUMER_NAME}'.")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running.set()
        self._message_loop()
        
    def stop(self):
        logger.info("Stopping GraphBuilder...")
        self.running.clear()
        self.shutdown_event.set()
        self.executor.shutdown(wait=True)
        
        logger.info(f"Service stopped. Processed: {self.processed_messages}, Failed: {self.failed_messages}")
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _message_loop(self):
        """Main message processing loop"""
        logger.info("Starting message processing loop...")
        
        while self.running.is_set():
            try:
                response = self.redis_client.client.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {STREAM_KEY: '>'}, 
                    count=10,
                    block=1000
                )
            
                if not response:
                    continue

                for _, messages in response:
                    for msg_id, msg_data in messages:
                        self.executor.submit(self._process_message, msg_id, msg_data)
                    
            except exceptions.ConnectionError as e:
                logger.error(f"Redis connection lost: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"An unexpected error occurred in the message loop: {e}", exc_info=True)
                time.sleep(1)
    
    def _handle_add_entity(self, data: bytes):
        with self.graph_lock.gen_wlock():
            try:
                entity_msg = Entity()
                entity_msg.ParseFromString(data)
                
                entity_data = {
                    "id": entity_msg,  # Generate ID
                    "name": entity_msg.text,
                    "type": entity_msg.type,
                    "confidence": entity_msg.confidence
                }
                
                self.driver.add_entity(entity_data)
                logger.debug(f"Added entity: {entity_data['name']} (ID: {entity_data['id']})")
                
            except Exception as e:
                logger.error(f"Failed to add entity: {e}")
                raise
    
    def _handle_add_relation(self, data: bytes):

        with self.graph_lock.gen_wlock():
            try:

                relation_msg = Relationship()
                relation_msg.ParseFromString(data)

                v1 = relation_msg.source_text
                v2 = relation_msg.target_text
                

                edge_data = {
                    "relation": relation_msg.relation,
                    "confidence": relation_msg.confidence
                }

            except Exception as e:
                logger.error(f"Failed to add relationship: {e}")
    
    #TODO: When switching to a mark and process flow, add a needs_review flag in protobuf message and also something similar as node property
    def _is_batch_valid(self, batch_msg: BatchMessage):

        entity_texts = {e.text for e in batch_msg.list_ents}
        for entity in batch_msg.list_ents:
            if not entity.text.strip() or not entity.type:
                logger.error("Missing entity text or type")
                return False
            
            #NOTE I might still add low confidence but maybe mark it as low confidence?
            if entity.confidence < 0.4:
                logger.error("Low confidence entity detected")
                return False
        
        for rel in batch_msg.list_relations:

            if not rel.source_text.strip() or not rel.target_text.strip() or rel.relation.strip():
                logger.error("Relationship missing source, target, or relation")
                return False
            
            
            if rel.source_text not in entity_texts or rel.target_text not in entity_texts:
                logger.error("Misaligned entity representation")
                return False
            
            #NOTE I might still add low confidence but maybe mark it as low confidence?
            if rel.confidence < 0.4:
                logger.error("Low confidence relationship detected")
                return False
        
        return True

    def _process_message(self, msg_id, msg_data):

        batch_msg = BatchMessage()
        batch_msg.ParseFromString(msg_data[b'data'])

        if not self._is_batch_valid(batch_msg):
            logger.warning(f"Invalid BatchMessage {msg_id}. Moving to DLQ")
            self.redis_client.client.xadd(DEAD_QUEUE, {'original_id': msg_id, 'data': msg_data[b'data']})
            self.redis_client.client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
            self.failed_messages += 1
            return
        
        

                

def main():
    service = GraphBuilder(max_workers=4)
    
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        service.stop()


if __name__ == "__main__":
    main()


    
