import logging
import threading
import time
from readerwriterlock import rwlock
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

from graph_driver import KnowGraph
from redisclient import RedisClient
from schema.common_pb2 import Entity, Relationship
from schema.graph_messages_pb2 import *

logger = logging.getLogger(__name__)

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
        
        self.channel_handlers = {
            'graph:add-entity': self._handle_add_entity,
            'graph:add-relationship': self._handle_add_relationship
        }

    
    def start(self):
        logger.info("Starting GraphBuilder...")
        
        channels = list(self.channel_handlers.keys())
        self.pubsub.subscribe(*channels)
        logger.info(f"Subscribed to channels: {channels}")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running.set()
        self._message_loop()
        
    def stop(self):
        logger.info("Stopping GraphBuilder...")
        self.running.clear()
        self.shutdown_event.set()
        self.pubsub.close()
        self.executor.shutdown(wait=True, timeout=30)
        
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
                message = self.pubsub.get_message(timeout=1.0)
                
                if message is None:
                    continue
                    
                if message['type'] == 'message':
                    self.executor.submit(self._process_message, message)
                    
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                if not self.running.is_set():
                    break
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


                

def main():

    import logging_setup
    logging_setup.setup_logging(log_file="graph_builder_service.log")
    service = GraphBuilder(max_workers=4)
    
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        service.stop()


if __name__ == "__main__":
    main()


    
