from datetime import timedelta
import logging
import os
from pathlib import Path
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
from schema.common_pb2 import BatchMessage
from timeloop import Timeloop
from main import entity_resolve as ER
logging_setup.setup_logging(log_file="graph_builder_service.log")
logger = logging.getLogger(__name__)

STREAM_KEY = "stream:ai_response"
CONSUMER_GROUP = "group:parsers"
DEAD_QUEUE = 'stream:parser_dead_letters'
CONSUMER_NAME = f"builder-{socket.gethostname()}-{os.getpid()}"


class GraphBuilder:
    WAIT_TIME_SECONDS = 1

    def __init__(self, max_workers: int = 4):
        self.driver: KnowGraph = KnowGraph()
        self.redis_client = RedisClient()
        self.pubsub = self.redis_client.client.pubsub()
        self.timer: 'Timeloop' = Timeloop()

        self.persistence_dir = Path("./graph_data")
        self.persistence_dir.mkdir(exist_ok=True)
        self.graph_path = self.persistence_dir / "gt_graph.gt"

        self.graph_lock = rwlock.RWLockFair()
        self.save_lock = threading.Lock()
        self.running = threading.Event()
        
        self.save_thread = None
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
        
        @self.timer.job(interval=timedelta(minutes=30))
        def scheduler():
            self._starting_saving_process()
        
        self.resolver = ER.EntityResolver()
    

    def _to_disk(self, snapshot):

        tmp_path = self.graph_path.with_suffix(".tmp")

        try:
            snapshot.save(str(tmp_path), fmt="gt")
            os.rename(tmp_path, self.graph_path)
        except Exception as e:
            logger.error(f"Failed to save graph to disk: {e}", exc_info=True)


    def _starting_saving_process(self, is_shutdown=False):
        
        if not self.save_lock.acquire(blocking=False):
            logger.info("Save operation in progress already")
            return False
    
        logger.info("Snapshotting graph...")
        try:
            with self.graph_lock.gen_rlock():
                snapshot = self.driver.snapshot_graph()
            if is_shutdown:
                self._to_disk(snapshot)
            else:
                self.save_thread = threading.Thread(target=self._to_disk, args=(snapshot,))
                self.save_thread.start()
        except Exception as e:
            logger.error(f"Error during save initiation: {e}", exc_info=True)
            return False 
        finally:
                self.save_lock.release()
    
    
    def start(self):
        
        self.driver.load_graph(self.graph_path)

        logger.info("Starting GraphBuilder service...")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.timer.start()
        self.running.set()
        self._message_loop()
        
    def stop(self):
        logger.info("Stopping GraphBuilder...")
        self.running.clear()

        if self.timer:
            logger.info("Stopping Timeloop scheduler...")
            self.timer.stop()

        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join()
        
        self._starting_saving_process(is_shutdown=True)

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
            logger.warning(f"Invalid BatchMessage {msg_id}. Moving to DLQ.")
            self.redis_client.client.xadd(DEAD_QUEUE, {'original_id': msg_id, 'data': msg_data[b'data']})
            self.redis_client.client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
            self.failed_messages += 1
            return
        
        try:

            with self.graph_lock.gen_wlock():

                for ents in batch_msg.list_ents:

                    entity_data = {
                        "id": ents.id,
                        "name": ents.text,
                        "type": ents.type,
                        "confidence": ents.confidence
                    }

                    self.driver.add_entity(entity_data=entity_data)
                
                for rel in batch_msg.list_relations:

                    relation_data = {
                        "relation": rel.relation,
                        "confidence": rel.confidence
                    }

                    self.driver.add_relationship(rel.source_text, rel.target_text, edge_data=relation_data)

            self.redis_client.client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
            self.processed_messages += 1
            logger.info(f"Successfully processed and acknowledged message {msg_id}.")
        except Exception as e:
            logger.error(f"CRITICAL FAILURE processing valid message {msg_id}: {e}. Moving to DLQ.", exc_info=True)
            self.redis_client.client.xadd(DEAD_QUEUE, {'original_id': msg_id, 'data': msg_data[b'data']})
            self.redis_client.client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
            self.failed_messages += 1

                

def main():
    from timeloop import Timeloop
    from datetime import timedelta

    service = GraphBuilder(max_workers=4)
    service.timer = Timeloop()

 

    try:
        service.start()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
    finally:
        service.stop()


if __name__ == "__main__":
    main()


    
