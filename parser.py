import logging
import signal
import os
import socket
import sys
import time
from redis import exceptions
from redisclient import RedisClient
from schema.common_pb2 import BatchMessage, Entity, Relationship

logger = logging.getLogger(__name__)

STREAM_KEY = "stream:ai_response"
CONSUMER_GROUP = "group:parsers"
CONSUMER_NAME = CONSUMER_NAME = f"parser-{socket.gethostname()}-{os.getpid()}"
DEAD_QUEUE = 'stream:parser_dead_letters'

class Parser:

    def __init__(self):
        self.redis = RedisClient()
        self.is_running = True

        try:
            logger.info(f"Consumer Group: {CONSUMER_GROUP} exisit for stream {STREAM_KEY}")
            self.redis.client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id='$', mkstream=True)
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.warning("Consumer group already in use")
            else:
                logger.error(f"An unexpected error occurred setting up consumer group: {e}")
                sys.exit(1)
    

    def _signal_handler(self, signum, frame):
        # if this shuts down, there needs to be a cascade of shut down
        logger.info(f"Recieved signal {signum}, initiating graceful shutdown")
        self.is_running = False
    
    def run(self):
        
        logger.info(f"From Parser, listening to stream {STREAM_KEY}")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        while self.is_running:
            try:
                response = self.redis.client.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {STREAM_KEY: '>'},
                    block=3000
                )
            
                if not response:
                    continue

                for _, messages in response:
                    for msg_id, msg_data in messages:
                        self.process_msg(msg_id, msg_data[b'data'])
            except exceptions.ConnectionError as e:
                logger.error(f"Redis connection lost: {e}. Retry in 3 seconds")
                time.sleep(3)
            except Exception as e:
                logger.error(f"Error in parser loop: Retry in 3 seconds")
                time.sleep(3)
        
        logger.info("Parser has shut down")

        
    def process_msg(self, msg_id: str, data: bytes):
        
        try:
            batch_data = BatchMessage()
            batch_data.ParseFromString(data)

            if not self.is_batch_valid(batch_data):
                logger.warning(f"Message {msg_id} is invalid. Moving to Dead Letter Queue.")
                self.redis.client.xadd(DEAD_QUEUE, {'original_id': msg_id, 'data': data})
                self.redis.client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                return
            
            self.redis.client.xadd('stream:graph_tasks', {'data': data})
            logger.info(f"Successfully routed valid BatchMessage {msg_id} to graph_tasks stream.")
            self.redis.client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
            logger.debug(f"Acknowledged message {msg_id}.")
        except Exception as e:
            logger.error(f"CRITICAL FAILURE processing message {msg_id}: {e}. Moving to Dead Letter Queue.")
            self.redis.client.xadd(DEAD_QUEUE, {'original_id': msg_id, 'data': data})
            self.redis.client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
    
    
            
        

if '__main__' == __name__:
    import logging_setup
    logging_setup.setup_logging(log_file="parser_service.log")
    parser = Parser()
    parser.run()

        




