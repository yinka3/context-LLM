from redisclient import RedisClient

class Parser:

    def __init__(self):
        self.redis = RedisClient()
        self.resolver = EntityResolver()

    def process_msg(self, msg_data):
        
        if not msg_data:
            return
        
        
    
    def recieve_messages(self):
        pubsub = self.redis.client.pubsub()
        pubsub.subscribe("ai_response")

        message = pubsub.listen()
        if message['type'] == 'message': 
            self.process_msg(message['data'])
    
    def send_data(self):
        
        
        
    
    
class EntityResolver:

    def __init__(self, context: str, msg):
        
        pass


if '__main__' == __name__:
    parser = Parser()
    while True:
        parser.recieve_messages()

        




