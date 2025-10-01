import json
from redisclient import RedisClient
from schema.common_pb2 import Message
class Parser:

    def __init__(self):
        self.redis = RedisClient()
        self.pubsub = self.redis.client.pubsub()
        self.resolver = EntityResolver(redis=self.redis)

        self.handlers = {
            "parser:ai_response": self.process_msg
        }

    def process_msg(self, data: bytes):
        
        msg_data = Message()
        msg_data.ParseFromString(data)
        self.resolver.send_to_ER(msg_data)



class EntityResolver:

    def __init__(self, redis):
        self.user_name = "Yinka"
        self.redis = redis
        def get_redis_context(self):
            sorted_set_key = f"recent_messages:{self.user_name}"
            recent_msg_ids = self.redis_client.client.zrevrange(sorted_set_key, 0, 49)
            
            context_text = []
            for msg_id in recent_msg_ids:
                msg_data = self.redis_client.client.hget(f"message_content:{self.user_name}", msg_id)
                if msg_data:
                    context_text.append(json.loads(msg_data)['message'])
            
            return " ".join(context_text)
        
        self.context = get_redis_context()
    

    def send_to_ER(self):
        pass
    





if '__main__' == __name__:
    parser = Parser()
    while True:
        parser.recieve_messages()

        




