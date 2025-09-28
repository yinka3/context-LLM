from graph_driver import KnowGraph
from redisclient import RedisClient
import threading


def build():

    driver: KnowGraph = KnowGraph()
    redis = RedisClient()

    pubsub = redis.client.pubsub()
    pubsub.subscribe('graph:*')

    while True:
        mess = pubsub.get_message(timeout=1.0)
        if mess['type'] == 'message':

            channel = mess['channel']
            if channel == 'graph:add-entity':
                data = mess['data']
                driver.add_entity(data)
            elif channel == 'graph:add-relation':
                data = mess['data']
                #NOTE get ent one and two and edge data
                driver.add_relationship()

threading.Thread(target=build, daemon=True).start()

    
