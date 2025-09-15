from typing import Dict, List
from redis import Redis
from networkx import DiGraph
from utils import hash_password
import os
from dotenv import load_dotenv
from rq import Worker
load_dotenv()


class GraphWriter:

    def __init__(self, queue_list: List[str], graph: DiGraph):
        self.client = Redis(password=os.getenv('REDIS_PASSWORD'))
        self.worker = Worker(queues=queue_list,connection=self.client)
        self.graph = graph
        self.status = {}

    def _status(self):
        workers = self.worker.all(connection=self.client)
        
        for i, worker in enumerate(workers):

            data = {
                "State": worker.state,
                "Current_job_id": worker.get_current_job_id(),
                "Successful_jobs": worker.successful_job_count,
                "Failed_jobs": worker.failed_job_count,
                "Total working time": worker.total_working_time
            }
        
            self.status[i] = data
    
    def apply_graph(self, task_type: str, transaction: Dict):

        #load graph here, will use the other method from the first version
        
        if task_type == "ADD":
            self.add_graph()
        elif task_type == "UPDATE":
            self.update_graph()
        pass
    
    def update_graph(self):
        pass

    def add_graph(self):
        pass
        