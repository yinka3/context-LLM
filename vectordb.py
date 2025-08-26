from typing import List, Union
import chromadb
from chromadb.api.models.Collection import Collection
import logging
import json

from dtypes import EntityData, MessageData

class ChromaClient:

    def __init__(self, file_path: str = "./chroma_serv"):

        self.client = chromadb.PersistentClient(path=file_path)
        self.collection: Collection = self.client.get_or_create_collection(name="llm-memory", 
                                                                           metadata={"hnsw:space": "cosine"})

        logging.info("Chroma client initialized")
    
    def get_item(self, item: Union[int, List], include: List[str] = ["embeddings", "metadatas", "documents"]):
        
        if isinstance(item, int):
            results = self.collection.get(ids=[str(item)], include=include)
        else:
            str_ids = [str(id_) for id_ in item]
            results = self.collection.get(ids=str_ids, include=include)
        
        return results

    def add_item(self, item: Union[MessageData, EntityData]):
        
        if isinstance(item, MessageData):
            id_ = f"msg_{item.id}"
            self.collection.add(ids=[id_],
                                documents=[item.message],
                                metadatas=[{"node_type": "message", 
                                            "sentiment": item.sentiment,
                                            "timestamp": item.timestamp}]
            )
        else:
            id_ = f"ent_{item.id}"
            self.collection.add(ids=[id_],
                                documents=[item.name],
                                metadatas=[{
                                "node_type": "entity",
                                "entity_type": item.type,
                                "aliases": json.dumps(item.aliases),
                                "confidence": item.confidence
                            }]
            )
    
    def query(self, text: str, n_results: int = 15, node_type: str = "message", 
              include: List[str] = ["embeddings", "metadatas", "documents", "distances"]):
        
        results = self.collection.query(
                query_texts=[text],
                where={"node_type": node_type},
                n_results=n_results,
                include=include)
        
        return results
