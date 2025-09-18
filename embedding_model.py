
import threading
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        if torch.backends.mkl.is_available():
            torch.backends.mkl.enabled = True
        
        self.model = SentenceTransformer('BAAI/bge-m3', device='cpu')
        self.model.eval()
        
        self._initialized = True
        print("BGE-M3 model loaded on CPU with optimizations")
        self._cache = {} 
        self._cache_max = 1000
    
    def encode(self, texts):
        """Use first model instance"""
        with torch.no_grad():
            if isinstance(texts, str):
                texts = [texts]
            return self.model.encode(texts)
    
    def encode_single(self, text):
        """Convenience for single text"""
        return self.encode([text])[0]
    
