import os
import yaml
import logging
logger = logging.getLogger(__name__)

from datetime import datetime
from typing import Dict, List, Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .ensembleRetriever import EnsembleRetriever
import GPUtil

class RAGManager:
    """Singleton class for managing RAG collections"""
    _collections: Dict[str, Tuple[Chroma, Chroma]] = {}
    _retrievers: List[EnsembleRetriever] = []

    _instance = None
    _config = None

    def __new__(cls, config: Dict = None, collections: Dict[str, int] = None):
        if cls._instance is None:
            if config is None:
                logger.error("No config provided")
                raise ValueError("No config provided for RAGManager")
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance._initialize(config, collections)
        return cls._instance

    def __init__(self, config: Dict = None, collections: Dict[str, int] = None):
        pass

    def _initialize(self, config: Dict, collections: Dict[str, int]):
        self._config = config
        self.embeddings_model_name = config['embeddings_model_name']
        self.batch_size = 5

        # Suppress warnings from GemmaTokenizerFast regarding __call__ method and logits type as below:
        # [You're using a GemmaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method 
        # is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding. 
        # Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it
        #  will always be FP32)]. -- from terminal
        import transformers
        transformers.logging.set_verbosity_error()
        #print('hihihi')
        try:
            logger.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
            logger.info("Embedding model loaded successfully.")
            import torch
            logger.warning("Load Embedding model: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

        if collections is not None:
            for collection, top_k in collections.items():
                if top_k <= 0:
                    continue
                self.create_collection(collection)
                self._retrievers.append(self.create_retriever(top_k, collection, retriever_type="ensemble"))

        
    def create_collection(self, collection_name: str):
        """Create a new collection with all supported retrievers"""
        if collection_name not in self._collections:
            # Initialize Chroma
            chroma = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=os.path.join(self._config['persist_directory'], "chroma"),
                relevance_score_fn="cosine" # l2, ip, cosine
            )
            
            ts_chroma = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=os.path.join(self._config['persist_directory'], "ts_chroma"),
                relevance_score_fn="cosine" # l2, ip, cosine
            )
            self._collections[collection_name] = (chroma, ts_chroma)
            import torch
            logger.warning("Load Chroma: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    def get_collection_documents(self, collection_name: str, doc_ids: Optional[List[str]] = None) -> List[Document]:
        """Get documents from a collection by document IDs. User should not assume that the order of the returned documents matches the order of the input IDs."""
        chroma, _ = self._collections[collection_name]
        if doc_ids is None:
            chroma_docs = chroma.get()
        else:
            chroma_docs = chroma.get(ids=doc_ids)

        documents = [
            Document(
                page_content=page_content,
                metadata=metadata
            )
            for page_content, metadata in zip(chroma_docs['documents'], chroma_docs['metadatas'])
        ]
        return documents
    
    def create_retriever(self, k: int, collection_name: str, retriever_type: str = "chroma"):
        """Create a specific retriever for a collection"""
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        bm25_dir = os.path.join(self._config['persist_directory'], "bm25_index", collection_name)

        chroma, ts_chroma = self._collections[collection_name]
        retriver = EnsembleRetriever(bm25_dir, chroma, ts_chroma, k, self.embeddings, enable_expand = True)
            
        return retriver


# Usage example
def main():
    config_path = "../../config/config_test.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    questions = [
        "Are there any new releases in 2023?",
        "Can you tell me how Lotus's approach to vehicle design evolved between 2000 and 2020?",
        "What are the unique technical features that make Lotus stand out in racing?" ,
        "Can you explain the lightweight design philosophy of Lotus?" ,
        "Which Lotus models are best known for their driving performance on the track?" ,
    ]
    
    rag = RAGManager(config)
    log_gpu_usage('RAGManager init')
    #rag.create_collection("lotus")
    rag.create_collection("zeekr")
    log_gpu_usage('RAGManager create collection')
    #retriever = rag.create_retriever(5, "lotus", "ensemble")
    retriever = rag.create_retriever(5, "zeekr", "ensemble")
    log_gpu_usage('RAGManager get retriever')

    for q in questions:
        documents = retriever.invoke(q)
        log_gpu_usage('RAGManager invoke retriever')
        print(f"Question: {q}")
        for i, doc in enumerate(documents):
            print(f"{i}: {doc}")
        print("")
        

def log_gpu_usage(event_name):
    gpus = GPUtil.getGPUs()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gpu_log_file = "gpu_usage.log"
    for gpu in gpus:
        gpu_info = (
            f"Timestamp: {timestamp}, Event: {event_name}, "
            f"GPU ID: {gpu.id}, GPU Name: {gpu.name}, "
            f"Memory Used: {gpu.memoryUsed} MB, Memory Total: {gpu.memoryTotal} MB"
        )
        # 将信息追加到日志文件
        with open(gpu_log_file, 'a') as f:
            f.write(gpu_info + '\n')

if __name__ == "__main__":
    main()
