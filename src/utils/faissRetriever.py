import logging
logger = logging.getLogger(__name__)
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional

class FaissRetriever:
    """Faiss retriever compatible with LangChain that supports metadata filtering."""
    
    def __init__(self, embeddings, embedding_fn: HuggingFaceEmbeddings):
        super().__init__()
        self.embeddings = embedding_fn
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]

        # res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatIP(dimension)
        # self.index = faiss.index_cpu_to_gpu(res, 0, index)

        x = embeddings.astype('float32')
        faiss.normalize_L2(x)

        self.index.add(x)
        
        logger.info(f"Building FAISS index with {len(embeddings)} vectors of dimension {dimension}")

    def invoke(
            self,
            querys: list[str],
            k: int
        ):
        query_vec_list = [self.embeddings.embed_query(q) for q in querys]
        query_vector = np.array(query_vec_list).astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        return indices, distances

if __name__ == "__main__":
    import os
    import yaml
    config_path = os.getenv('CONFIG_PATH', '../../config/config_vllm.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    persist_directory = os.path.join(config['persist_directory'], "chroma")
    embeddings_model_name = config['embeddings_model_name']
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    from langchain_chroma import Chroma
    chroma = Chroma(
        #collection_name="lotus",
        collection_name="zeekr",
        embedding_function =embeddings,
        persist_directory=persist_directory,
        relevance_score_fn="l2" # l2, ip, cosine
    )
    
    docs = chroma.get(include=["metadatas", "embeddings"])
    retriever = FaissRetriever(docs['embeddings'], embeddings)

    #querys = ["Lotus Technology Company (LTC) was incorporated as an exempted company in accordance with the laws and regulations of the Cayman Islands on August 9, 2021. The mailing address of Lotus Technology's principal executive office is No. 800 Century Avenue, Pudong District, Shanghai, People’s Republic of China, and the phone number is +86 21 5466 - 6258. Lotus Technology's corporate website address is www.group-lotus.com. The information contained in, or accessible through, Lotus Technology's website does not constitute a part of this prospectus."]
    querys = ["Zeekr Intelligent Technology Holding Limited was incorporated as an exempted company under the laws of the Cayman Islands on July 7, 2021. The principal executive office of Zeekr is located at Building 20, No. 3333 Huaning Road, Minhang District, Shanghai 201108, People’s Republic of China, and the contact phone number is +86 21 6065 0666. Zeekr's official corporate website is www.zeekrlife.com. The information contained in or accessible through Zeekr’s website does not constitute part of this prospectus or any other regulatory filing."]

    indices, distances = retriever.invoke(querys, 1000)

    # save indices and distances into a log file
    with open("faiss_retriever2.log", "w") as f:
        f.write(f"indices: {indices}\n")
        f.write(f"distances: {distances}\n")
