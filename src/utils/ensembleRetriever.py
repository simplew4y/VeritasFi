import logging
import torch
logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Set, Union, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .profiler import profiler
import time

from .bm25Retriever import BM25Retriever
from .faissRetriever import FaissRetriever

class EnsembleRetriever:
    """Base class for retriever wrappers that handle document content retrieval"""
    
    def __init__(self, bm25_dir: str,
                 chroma: Chroma,
                 ts_chroma: Chroma,
                 k: int,
                 embeddings: HuggingFaceEmbeddings,
                 faiss_k: int = None,
                 bm25_k: int = None,
                 faiss_ts_k: int = None,
                 enable_expand: bool = False,
                 ):
        super().__init__()
        self.embeddings = embeddings
        self.faiss_k = faiss_k if faiss_k is not None else k
        self.bm25_k = bm25_k if bm25_k is not None else k
        self.faiss_ts_k = faiss_ts_k if faiss_ts_k is not None else k
        self.enable_expand = enable_expand
        self.chroma = chroma
        #print(bm25_dir)
        self.bm25_retriever = BM25Retriever(bm25_dir)
        
        docs = chroma.get(include=["metadatas", "embeddings"])
        self.faiss_retriever = FaissRetriever(docs['embeddings'], embeddings)

        ts_docs = ts_chroma.get(include=["documents", "embeddings"])
        self.title_summary_faiss_retriever = FaissRetriever(ts_docs['embeddings'], embeddings)
        
        self.chunk_metadata = docs['metadatas']
        self.docid2idx = {doc['doc_id']: idx for idx, doc in enumerate(self.chunk_metadata)}
        self.num_chunk = len(docs['metadatas'])
        self.title_summaries = ts_docs['documents']
        
    @profiler.profile_function(name="retrieve")
    def invoke(
        self,
        input: str,
        hyde_chunks: list[str],
    ) -> List[Dict]:
        """Get documents with their content"""

        seen_ids = set()
        chunk_list = []
        bundle_cnt = 0

        if self.faiss_k > 0:
            profiler.start("retrieve_faiss")
            inputs = [input] + hyde_chunks
            # inputs = hyde_chunks
            faiss_ids_list, faiss_scores_list = self.faiss_retriever.invoke(inputs, 2048)
            for inp, faiss_ids, faiss_scores in zip(inputs, faiss_ids_list, faiss_scores_list):
                effective_ids = {idx: score for idx, score in zip(faiss_ids, faiss_scores)}
                # augment retrieved content with precious and next chunk
                top_k_ids, top_k_scores = faiss_ids[:self.faiss_k], faiss_scores[:self.faiss_k]
                for idx, score in zip(top_k_ids, top_k_scores):
                    if idx in seen_ids:
                        continue
                    seen_ids.add(idx)
                    ids = [idx]
                    doc_metadata = self.chunk_metadata[idx]
                    # gather bundle if bundle_id is not null
                    if doc_metadata.get('bundle_id', None) != None:
                        bundle_id = doc_metadata['bundle_id']
                        # find corresponding bundle_id from self.chunk_metadata
                        bundle_ids = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('bundle_id', None) == bundle_id]
                        ids = bundle_ids
                        seen_ids.update(bundle_ids)

                    # expand chunk if score is high and expandation is enabled
                    if (score > 0.72) and self.enable_expand:
                        prev_doc_id = doc_metadata['prev_chunk_id']
                        next_doc_id = doc_metadata['next_chunk_id']
                        while len(ids) < 4:
                            flag = False
                            if prev_doc_id != "" and self.docid2idx.get(prev_doc_id, -1) != -1:
                                prev_id = self.docid2idx[prev_doc_id]
                                if effective_ids.get(prev_id, 0) > 0.66 and prev_id not in seen_ids:
                                    flag = True
                                    # doc_metadata['chunk_num'] += 1
                                    seen_ids.add(prev_id)
                                    ids.insert(0, prev_id)
                                    prev_doc_id = self.chunk_metadata[prev_id]['prev_chunk_id']

                            if next_doc_id != "" and self.docid2idx.get(next_doc_id, -1) != -1:
                                next_id = self.docid2idx[next_doc_id]
                                if effective_ids.get(next_id, 0) > 0.66 and next_id not in seen_ids:
                                    flag = True
                                    # doc_metadata['chunk_num'] += 1
                                    seen_ids.add(next_id)
                                    ids.append(next_id)
                                    next_doc_id = self.chunk_metadata[next_id]['next_chunk_id']
                            if not flag:
                                break

                    doc_ids = [self.chunk_metadata[idx]['doc_id'] for idx in ids]
                        
                    docs_dict = self.chroma.get(ids=doc_ids, include=['documents', 'metadatas'])

                    # candidate chunks bring the whole bundle
                    for idx in range(len(docs_dict['documents'])):
                        # logger.info(f"{len(chunk_list)} chunk score: {effective_ids.get(self.docid2idx[docs_dict['metadatas'][idx]['doc_id']], 0)}")
                        # logger.info(f"{len(chunk_list)} chunk doc_id: {docs_dict['metadatas'][idx].get('doc_id', '')}")
                        # logger.info(f"{len(chunk_list)} chunk content: {docs_dict['documents'][idx]}")
                        
                        chunk_list.append(
                            {
                                "retriever": "FAISS",
                                # Convert score to Python's native float type to ensure JSON serialization
                                # (NumPy numeric types like float32 aren't directly JSON-serializable)
                                "score": float(score), 
                                "page_content": docs_dict['documents'][idx],
                                "metadata": docs_dict['metadatas'][idx],
                                "bundle_id": bundle_cnt
                            }
                        )
                        
                    bundle_cnt += 1

            profiler.end("retrieve_faiss")

        if self.faiss_ts_k > 0:
            profiler.start("retrieve_faiss_ts")
            title_summary_ids, title_summary_scores = self.title_summary_faiss_retriever.invoke([input], self.faiss_ts_k)
            title_summary_ids, title_summary_scores = title_summary_ids[0], title_summary_scores[0]
            # logger.info(f"Top {self.faiss_ts_k} Title Summary FAISS results:")
            for title_idx, score in zip(title_summary_ids, title_summary_scores):
                title_summary = self.title_summaries[title_idx]
                # find corresponding chunk idx from self.chunk_metadata
                chunk_idxs = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('title_summary', '') == title_summary]
                # logger.info("score: {score} title_summary: {title_summary}".format(score=score, title_summary=title_summary.replace('\n', ' ')))
                for idx in chunk_idxs:
                    if idx in seen_ids:
                        continue
                    seen_ids.add(idx)
                    ids = [idx]
                    doc_metadata = self.chunk_metadata[idx]
                    # gather bundle if bundle_id is not null
                    if doc_metadata.get('bundle_id', None) != None:
                        bundle_id = doc_metadata['bundle_id']
                        # find corresponding bundle_id from self.chunk_metadata
                        bundle_ids = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('bundle_id', None) == bundle_id]
                        ids = bundle_ids
                        seen_ids.update(bundle_ids)

                    # get content of ids
                    doc_ids = [self.chunk_metadata[idx]['doc_id'] for idx in ids]
                    docs_dict = self.chroma.get(ids=doc_ids, include=['documents', 'metadatas'])
                    doc_content = "\n".join(docs_dict['documents'])
                    title_summary = doc_metadata.get('title_summary', '').replace('\n', ' ')

                    # candidate chunks bring the whole bundle
                    # logger.info(f"Bundle {bundle_cnt}")
                    for idx in range(len(docs_dict['documents'])):
                        # logger.info(f"{len(chunk_list)} chunk doc_id: {docs_dict['metadatas'][idx].get('doc_id', '')}")
                        # logger.info(f"{len(chunk_list)} chunk content: {docs_dict['documents'][idx]}")
                    
                        chunk_list.append(
                            {
                                "retriever": "Title Summary",
                                "score": float(score),
                                "page_content": docs_dict['documents'][idx],
                                "metadata": docs_dict['metadatas'][idx],
                                "bundle_id": bundle_cnt
                            }
                        )

                    bundle_cnt += 1
            
            profiler.end("retrieve_faiss_ts")

        if self.bm25_k > 0:
            profiler.start("retrieve_bm25")
            bm25_ids, bm25_scores = self.bm25_retriever.invoke(input, self.num_chunk)
            top_k_ids, top_k_scores = bm25_ids[:self.bm25_k], bm25_scores[:self.bm25_k]
            
            # logger.info(f"Top {self.bm25_k} BM25 results:")
            for idx, score in zip(top_k_ids, top_k_scores):
                if idx in seen_ids:
                    continue
                seen_ids.add(idx)
                ids = [idx]
                doc_metadata = self.chunk_metadata[idx]
                # gather bundle if bundle_id is not null
                if doc_metadata.get('bundle_id', None) != None:
                    bundle_id = doc_metadata['bundle_id']
                    # find corresponding bundle_id from self.chunk_metadata
                    bundle_ids = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('bundle_id', None) == bundle_id]
                    ids = bundle_ids
                    seen_ids.update(bundle_ids)

                # get content of ids
                doc_ids = [self.chunk_metadata[idx]['doc_id'] for idx in ids]
                docs_dict = self.chroma.get(ids=doc_ids, include=['documents', 'metadatas'])

                # candidate chunks bring the whole bundle
                # logger.info(f"Bundle {bundle_cnt} score: {score}")
                for idx in range(len(docs_dict['documents'])):
                    # logger.info(f"{len(chunk_list)} chunk doc_id: {docs_dict['metadatas'][idx].get('doc_id', '')}")
                    # logger.info(f"{len(chunk_list)} chunk content: {docs_dict['documents'][idx]}")

                    chunk_list.append(
                        {
                            "retriever": "BM25",
                            "score": float(score),
                            "page_content": docs_dict['documents'][idx],
                            "metadata": docs_dict['metadatas'][idx],
                            "bundle_id": bundle_cnt
                        }
                    )

                bundle_cnt += 1
            
            profiler.end("retrieve_bm25")

        profiler.add_metric("retrieved_chunks", len(chunk_list))
            
        return chunk_list

    def compute_similarity(self, chunks: List[str], selected_indices: List[int], candidate_index: int) -> List[float]:
        """
        计算 candidate_index 对应 chunk 和 selected_indices 对应 chunks 的相似度（GPU 加速）。
        
        参数:
            chunks (List[str]): 文档块的字符串列表。
            selected_indices (List[int]): 选定的索引列表。
            candidate_index (int): 候选索引。
            
        返回:
            List[float]: candidate_index 对应 chunk 和 selected_indices 对应 chunks 的相似度列表。
        """
        # 将字符串转化为嵌入向量
        embeddings = torch.stack([torch.tensor(self.embeddings.embed_query(chunk), device='cuda') for chunk in chunks])
        
        # 提取 candidate_index 对应的嵌入向量
        candidate_embedding = embeddings[candidate_index].unsqueeze(0)  # 添加 batch 维度
        
        # 提取 selected_indices 对应的嵌入向量
        selected_embeddings = embeddings[selected_indices]

        # 归一化嵌入向量
        candidate_embedding = torch.nn.functional.normalize(candidate_embedding, dim=-1)
        selected_embeddings = torch.nn.functional.normalize(selected_embeddings, dim=-1)
        
        # 计算余弦相似度 (使用点积)
        similarity = torch.matmul(selected_embeddings, candidate_embedding.T).squeeze(-1)
        
        return similarity
    
    def compute_similarity_mtx(self, chunks: List[str]) -> torch.Tensor:
        """
        计算 chunks 两两之间的相似度矩阵（GPU 加速）。
        
        参数:
            chunks (List[str]): 文档块的字符串列表。
            
        返回:
            torch.Tensor: chunks 两两之间的相似度矩阵。
        """
        embeddings = torch.stack([torch.tensor(self.embeddings.embed_query(chunk), device='cuda') for chunk in chunks])
        
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        similarity_mtx = torch.matmul(embeddings, embeddings.T)
        
        return similarity_mtx

if __name__ == "__main__":
    import os
    import yaml
    config_path = os.getenv('CONFIG_PATH', '../config/production.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    #collection_name = "lotus"
    collection_name = 'zeekr'
    embeddings_model_name = config['embeddings_model_name']
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    from langchain_chroma import Chroma
    chroma = Chroma(
        collection_name=collection_name,
        embedding_function =embeddings,
        persist_directory=os.path.join(config['persist_directory'], "chroma"),
        relevance_score_fn="l2" # l2, ip, cosine
    )
    ts_chroma = Chroma(
        collection_name=collection_name,
        embedding_function =embeddings,
        persist_directory=os.path.join(config['persist_directory'], "ts_chroma"),
        relevance_score_fn="l2" # l2, ip, cosine
    )
    
    bm25_dir = os.path.join(config['persist_directory'], "bm25_index", collection_name)
    retriever = EnsembleRetriever(bm25_dir, chroma, ts_chroma, 10, embeddings,
                                  faiss_k=0, bm25_k=0, faiss_ts_k=10)

    rewritten = "title: Recommendation to LCAA Shareholders summary: The LCAA Board expresses strong confidence in the fairness and advantages of all proposals set for discussion at the Extraordinary General Meeting. They unanimously urge shareholders to support the NTA Proposal, the Business Combination Proposal, the Merger Proposal, and the Adjournment Proposal, should it be introduced."

    hyde_chunks = []

    recall_chunks = retriever.invoke(rewritten, hyde_chunks)

    # save indices and distances into a log file
    with open("ensemble_retriever.log", "w") as f:
        f.write(str(recall_chunks))
