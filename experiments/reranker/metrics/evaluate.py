import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading SentenceTransformer model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Model loaded successfully.")

def is_similar(chunk, ground_truth_chunks, threshold=0.95):
    """
    Check if a chunk is similar to any of the ground truth chunks using cosine similarity.
    """
    if not ground_truth_chunks:
        return False
    chunk_vec = model.encode([chunk])
    gt_vecs = model.encode(ground_truth_chunks)
    sims = cosine_similarity(chunk_vec, gt_vecs)[0]
    return np.max(sims) >= threshold

def calculate_ndcg(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str]
) -> float:
    # This function remains the same as it already handles duplicates correctly
    if not retrieved_chunks or not ground_truth_chunks:
        return 0.0
    ground_truth_chunks_unique = list(set(ground_truth_chunks))
    relevance_scores = [1 if is_similar(chunk, ground_truth_chunks_unique) else 0 for chunk in retrieved_chunks]
    num_relevant = sum(relevance_scores)
    ideal_scores = [1] * num_relevant + [0] * (len(retrieved_chunks) - num_relevant)
    positions = np.arange(1, len(retrieved_chunks) + 1)
    discount = 1 / np.log2(positions + 1)
    dcg = np.sum(relevance_scores * discount)
    idcg = np.sum(ideal_scores * discount)
    # if not idcg or not dcg:
    #     print(f"dcg: {dcg}: \n idcg: {idcg}\n")
    #     print("-"*30)
    return dcg / idcg if idcg > 0 else 0.0

def compute_rr(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str]
) -> float:
    """
    Compute Reciprocal Rank (RR) for the first relevant chunk.
    """
    ground_truth_chunks_unique = list(set(ground_truth_chunks))
    for i, chunk in enumerate(retrieved_chunks, 1):
        if is_similar(chunk, ground_truth_chunks_unique):
            return 1 / i
    return 0.0

def calculate_precision(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str]
) -> float:
    """
    Compute Precision for the retrieved chunks.
    Precision = (Number of relevant retrieved chunks) / (Total number of retrieved chunks)
    """
    if not retrieved_chunks:
        return 0.0
    
    ground_truth_chunks_unique = list(set(ground_truth_chunks))
    if not ground_truth_chunks_unique:
        return 0.0

    relevant_retrieved_count = sum(1 for chunk in retrieved_chunks if is_similar(chunk, ground_truth_chunks_unique))
    
    return relevant_retrieved_count / len(retrieved_chunks)

def calculate_recall(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str]
) -> float:
    """
    Compute Recall for the retrieved chunks.
    Recall = (Number of unique ground truth chunks found in retrieved) / (Total number of unique ground truth chunks)
    """
    ground_truth_chunks_unique = list(set(ground_truth_chunks))
    if not ground_truth_chunks_unique:
        return 1.0 if not retrieved_chunks else 0.0

    if not retrieved_chunks:
        return 0.0
    
    # Count how many of the unique ground truth chunks are represented in the retrieved set
    found_gt_chunks_count = 0
    for gt_chunk in ground_truth_chunks_unique:
        if is_similar(gt_chunk, retrieved_chunks):
            found_gt_chunks_count += 1
            
    return found_gt_chunks_count / len(ground_truth_chunks_unique)


def load_gt_and_reranked_chunks(gt_file: str, reranked_file_dirs: list):
    """
    Load ground truth and reranked chunks from JSON files.
    Accepts any number of reranked_file_dirs (1 for each batch).
    """
    # Load ground truth chunks
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    df_gt = pd.DataFrame(gt_data)
    df_gt.columns = ['question', 'rewritten_question', 'answer', 'relevant_chunks']
    print(f"Ground truth chunks dataframe loaded, shape: {df_gt.shape}")
    print(df_gt.head())

    # Load reranked chunks
    # Get all JSON files from the provided directories
    json_files = []
    for dir_path in reranked_file_dirs:
        print(f"Searching for JSON files in directory: {dir_path}")
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                print(f"Loading file: {filename}")
                file_path = os.path.join(dir_path, filename)
                json_files.append(file_path)
        print(f"Total of {len(json_files)} json files found in the directories.")

    # Read and process each JSON file
    records = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for q in data.get('questions', []):
                # Extract original question and reranked content
                orig_q = q.get('original_question', '')
                rag_info = q.get('rag_info', [])
                chunk_contents = [item.get('chunk_content', '') for item in rag_info]
                records.append({'original_question': orig_q, 'reranked_content': chunk_contents})

    # Save the processed records to a temporary JSON file
    # This is to ensure that the JSON file is saved with proper UTF-8 encoding
    with open('tmp_utf8.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open('tmp_utf8.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df_reranked = pd.DataFrame(data)

    return df_gt, df_reranked

def normalize_question(q):
    return str(q).strip().lower()

def compute_metrics(df_gt: pd.DataFrame, df_reranked: pd.DataFrame):
    # Helper functions to apply to the DataFrame
    def calc_ndcg(row):
        if isinstance(row['relevant_chunks'], list) and isinstance(row['reranked_content'], list):
            return calculate_ndcg(row['reranked_content'], row['relevant_chunks'])
        return np.nan

    def calc_mrr(row):
        if isinstance(row['relevant_chunks'], list) and isinstance(row['reranked_content'], list):
            return compute_rr(row['reranked_content'], row['relevant_chunks'])
        return np.nan

    def calc_precision(row):
        if isinstance(row['relevant_chunks'], list) and isinstance(row['reranked_content'], list):
            return calculate_precision(row['reranked_content'], row['relevant_chunks'])
        return np.nan

    def calc_recall(row):
        if isinstance(row['relevant_chunks'], list) and isinstance(row['reranked_content'], list):
            return calculate_recall(row['reranked_content'], row['relevant_chunks'])
        return np.nan

    # Normalize question strings for robust merging
    df_gt['question_norm'] = df_gt['question'].apply(normalize_question)
    df_reranked['question_norm'] = df_reranked['original_question'].apply(normalize_question)

    # Merge the ground truth and reranked dataframes
    merged_df = pd.merge(
        df_reranked,
        df_gt[['question_norm', 'relevant_chunks']],
        on='question_norm',
        how='right'
    )

    # Calculate all metrics
    merged_df['ndcg'] = merged_df.apply(calc_ndcg, axis=1)
    merged_df['mrr'] = merged_df.apply(calc_mrr, axis=1)
    merged_df['precision'] = merged_df.apply(calc_precision, axis=1)
    merged_df['recall'] = merged_df.apply(calc_recall, axis=1)

    # Return the dataframe with all metrics
    return merged_df[['original_question', 'reranked_content', 'relevant_chunks', 'ndcg', 'mrr', 'precision', 'recall']]

if __name__ == "__main__":
    #root_dir = '/home/xinyu/mamba-wxy/data'
    root_dir = '/root/autodl-tmp/irelia_pipeline/data'
    

    #gt_file = '/root/autodl-tmp/RAG_Agent_data/data_for_annotation/annotated/all_annotated.json'
    #gt_file = f'{root_dir}/model_answers/zeekr_20250627/question_72_annotated.json'  
    gt_file = f'{root_dir}/model_answers/zeekr_20250627/question_72_annotated.json'  
    #reranked_dirs = ["/root/autodl-tmp/RAG_Agent_thomas/src/test/test_questions/zeekr_questions/question_batch1_bge-reranker-v2-gemma","/root/autodl-tmp/RAG_Agent_thomas/src/test/test_questions/zeekr_questions/question_batch2_bge-reranker-v2-gemma"]
    
    #rerankers = list(range(100, 1701, 100))
    rerankers = list(range(100, 1601, 100)) + [1650]

    rerankers = [f'checkpoint-{r}' for r in rerankers]
    
    #reranker_name = 'checkpoint-100'
    for reranker_name in tqdm(rerankers):
        #reranked_dirs = [f'{root_dir}/qas/zeekr_questions/20250714/question_batch_chi{i + 1}_{reranker_name}_rewritten' for i in range(4)] + [f'{root_dir}/qas/zeekr_questions/20250714/question_batch_eng{i + 1}_{reranker_name}_rewritten' for i in range(4)]
        reranked_dirs = [f'/root/autodl-tmp/RAG_Agent_vllm_hyc2/RAG_Agent/src/test/test_onlyzeekr/question_72_top10_{reranker_name}']
    
        # reranked_dirs = ["/root/autodl-tmp/RAG_Agent_thomas/src/test/test_questions/zeekr_questions/question_batch1_checkpoint-1780","/root/autodl-tmp/RAG_Agent_thomas/src/test/test_questions/zeekr_questions/question_batch2_checkpoint-1780"]

        df_gt, df_reranked = load_gt_and_reranked_chunks(gt_file, reranked_dirs)
        metrics_df = compute_metrics(df_gt, df_reranked)

        #output_file = f'{root_dir}/evaluation_results/{20250714}/both_lang/metrics_output_{reranker_name}.csv'
        output_file = f'{root_dir}/evaluation_results/{20250714}/xbhtest/metrics_output_{reranker_name}.csv'
        metrics_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Metrics computed and saved to {output_file}")

        print("\n--- Metrics Summary ---")
        print(metrics_df[['ndcg', 'mrr', 'precision', 'recall']].mean())
        print("\n--- Full Metrics DataFrame ---")
        print(metrics_df)