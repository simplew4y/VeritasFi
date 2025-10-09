'''
1. Retrieve top-k chunks for each evidence
2. Compare each chunk with evidence using OpenAI API
3. Calculate hit rate for each evidence / overall hit rate
'''

import json
import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import time
import random
from math import ceil

# --- Global variables for worker processes ---
# These will be initialized in each worker process by worker_init
worker_qwen_model = None
worker_qwen_tokenizer = None
worker_openai_client = None
worker_device = None
worker_openai_model_name = None
worker_top_k = None
worker_compare_method = None
worker_threshold = None

# Global GPU assignments list (set before creating pool)
gpu_assignments = []

def worker_init(args, gpu_assignments_list):
    """
    Initializes models and clients for each worker process.
    This function is called once per worker when the Pool is created.
    Args:
        args: Command line arguments
        gpu_assignments_list: List of GPU assignments for each worker
    """
    global worker_qwen_model, worker_qwen_tokenizer, worker_openai_client, worker_device, worker_openai_model_name, worker_top_k, worker_compare_method, worker_threshold

    # Get GPU assignment for this worker
    worker_id = mp.current_process()._identity[0] - 1
    gpu_id = gpu_assignments_list[worker_id] if worker_id < len(gpu_assignments_list) else None

    if gpu_id is not None:
        worker_device = f"cuda:{gpu_id}"
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
    else:
        worker_device = "cpu"
    
    print(f"Worker {worker_id} initialized on device: {worker_device}")

    # Load Qwen3 model and tokenizer
    worker_qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True, padding_side='left', cache_dir="")
    worker_qwen_model = AutoModel.from_pretrained(args.qwen_model, trust_remote_code=True, cache_dir="", device_map=worker_device)
    worker_qwen_model.eval()

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    worker_openai_client = OpenAI(
        base_url="https://api.openai.com/v1",
        api_key=api_key
    )
    worker_openai_model_name = args.openai_model
    worker_top_k = args.top_k
    worker_compare_method = args.compare_method
    worker_threshold = args.threshold

def process_sample(example):
    """
    Processes a single sample. This function is executed by worker processes.
    It accesses the models and clients initialized in the global worker scope.
    """
    global worker_qwen_model, worker_qwen_tokenizer, worker_openai_client, worker_device, worker_openai_model_name, worker_top_k, worker_compare_method, worker_threshold

    try:
        # Initialize new fields
        example['top_chunks'] = []
        example['overall_hit_rate'] = 0.0
        example['unhit_evidence'] = []
        example['hit_chunk_retrievers'] = []
        
        question = example.get('question')
        answer = example.get('answer')
        evidence_list = example.get('evidence', [])
        query_chunks = example.get('query_chunks', [])
        retriever_chunks = example.get('retriever_chunks', [])
        
        if not evidence_list or not query_chunks:
            return example
        
        # Batch select top-k chunks for all evidences at once
        batch_results = select_top_chunks_batch(evidence_list, query_chunks, worker_qwen_model, 
                                                worker_qwen_tokenizer, worker_device, 
                                                top_k=worker_top_k, batch_size=16)
        
        top_chunks_per_evidence = []
        evidences_with_hits = 0
        unhit_evidence_list = []
        hit_chunk_retrievers = []
        scores = []
        
        for evidence, (top_chunks, top_similarities) in zip(evidence_list, batch_results):
            top_chunks_per_evidence.append(top_chunks)
            scores.append(top_similarities)
            
            # Compare each top chunk with evidence
            is_hit = False
            hit_chunk_idx = -1
            if worker_compare_method == 'similarity':
                is_hit = top_similarities[0] >= worker_threshold if top_similarities else False
                if is_hit:
                    hit_chunk_idx = 0
            else:
                for idx, chunk in enumerate(top_chunks):
                    if worker_compare_method == 'llm':
                        is_hit = compare_with_openai(question, answer, evidence, chunk, worker_openai_client, worker_openai_model_name)
                    else:
                        is_hit = (chunk.strip() == evidence.strip())

                    if is_hit:
                        hit_chunk_idx = idx
                        break
            
            # Track if this evidence has at least one hit
            if is_hit:
                evidences_with_hits += 1
                # Find the retriever for the hit chunk
                # top_chunks are selected from query_chunks, need to find original index
                if hit_chunk_idx >= 0 and hit_chunk_idx < len(top_chunks):
                    hit_chunk = top_chunks[hit_chunk_idx]
                    # Find this chunk in the original query_chunks
                    try:
                        original_idx = query_chunks.index(hit_chunk)
                        if retriever_chunks and original_idx < len(retriever_chunks):
                            hit_chunk_retrievers.append(retriever_chunks[original_idx])
                        else:
                            hit_chunk_retrievers.append("unknown")
                    except ValueError:
                        hit_chunk_retrievers.append("unknown")
                else:
                    hit_chunk_retrievers.append("unknown")
            else:
                # Track unhit evidence
                unhit_evidence_list.append(evidence)
            
        example['top_chunks'] = [chunk for chunks in top_chunks_per_evidence for chunk in chunks]
        example['num_hits'] = evidences_with_hits
        # Calculate overall hit rate: proportion of evidences with at least one hit
        example['hit_rate'] = evidences_with_hits / len(evidence_list) if evidence_list else 0.0
        example['num_chunks'] = len(query_chunks)
        example['num_evidences'] = len(evidence_list)
        example['unhit_evidence'] = unhit_evidence_list
        example['hit_chunk_retrievers'] = hit_chunk_retrievers
        example['final_score'] = scores
        
        if 'query_chunks' in example:
            del example['query_chunks']
        if 'retriever_chunks' in example:
            del example['retriever_chunks']
            
        return example
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None # Return None on error

def load_json_dataset(file_path):
    """Load dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    if not texts:
        return np.array([])
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          return_tensors="pt", max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use last token pooling instead of mean pooling
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


# def get_embeddings(texts, model, tokenizer, device, batch_size=32):
#     """Get embeddings using Qwen3 model with batching to prevent OOM"""
#     if not texts:
#         return np.array([])
    
#     all_embeddings = []
    
#     # Process in batches
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]
#         inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=4096)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embeddings = outputs.last_hidden_state.mean(dim=1)
#             all_embeddings.append(embeddings.cpu().numpy())
    
#     # Concatenate all batch embeddings
#     return np.vstack(all_embeddings)

def select_top_chunks(evidence, query_chunks, model, tokenizer, device, top_k=3, batch_size=32):
    """Select top-k query chunks with highest cosine similarity to evidence.
    If top_k=-1, return all chunks sorted by similarity.
    Returns: (top_chunks, similarities_of_top_chunks)"""
    if not query_chunks:
        return [], []
    
    evidence_emb = get_embeddings([evidence], model, tokenizer, device, batch_size=1)
    chunks_emb = get_embeddings(query_chunks, model, tokenizer, device, batch_size=batch_size)
    
    similarities = cosine_similarity(evidence_emb, chunks_emb)[0]
    
    if top_k == -1:
        # Return all chunks sorted by similarity (highest first)
        top_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_chunks = [query_chunks[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    
    return top_chunks, top_similarities

def select_top_chunks_batch(evidence_list, query_chunks, model, tokenizer, device, top_k=3, batch_size=32):
    """Batch version: Select top-k query chunks for multiple evidences at once.
    Args:
        evidence_list: List of evidence strings
        query_chunks: List of chunk strings to search from
        model, tokenizer, device: Model components
        top_k: Number of top chunks to select per evidence (-1 for all)
        batch_size: Batch size for embedding generation
    Returns:
        List of tuples: [(top_chunks, top_similarities), ...] for each evidence
    """
    if not evidence_list or not query_chunks:
        return [([], [])] * len(evidence_list)
    
    # Get embeddings for all evidences in batch
    evidence_embs = get_embeddings(evidence_list, model, tokenizer, device, batch_size=batch_size)
    # Get embeddings for all chunks once
    chunks_emb = get_embeddings(query_chunks, model, tokenizer, device, batch_size=batch_size)
    
    # Compute similarities for all evidences at once: shape (num_evidences, num_chunks)
    similarities_matrix = cosine_similarity(evidence_embs, chunks_emb)
    
    results = []
    for similarities in similarities_matrix:
        if top_k == -1:
            # Return all chunks sorted by similarity (highest first)
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        top_chunks = [query_chunks[i] for i in top_indices]
        top_similarities = [similarities[i] for i in top_indices]
        results.append((top_chunks, top_similarities))
    
    return results

def compare_with_openai(question, answer, evidence, chunk, client, model, evaluation_round=1):
    """Compare evidence and chunk using OpenAI API with retry logic.
    Evaluates the chunk 'evaluation_round' times and only considers it effective if LLM responds YES more than ceil(round/2) times."""
    prompt = f"""Your task is to verify if the 'Chunk' provides factual or numerical information that is relevant to answering the 'Question' and is correctly sourced from the 'Evidence'.

The core rule is that the 'Chunk' must contain at least one factual or numerical statement that is used in the reasoning process to get the 'Answer'. This statement must be identical to (or included in) the 'Evidence'.

Crucially, the 'Chunk' does not need to contain enough information to derive the final 'Answer' on its own. It only needs to provide a correct and relevant piece of the puzzle.

When the 'Evidence' is a table or structured list and the 'Answer' is numerical, the 'Chunk' must contain the specific numbers and context needed to either find or calculate the 'Answer'.

Question: {question}
Answer: {answer}
Evidence: {evidence}
Chunk: {chunk}

Does the Chunk provide a correct and relevant fact from the Evidence for answering the Question? Respond with only "YES" or "NO"."""

    # Evaluate 'evaluation_round' times and count YES responses
    yes_count = 0
    
    for round_idx in range(evaluation_round):
        toomany_cnt = 0
        cnt = 0
        
        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                result = response.choices[0].message.content.strip().upper()
                print(f"Round {round_idx + 1}: {result}")
                
                if result == "YES":
                    yes_count += 1
                break  # Successfully got a response, move to next round
                
            except Exception as e:
                if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429:
                    toomany_cnt += 1
                    sleep_time = random.randint(2 * toomany_cnt, 5 * toomany_cnt)
                    print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                    time.sleep(sleep_time)
                    continue
                print(f"Error calling OpenAI API: {e}")
                cnt += 1
                if cnt < 5:
                    print(f"Retrying... (attempt {cnt})")
                    time.sleep(5)
                    continue
                # If all retries failed for this round, break and continue to next round
                break
    
    # Consider chunk effective only if YES count > evaluation_round/2
    threshold = evaluation_round / 2
    is_effective = yes_count > threshold
    print(f"Final decision: {yes_count}/{evaluation_round} YES responses (threshold: >{threshold}) -> {'EFFECTIVE' if is_effective else 'NOT EFFECTIVE'}")
    return is_effective

def save_results(dataset, output_dir):
    """Save results to both JSONL and Hugging Face dataset format"""
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "result.jsonl")
    dataset.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
    
    print(f"Results saved to:")
    print(f"  JSONL: {jsonl_path}")

    all_hits = dataset['num_hits']
    all_evidence = dataset['num_evidences']
    all_num_chunk = dataset['num_chunks']
    all_hit_rate = dataset['hit_rate']
    overall_hit_rate = sum(all_hit_rate) / len(dataset)
    record = {
        "num_samples": len(dataset),
        "avg_hits": sum(all_hits) / len(dataset),
        "avg_evidence": sum(all_evidence) / len(dataset),
        "avg_chunk": sum(all_num_chunk)/len(dataset),
        "avg_hit_rate": overall_hit_rate,
    }
    with open(os.path.join(output_dir, "statistic.json"), "w") as f:
        json.dump(record, f, indent=4)
    print(f"Overall hit rate: {overall_hit_rate}")

def main():
    parser = argparse.ArgumentParser(description="Process dataset with Qwen3 embeddings and OpenAI comparison")
    parser.add_argument("--input_file", required=True, help="Input JSONL file path")
    parser.add_argument("--output_dir", required=True, help="Output directory to save results")
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-Embedding-4B", help="Qwen3 model name")
    parser.add_argument("--openai_model", default="deepseek-v3", help="OpenAI model for comparison")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for embedding generation")
    parser.add_argument("--top_k", type=int, default=-1, help="Number of top chunks to select per evidence (-1 for all chunks)")
    parser.add_argument("--compare_method", type=str, default="similarity", choices=["llm", "string", "similarity"], help="Chunk comparison method: 'llm' for OpenAI comparison, 'string' for string equality, 'similarity' for cosine similarity >= threshold")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for cosine similarity")
    
    args = parser.parse_args()

    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Ensure the OpenAI API key is set
    # It's better to set this as an environment variable before running the script
    if 'OPENAI_API_KEY' not in os.environ:
    #    os.environ['OPENAI_API_KEY'] = "sk-Oxo5Zm17u8W97eMs4l3ZlKqc8TxQ9wF4TpYg1HsDo5Tyry6q" # Fallback if not set
    #    os.environ['OPENAI_API_KEY'] = "sk-0af0d9350cfe4ce1b6a3b5ef37f0ae32" # my key
       os.environ['OPENAI_API_KEY'] = "sk-4768b45eb65f407790a619db44c37f32" # tzh key
       print("Warning: OPENAI_API_KEY not found in environment. Using a placeholder.")

    # Determine GPU distribution
    global gpu_assignments
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        print(f"Found {num_gpus} GPU(s)")
        # Distribute workers evenly across GPUs
        workers_per_gpu = args.num_workers // num_gpus
        remaining_workers = args.num_workers % num_gpus
        
        gpu_assignments = []
        for gpu_id in range(num_gpus):
            # Assign base number of workers to each GPU
            num_workers_for_gpu = workers_per_gpu
            # Distribute remaining workers to first few GPUs
            if gpu_id < remaining_workers:
                num_workers_for_gpu += 1
            gpu_assignments.extend([gpu_id] * num_workers_for_gpu)
        
        print(f"Worker distribution across GPUs:")
        for gpu_id in range(num_gpus):
            count = gpu_assignments.count(gpu_id)
            print(f"  GPU {gpu_id}: {count} workers")
    else:
        print("No GPU found, using CPU")
        gpu_assignments = [None] * args.num_workers
        
    print(f"{gpu_assignments=}")

    # Load dataset
    print("Loading dataset...")
    dataset = load_json_dataset(args.input_file)
    # dataset = dataset.select(range(10)) # Uncomment for quick testing
    
    processed_data = []
    
    # Create worker pool with GPU assignments
    print(f"Starting parallel processing with {args.num_workers} workers...")
    
    # Create initializer with args and gpu_assignments
    initializer = partial(worker_init, args, gpu_assignments)
    
    with Pool(processes=args.num_workers, initializer=initializer) as pool:
        with tqdm(total=len(dataset), desc="Processing samples") as pbar:
            # imap_unordered is generally more efficient
            for result in pool.imap_unordered(process_sample, dataset):
                if result:  # Only append successful results
                    processed_data.append(result)
                pbar.update(1)

    # Convert results back to a Hugging Face Dataset
    processed_dataset = Dataset.from_list(processed_data)
    
    # Save results
    save_results(processed_dataset, args.output_dir)

if __name__ == "__main__":
    main()

