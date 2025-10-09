'''
Continuous retrieval benchmark script - runs retrieval non-stop without file writing or LLM judging.
Only performs embedding-based retrieval to measure throughput and latency.
'''

import json
import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import time

# Set available GPUs to 0 and 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# --- Global variables for worker processes ---
worker_qwen_model = None
worker_qwen_tokenizer = None
worker_device = None

# Global GPU assignments list (set before creating pool)
gpu_assignments = []

def worker_init(args, gpu_assignments_list):
    """
    Initializes models for each worker process.
    This function is called once per worker when the Pool is created.
    Args:
        args: Command line arguments
        gpu_assignments_list: List of GPU assignments for each worker
    """
    global worker_qwen_model, worker_qwen_tokenizer, worker_device

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
    worker_qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model, 
        trust_remote_code=True, 
        padding_side='left', 
        cache_dir="/work/xinyu/RAG_Agent/model"
    )
    worker_qwen_model = AutoModel.from_pretrained(
        args.qwen_model, 
        trust_remote_code=True, 
        cache_dir="/work/xinyu/RAG_Agent/model", 
        device_map=worker_device
    )
    worker_qwen_model.eval()

def process_sample(example):
    """
    Processes a single sample - performs retrieval only without LLM judging.
    Returns retrieval statistics.
    """
    global worker_qwen_model, worker_qwen_tokenizer, worker_device

    try:
        start_time = time.time()
        
        question = example.get('question')
        evidence_list = example.get('evidence', [])
        query_chunks = example.get('query_chunks', [])
        
        if not evidence_list or not query_chunks:
            return {
                'status': 'skipped',
                'num_evidences': 0,
                'num_chunks': 0,
                'retrieval_time': 0.0
            }
        
        # For each evidence, select top 3 chunks
        total_retrievals = 0
        for evidence in evidence_list:
            top_chunks = select_top_chunks(
                evidence, 
                query_chunks, 
                worker_qwen_model, 
                worker_qwen_tokenizer, 
                worker_device, 
                top_k=3, 
                batch_size=16
            )
            total_retrievals += len(top_chunks)
        
        retrieval_time = time.time() - start_time
        
        return {
            'status': 'success',
            'num_evidences': len(evidence_list),
            'num_chunks': len(query_chunks),
            'total_retrievals': total_retrievals,
            'retrieval_time': retrieval_time
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

def load_json_dataset(file_path):
    """Load dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    """Get embeddings using Qwen3 model with batching to prevent OOM"""
    if not texts:
        return np.array([])
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all batch embeddings
    return np.vstack(all_embeddings)

def select_top_chunks(evidence, query_chunks, model, tokenizer, device, top_k=3, batch_size=32):
    """Select top-k query chunks with highest cosine similarity to evidence"""
    if not query_chunks:
        return []
    
    evidence_emb = get_embeddings([evidence], model, tokenizer, device, batch_size=1)
    chunks_emb = get_embeddings(query_chunks, model, tokenizer, device, batch_size=batch_size)
    
    similarities = cosine_similarity(evidence_emb, chunks_emb)[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_chunks = [query_chunks[i] for i in top_indices]
    
    return top_chunks

def print_statistics(results, elapsed_time):
    """Print retrieval statistics"""
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful retrievals")
        return
    
    total_samples = len(successful)
    total_retrievals = sum(r['total_retrievals'] for r in successful)
    total_retrieval_time = sum(r['retrieval_time'] for r in successful)
    avg_retrieval_time = total_retrieval_time / total_samples
    
    print(f"\n{'='*60}")
    print(f"Retrieval Statistics")
    print(f"{'='*60}")
    print(f"Total samples processed: {total_samples}")
    print(f"Total retrievals: {total_retrievals}")
    print(f"Average retrieval time per sample: {avg_retrieval_time:.4f}s")
    print(f"Throughput: {total_samples / elapsed_time:.2f} samples/sec")
    print(f"Total elapsed time: {elapsed_time:.2f}s")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Continuous retrieval benchmark without file writing or LLM judging")
    parser.add_argument("--input_file", required=True, help="Input JSONL file path")
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-Embedding-4B", help="Qwen3 model name")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations (None for infinite)")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of samples per iteration")
    
    args = parser.parse_args()

    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

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
    
    if args.sample_limit:
        dataset = dataset.select(range(min(args.sample_limit, len(dataset))))
        print(f"Limited to {len(dataset)} samples")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create initializer with args and gpu_assignments
    initializer = partial(worker_init, args, gpu_assignments)
    
    # Create worker pool
    print(f"Starting parallel processing with {args.num_workers} workers...")
    pool = Pool(processes=args.num_workers, initializer=initializer)
    
    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"\n{'#'*60}")
            print(f"Starting iteration {iteration}")
            print(f"{'#'*60}")
            
            start_time = time.time()
            results = []
            
            with tqdm(total=len(dataset), desc=f"Iteration {iteration}") as pbar:
                for result in pool.imap_unordered(process_sample, dataset):
                    results.append(result)
                    pbar.update(1)
            
            elapsed_time = time.time() - start_time
            print_statistics(results, elapsed_time)
            
            # Check if we should stop
            if args.iterations is not None and iteration >= args.iterations:
                print(f"Completed {args.iterations} iterations. Stopping.")
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down...")
    finally:
        pool.close()
        pool.join()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
