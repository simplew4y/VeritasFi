import torch
import threading
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque
from datetime import datetime
import json
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.ragManager import RAGManager
from utils.vllmChatService import ChatService

question_file = ""
with open(question_file, 'r', encoding='utf-8') as f:
    questions = [q.strip() for q in f.readlines() if q.strip()]

# Global statistics variables
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_inference_calls = 0
        self.inference_calls_per_model = {}
        self.recent_inference_calls = deque(maxlen=100)  # Store recent timestamps for throughput calculation
        self.start_time = None  # Will be set after warm-up period
        self.warm_up_complete = False
        self.warm_up_start_time = time.time()
        self.warm_up_period = 30  # 30 seconds warm-up period
    
    def add_inference_call(self, model_name):
        with self.lock:
            # Check if we're still in warm-up period
            current_time = time.time()
            if not self.warm_up_complete:
                if current_time - self.warm_up_start_time >= self.warm_up_period:
                    # Warm-up period is over, start counting from now
                    self.warm_up_complete = True
                    self.start_time = current_time
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warm-up period complete. Starting to collect metrics.")
                else:
                    # Still in warm-up, don't count this call
                    return
            
            # Count this call since warm-up is complete
            self.total_inference_calls += 1
            if model_name not in self.inference_calls_per_model:
                self.inference_calls_per_model[model_name] = 0
            self.inference_calls_per_model[model_name] += 1
            self.recent_inference_calls.append((current_time, 1))
    
    def get_stats(self):
        with self.lock:
            # If we're still in warm-up period, show that
            if not self.warm_up_complete:
                current_time = time.time()
                warm_up_elapsed = current_time - self.warm_up_start_time
                remaining = max(0, self.warm_up_period - warm_up_elapsed)
                return {
                    "in_warm_up": True,
                    "warm_up_remaining": remaining,
                    "total_inference_calls": 0,
                    "inference_calls_per_model": {},
                    "overall_rate": 0,
                    "recent_rate": 0,
                    "elapsed_time": 0
                }
            
            # Calculate stats after warm-up period
            elapsed = time.time() - self.start_time if self.start_time else 0
            overall_rate = self.total_inference_calls / elapsed if elapsed > 0 else 0
            
            # Calculate recent throughput (last 100 inference calls)
            recent_rate = 0
            if self.recent_inference_calls:
                oldest_time = self.recent_inference_calls[0][0]
                newest_time = self.recent_inference_calls[-1][0]
                recent_calls = sum(calls for _, calls in self.recent_inference_calls)
                recent_elapsed = newest_time - oldest_time
                if recent_elapsed > 0:
                    recent_rate = recent_calls / recent_elapsed
            
            return {
                "in_warm_up": False,
                "total_inference_calls": self.total_inference_calls,
                "inference_calls_per_model": dict(self.inference_calls_per_model),
                "overall_rate": overall_rate,
                "recent_rate": recent_rate,
                "elapsed_time": elapsed
            }

# Initialize global stats
stats = Stats()

def get_inputs(pairs, tokenizer, device='cuda:0', prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    # Pad the inputs and move to the specified device
    padded_inputs = tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )
    
    # Move all tensors to the specified device
    for key in padded_inputs:
        if isinstance(padded_inputs[key], torch.Tensor):
            padded_inputs[key] = padded_inputs[key].to(device)
    
    return padded_inputs
# Model configurations - all using GPU
MODEL_CONFIGS = [
    {'name': 'model_name', 'model_id': 'path/to/your/model'},
    
]

def generate_random_pairs(num_pairs=155):
    """Generate random question-chunk pairs for scoring"""
    random_question = random.choice(questions)
    random_chunks = random.choices(chunks, k=num_pairs)
    return [(random_question, chunk) for chunk in random_chunks]

def batch_pairs(pairs, batch_size=4):  # Reduced batch size to fit in memory
    """Split pairs into batches of specified size"""
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]

def print_stats():
    """Print current statistics"""
    while True:
        current_stats = stats.get_stats()
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performance Statistics:")
        
        # Check if we're still in warm-up period
        if current_stats.get('in_warm_up', False):
            print(f"WARM-UP PERIOD: {current_stats['warm_up_remaining']:.1f} seconds remaining")
            print("Metrics will start being collected after warm-up period completes.")
        else:
            # Regular stats display after warm-up
            print(f"Total inference calls: {current_stats['total_inference_calls']}")
            print(f"Elapsed time: {current_stats['elapsed_time']:.2f} seconds")
            print(f"Overall rate: {current_stats['overall_rate']:.2f} inference calls/second")
            print(f"Recent rate: {current_stats['recent_rate']:.2f} inference calls/second")
            print("Inference calls per model:")
            for model, count in current_stats['inference_calls_per_model'].items():
                print(f"  {model}: {count}")
        
        time.sleep(10)  # Update stats every 10 seconds

def model_worker(config):
    """Worker function to run a model in a separate thread"""
    
    model_name = config['name']
    model_id = config['model_id']
    model = config['model']
    retriever = config['retriever']
    chat_manager = config['chat_manager']
    device = config.get('device', 'cuda:0')
    tokenizer = config['tokenizer']  # Get pre-loaded tokenizer
    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
    
    print(f"{model_name} initialized and ready on {device}")
    
    # Run inference in an infinite loop
    while True:
        try:
            question = random.choice(questions)
            chunks = retriever.invoke(question, [])
            all_pairs = [(question, chunk['page_content']) for chunk in chunks]
            # print(all_pairs[0])
            # input()
            
            # Process batches
            batch_count = 0
            for batch in batch_pairs(all_pairs, batch_size=8):  # Smaller batch size to fit in memory
                try:
                    # Compute scores for this batch
                    with torch.no_grad():
                        # Pass the device to get_inputs to ensure tensors are on the right device
                        inputs = get_inputs(batch, tokenizer, device=device)
                        # Ensure we're using the correct data type and device
                        logits = model(**inputs, return_dict=True).logits
                        # Handle tensor type explicitly
                        if logits.dtype != torch.float32:
                            logits = logits.to(torch.float32)
                        # Get scores safely - ensure yes_loc is on the same device
                        yes_loc_tensor = torch.tensor(yes_loc, device=logits.device)
                        scores = logits[:, -1, yes_loc_tensor].view(-1)
                    
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in batch processing for {model_name}: {str(e)}")
                    # Continue with next batch instead of failing the entire loop
                    continue

            mtx = retriever.compute_similarity_mtx([chunk['page_content'] for chunk in chunks])
            
            stats.add_inference_call(model_name)
            
            torch.cuda.empty_cache()

            print(f"{model_name} processed {batch_count} batches with {len(all_pairs)} pairs")
            
        except Exception as e:
            print(f"Error in {model_name}: {str(e)}")
            time.sleep(1)  # Prevent tight loop in case of recurring errors

# Start the stats printing thread
stats_thread = threading.Thread(target=print_stats, daemon=True)
stats_thread.start()

config_path = os.getenv('CONFIG_PATH', '../../config/production.yaml')
with open(config_path, 'r') as file:
    rag_config = yaml.safe_load(file)

collections = {'lotus': 10}
rag_manager = RAGManager(config=rag_config, collections=collections)
rag_manager.create_collection("lotus")

# Pre-load all models and tokenizers to GPU
print("Pre-loading all models to GPU...")
device = 'cuda:0'
for config in MODEL_CONFIGS:
    model_name = config['name']
    model_id = config['model_id']
    print(f"Loading {model_name} with {model_id} to {device}...")
    
    # Load with half precision to save memory
    config['tokenizer'] = AutoTokenizer.from_pretrained(model_id)
    config['model'] = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map=device
    )
    config['model'].eval()
    config['retriever'] = rag_manager.create_retriever(10, "lotus", retriever_type="ensemble")
    chat_service = ChatService(rag_config, None, 5)
    config['chat_manager'] = chat_service.get_or_create_chat_manager(config['name'])
    print(f"{model_name} loaded successfully")

# Start model worker threads
model_threads = []
for config in MODEL_CONFIGS:
    thread = threading.Thread(target=model_worker, args=(config,), daemon=True)
    thread.start()
    model_threads.append(thread)
    time.sleep(0.5)  # Small delay between thread starts

# Function to save stats to a file
def save_stats_to_file(filename):
    current_stats = stats.get_stats()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Format the stats as a readable text file
    with open(filename, 'w') as f:
        f.write(f"Test Reranker Statistics - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Runtime: {current_stats['elapsed_time']:.2f} seconds\n")
        f.write(f"Total Inference Calls: {current_stats['total_inference_calls']}\n")
        f.write(f"Overall Rate: {current_stats['overall_rate']:.2f} inference calls/second\n")
        f.write(f"Recent Rate: {current_stats['recent_rate']:.2f} inference calls/second\n\n")
        
        f.write("Inference Calls Per Model:\n")
        for model, count in current_stats['inference_calls_per_model'].items():
            f.write(f"  {model}: {count}\n")
    
    print(f"\nFinal statistics saved to {filename}")

# Run for exactly 10 minutes (600 seconds)
print(f"\nTest will run for 10 minutes and then save results to 'reranker_stats.txt'")
start_time = time.time()
test_duration = 1200  # 10 minutes in seconds

try:
    while time.time() - start_time < test_duration:
        time.sleep(1)
    
    print("\nTest completed after 10 minutes.")
    # Save the final stats to a file
    save_stats_to_file(f"stress_stats_{len(MODEL_CONFIGS)}models.txt")
    print("Shutting down...")
    
except KeyboardInterrupt:
    print("\nTest interrupted before completion.")
    # Still save whatever stats we have
    save_stats_to_file(f"stress_stats_{len(MODEL_CONFIGS)}models.txt")
    print("Shutting down...")
    
# The daemon threads will be automatically terminated when the main thread exits
