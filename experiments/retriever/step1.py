'''
Step 1: Hyde
    1. Calculate the Perplexity of hyde
    2. Get hyde content

Input
    JSON file with the following structure:
    [
        {
            "question": "original question",
            "rewritten": ["rewritten question 1", "rewritten question 2"]
        },
        ...
    ]

Output
    Json with the following structure:
    {
        "question": "original question",
        "rewritten": "rewritten question",
        "hyde": ["hyde 1", "hyde 2", ... ],
        "perplexity": 0.0,
        "evidence": ["evidence 1", "evidence 2", ... ],
        "answer": "answer"
    }

'''

import math
import numpy as np
from openai import OpenAI
import os
import json
import logging
import sys
from tqdm import tqdm
import argparse
from datasets import load_dataset

logging.basicConfig(
    filemode='w',
    filename='step1.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.prompts.hyde import get_hypo_sys_prompt

def load_json_file(file_path):
    data = load_dataset('json', data_files=file_path, split='train')
    return data.to_list()
    # with open(file_path, "r") as f:
    #     return json.load(f)

def save_json_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f,ensure_ascii=False, indent=4)

def get_log_probs(prompt, model, client):
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_hypo_sys_prompt(3)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,           # Keep at 0 for deterministic output
                top_p=1.0,                # Add this to disable nucleus sampling
                frequency_penalty=0.0,     # Change from 0.5 to 0 for consistency
                presence_penalty=0.0,      # Add this to disable presence penalty
                seed=42,                  # Add a fixed seed for reproducibility
                logprobs=True             # You can keep this as is
            )

            log_probs = response.choices[0].logprobs.content
            content = response.choices[0].message.content
            return content, [log_prob.logprob for log_prob in log_probs]
        except Exception as e:
            err = e
            continue
    raise err

def calculate_perplexity(log_probs):
    average_log_prob = np.mean(log_probs)
    return math.exp(-average_log_prob)

def hyde_rewritten(file_path, model, client, output_file: str):
    data = load_json_file(file_path)
    results = []

    for idx, entry in tqdm(enumerate(data)):
        try:
            rewritten = entry.get("rewritten", "")

            if rewritten == "":
                logging.warning(f"Skipping entry {idx}: No rewritten questions found")
                continue

            rewritten = [rewritten] if type(rewritten) == str else rewritten

            evidence = [e["evidence_text"] for e in entry.get("evidence", [])]

            hydes = []
            ppls = []
            for q in rewritten:
                logging.info(f"Evaluation questions: {q}")
                content, log_probs = get_log_probs(q, model, client)
                hyde = [chunk.strip() for chunk in content.split("ANSWER:")[1:]]
                hydes.append(hyde)
                if len(hyde) == 0:
                    logging.warning(f"No hyde content found for question: {q}")
                logging.info(f"Start to calculate")
                perplexity = calculate_perplexity(log_probs)
                ppls.append(perplexity)
            results.append({
                "question": entry.get("question"),
                "rewritten": rewritten,
                "hyde": hydes,
                "perplexity": ppls,
                "evidence": evidence,
                "answer": entry.get("answer"),
            })
            save_json_file(results, output_file)
        except Exception as e:
            logging.warning(f"Error processing entry {idx}: {e}")

def main():
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='Evaluate rewritten questions using hyde models')
    parser.add_argument('--input', type=str, default='answer/75.json', help='Input JSON file path')
    parser.add_argument('--output', type=str, default='hyde_results_75.json', help='Output file path')
    parser.add_argument('--model_name', type=str, default='hyde-lora', help='Hyde model name')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default='http://localhost:8001/v1', help='OpenAI API base url')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    model_name = args.model_name
    api_key = args.api_key
    base_url = args.base_url

    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    hyde_rewritten(input_file, model_name, client, output_file)
    logging.info(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    main()
