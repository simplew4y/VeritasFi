import os
os.environ['OPENAI_API_KEY'] = ""
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pathlib import Path
import asyncio
import json
import pandas as pd
from glob import glob
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall, LLMContextPrecisionWithoutReference
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.dataset_schema import SingleTurnSample

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
deepseek_llm = ChatOpenAI(
    model="deepseek-v3",
    openai_api_base="https://api.lkeap.cloud.tencent.com/v1",
    openai_api_key=""
)

evaluator_llm = LangchainLLMWrapper(deepseek_llm)
evaluator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# OpenAIEmbeddings(client=openai_client)

faithfulness = Faithfulness(llm=evaluator_llm)
response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
context_recall = LLMContextRecall(llm=evaluator_llm)
context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
# context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
def split_question(text):
    count = text.count("?")
    
    if count <= 1:
        return [text.strip()]
    
    parts = [q.strip() for q in text.split("?") if q.strip()]
    return [q + "?" for q in parts]

def load_zeekr_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    gt_map = {}
    for rec in gt_list:
        key = " ".join(rec.get("rewritten_question"))
        gt_map[key] = rec

    return gt_map

def load_lotus_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    gt_map = {}
    for rec in gt_list:
        key = rec.get("original_question")
        gt_map[key] = rec

    return gt_map

def load_parquet_ground_truth(gt_path):
    df = pd.read_parquet(gt_path)

    if "text" not in df.columns or "answer" not in df.columns:
        raise ValueError("Parquet file must contain 'text' and 'answer' columns")

    gt_map = {}
    for _, row in df.iterrows():
        key = str(row["text"]).strip()
        gt_map[key] = {
            "original_question": key,
            "answer": row["answer"]
        }

    return gt_map

def load_csv_ground_truth(gt_path):
    df = pd.read_csv(gt_path)

    required_cols = ["Query", "Response"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain {required_cols}")

    gt_map = {}
    for _, row in df.iterrows():
        key = str(row["Query"]).strip()
        gt_map[key] = {
            "original_question": key,
            "answer": row["Response"],
            # "context": row.get("Context", None),
            # "category": row.get("Category", None),
            # "filename": row.get("Filename", None),
            # "source": row.get("Source", None),
        }

    return gt_map

def load_jsonl_ground_truth(gt_path):
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("question", "").strip()
            if not key:
                continue
            gt_map[key] = {
                "original_question": key,
                "answer": rec.get("answer"),
                # "financebench_id": rec.get("financebench_id"),
                # "company": rec.get("company"),
                # "doc_name": rec.get("doc_name"),
                # "question_type": rec.get("question_type"),
                # "question_reasoning": rec.get("question_reasoning"),
                # "justification": rec.get("justification"),
                # "dataset_subset_label": rec.get("dataset_subset_label"),
                # "evidence": rec.get("evidence", []),
            }
    return gt_map

async def single_question_eval(sample: SingleTurnSample, semaphore: asyncio.Semaphore):
    """
    Evaluate single question
    """
    async with semaphore:
        try:
            f_score = await faithfulness.single_turn_ascore(sample)
        except Exception as e:
            print(f"Faithfulness failed for question '{sample.user_input[:20]}...': {e}")
            f_score = 0

        try:
            rr_score = await response_relevancy.single_turn_ascore(sample)
        except Exception as e:
            print(f"Response Relevancy failed for question '{sample.user_input[:20]}...': {e}")
            rr_score = 0

        try:
            cr_score = await context_recall.single_turn_ascore(sample)
        except Exception as e:
            print(f"Context Recall failed for question '{sample.user_input[:20]}...': {e}")
            cr_score = 0

        try:
            cp_score = await context_precision.single_turn_ascore(sample)
        except Exception as e:
            print(f"Context Precision failed for question '{sample.user_input[:20]}...': {e}")
            cp_score = 0

    result_entry = {
        "question": sample.user_input,
        "reference_answer": sample.reference,
        "response": sample.response,
        "retrieved_contexts": sample.retrieved_contexts,
        "scores": {
            "faithfulness": float(f_score),
            "response_relevancy": float(rr_score),
            "context_recall": float(cr_score),
            "context_precision": float(cp_score),
        }
    }
    
    print(f"✅ Finished: {sample.user_input[:50]}...")
    return result_entry


async def evaluate_all(eval_dir, gt_map, output_json_path=None, max_concurrent_tasks=60):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    tasks = []
    total_samples = 0

    json_paths = glob(os.path.join(eval_dir, "question_*.json"))
    print(f"Found files:\n{json_paths}")
    
    for p in json_paths:
        print(f"-----collecting tasks from {p}-----")
        with open(p, 'r', encoding='utf-8') as f:
            rec = json.load(f)

        for q in rec.get("questions", []):
            orig_q = q.get("original_question")
            answer = q.get("answer")
            key = orig_q

            if key not in gt_map:
                print(f"[WARN] question key {key} not in ground truth; skip.")
                continue

            gt = gt_map[key]
            reference_answer = gt.get("answer")

            retrieved = [ri.get("chunk_content") for ri in q.get("rag_info", []) if ri.get("chunk_content")][:10]

            sample = SingleTurnSample(
                user_input=orig_q,
                response=answer,
                reference=reference_answer,
                retrieved_contexts=retrieved,
            )
            
            task = asyncio.create_task(single_question_eval(sample, semaphore))
            tasks.append(task)
            total_samples += 1

    if total_samples == 0:
        print("No samples collected for evaluation.")
        return

    print(f"\n--- Starting evaluation for {total_samples} samples with max concurrency of {max_concurrent_tasks} ---")
    
    results = await asyncio.gather(*tasks)
    
    sum_f = 0.0
    sum_rr = 0.0
    sum_cr = 0.0
    sum_cp = 0.0
    count = 0
    
    final_results = []
    
    for result_entry in results:
        if result_entry is None:
            continue
        
        scores = result_entry["scores"]
        
        sum_f += scores["faithfulness"]
        sum_rr += scores["response_relevancy"]
        sum_cr += scores["context_recall"]
        sum_cp += scores["context_precision"]
        count += 1
        final_results.append(result_entry)


    if count == 0:
        print("No successful samples evaluated.")
        return

    summary = {
        "num_samples": count,
        "avg_faithfulness": sum_f / count,
        "avg_response_relevancy": sum_rr / count,
        "avg_context_recall": sum_cr / count,
        "avg_context_precision": sum_cp / count,
    }

    output_data = {
        "results": final_results,
        "summary": summary
    }

    if not output_json_path:
        output_json_path = os.path.join("./", "ragas_eval_results.json")

    output_file = Path(output_json_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n--- Done ---")
    print(f"Results saved to {output_json_path}")
    print("Result Summary:", summary)
    return summary



if __name__ == "__main__":
    gt_map = load_csv_ground_truth("/path/to/ground_truth.csv")

    result = asyncio.run(evaluate_all(
        eval_dir="/path/to/eval_dir",
        gt_map = gt_map,
        output_json_path="/path/to/eval_results.json",
    ))