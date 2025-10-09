import os
os.environ['OPENAI_API_KEY'] = ""
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pathlib import Path
import asyncio
import json
import pandas as pd
from glob import glob
from ragas import Dataset, experiment
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._factual_correctness import FactualCorrectness
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

scorer = FactualCorrectness(llm = evaluator_llm)

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
        # key = rec.get("original_question")
        gt_map[key] = rec

    return gt_map

def load_lotus_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    gt_map = {}
    for rec in gt_list:
        # key = " ".join(rec.get("rewritten_question"))
        key = rec.get("rewritten")
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

async def single_question_eval(sample: SingleTurnSample, fname: str, semaphore: asyncio.Semaphore):
    """
    对单个问题进行 RAGAS 评分，并使用信号量进行并发控制。
    """
    # 获取信号量（如果当前并发数已达上限，会在此等待）
    async with semaphore:
        # 在这里执行需要并发控制的 I/O 密集型操作 (RAGAS 评分)
        try:
            correctness_score = await scorer.single_turn_ascore(sample)
        except Exception as e:
            print(f"Correctness score failed for question '{sample.user_input[:20]}...': {e}")
            correctness_score = 0
            
    # 信号量在此自动释放

    result_entry = {
        "filename": fname,
        "question": sample.user_input,
        "reference_answer": sample.reference,
        "response": sample.response,
        "retrieved_contexts": sample.retrieved_contexts,
        "scores": {
            "correctness": correctness_score
        }
    }
    
    print(f"✅ Finished: {sample.user_input[:50]}...")
    return result_entry


async def evaluate_all(eval_dir, gt_map, output_json_path=None, max_concurrent_tasks=80):
    
    # 初始化信号量，限制最大并发数为 max_concurrent_tasks
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    tasks = [] # 存储所有任务
    total_samples = 0

    json_paths = glob(os.path.join(eval_dir, "question_*.json"))
    print(f"Found files:\n{json_paths}")
    
    for p in json_paths:
        print(f"-----collecting tasks from {p}-----")
        with open(p, 'r', encoding='utf-8') as f:
            rec = json.load(f)

        for q in rec.get("questions", []):
            orig_q = q.get("original_question").replace("\n", " ").strip()
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
            
            # 创建异步任务并将其添加到任务列表中
            task = asyncio.create_task(single_question_eval(sample, p, semaphore))
            tasks.append(task)
            total_samples += 1

    if total_samples == 0:
        print("No samples collected for evaluation.")
        return

    print(f"\n--- Starting evaluation for {total_samples} samples with max concurrency of {max_concurrent_tasks} ---")
    
    # 并行运行所有任务，等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    # 汇总结果
    sum_cr = 0.0
    count = 0
    
    final_results = []
    
    for result_entry in results:
        if result_entry is None:
            continue
        
        scores = result_entry["scores"]
        
        sum_cr += scores["correctness"]
        count += 1
        final_results.append(result_entry)


    if count == 0:
        print("No successful samples evaluated.")
        return

    summary = {
        "num_samples": count,
        "avg_correctness": sum_cr / count,
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
    # gt_map = load_lotus_ground_truth("/root/autodl-tmp/RAG_Agent_componentTest/RAGAS_eval/gt_data/lotus_gt_1003.json")
    gt_map = load_lotus_ground_truth("/root/autodl-tmp/RAG_Agent_componentTest/RAGAS_eval/gt_data/109_lotus_gt.json")
    # gt_map = load_zeekr_ground_truth("/root/autodl-tmp/RAG_Agent_componentTest/RAGAS_eval/gt_data/question_zeekr_gt.json")
    # gt_map = load_jsonl_ground_truth("/root/autodl-tmp/RAG_Agent_componentTest/RAGAS_eval/gt_data/financebench.jsonl")
    # gt_map = load_csv_ground_truth("/root/autodl-tmp/RAG_Agent_componentTest/RAGAS_eval/gt_data/finqabench.csv")
    # gt_map = load_parquet_ground_truth("/root/autodl-tmp/RAG_Agent_componentTest/RAGAS_eval/gt_data/FinDer_sampled.parquet")

    result = asyncio.run(evaluate_all(
        # eval_dir="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinQA/test",
        # eval_dir="/root/autodl-tmp/RAG_Agent_componentTest/e2e_test_results/finDER_bm10",
        # eval_dir="/root/autodl-tmp/RAG_Agent_componentTest/e2e_test_results/finDER_faiss10",
        # eval_dir="/root/autodl-tmp/RAG_Agent_componentTest/e2e_test_results/finDER_faissbm10",
        # eval_dir="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinQA/finqa_questions",
        # eval_dir="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinQA/finqa_questions_cp",
        # eval_dir="/root/autodl-tmp/hyc_production/RAG_Agent/src/test/test_questions/lotus_all/other_questions",
        # eval_dir="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinanceBench/financebench_questions_cp",
        # eval_dir="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinDer/finder_questions",
        # eval_dir="/root/autodl-tmp/RAG_Agent_componentTest/e2e_test_results/finQA_faissbm10",
        eval_dir="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_LightRAG/lotus_q",
        gt_map = gt_map,
        # output_json_path="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinQA/finqa_correctness_bm25_wo_ffp.json"
        # output_json_path="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinDer/finder_correctness_veritasFi_2.json"
        # output_json_path="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinQA/finqa_correctness_veritasFi_2.json"
        # output_json_path="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinQA/finqa_correctness_faissbm25.json"
        # output_json_path="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_FinanceBench/financebench_correctness_veritasFi2.json"
        # output_json_path="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_GraphRAG/woffp_zeekr_correctness_graphrag.json",
        output_json_path="/root/autodl-tmp/dir_whw/RAGAS/RAGAS_LightRAG/lotus_correctness_lightrag.json",
    ))