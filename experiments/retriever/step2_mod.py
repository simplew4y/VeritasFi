"""
Step 2 (Modified):
    1. Recall from rewritten questions (and optional HYDE chunks)
    2. Save retrieved chunks along with metadata using Hugging Face Datasets

Input
    JSON file with structure (see `step2.py` for details)

Output
    JSON file containing:
        {
            "question": "original question",
            "rewritten": ["rewritten question 1", ...],
            "hyde_ppl": 0.0,
            "num_recalls": 0,
            "query_chunks": ["retrieved chunk 1", ...],
            "retriever_chunks": ["FAISS", "Title Summary", "BM25"],
            "score_chunks": [0.0, ...],
            "evidence": ["evidence 1", ...],
            "answer": "answer"
        }
"""

import argparse
import json
import os
import sys
from typing import List

import yaml
from datasets import Dataset, load_dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.ensembleRetriever import EnsembleRetriever


def load_json_file(file_path: str) -> List[dict]:
    dataset = load_dataset("json", data_files=file_path, split="train")
    return dataset.to_list()


def save_results_as_dataset(data: List[dict], file_path: str) -> None:
    dataset = Dataset.from_list(data)
    dataset.to_json(file_path, lines=True, force_ascii=False)


def collect_query_chunks(
    file_path: str,
    retriever: EnsembleRetriever,
    args: argparse.Namespace,
) -> List[dict]:
    data = load_json_file(file_path)

    results = []

    for idx, entry in enumerate(tqdm(data, desc="Collecting query chunks")):
        try:
            rewritten = entry.get("rewritten", [])
            hyde = entry.get("hyde", [])
            hyde_ppl = entry.get("perplexity", -1)
            query_chunks: List[str] = []
            retriever_chunks: List[str] = []
            score_chunks: List[float] = []

            if isinstance(rewritten, str):
                rewritten = [rewritten]

            if args.enable_hyde:
                if not hyde:
                    print(
                        f"Warning: HYDE enabled but no hyde content for question index {idx}"
                    )
                for q, h in zip(rewritten, hyde):
                    chunks = retriever.invoke(q, h)
            else:
                for q in rewritten:
                    chunks = retriever.invoke(q, [])

            query_chunks.extend(chunk["page_content"].strip() for chunk in chunks)
            retriever_chunks.extend(chunk["retriever"] for chunk in chunks)
            score_chunks.extend(chunk["score"] for chunk in chunks)

            results.append(
                {
                    "question": entry.get("question"),
                    "rewritten": rewritten,
                    "hyde_ppl": hyde_ppl,
                    "num_recalls": len(query_chunks),
                    "query_chunks": query_chunks,
                    "retriever_chunks": retriever_chunks,
                    "score_chunks": score_chunks,
                    "evidence": entry.get("evidence", []),
                    "answer": entry.get("answer", ""),
                }
            )

        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error processing entry {idx}: {exc}")
            print(f"Entry: {json.dumps(entry, ensure_ascii=False)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect retrieval chunks and save as Hugging Face dataset JSON"
    )
    parser.add_argument("--input", type=str, default="hyde.json", help="Input JSON file path")
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_chunks/result.json",
        help="Output JSON file path",
    )
    parser.add_argument("--faiss_k", type=int, default=40, help="Faiss topk")
    parser.add_argument("--bm25_k", type=int, default=10, help="BM25 topk")
    parser.add_argument("--faiss_ts_k", type=int, default=10, help="Faiss topk for title and snippet")
    parser.add_argument("--enable_expand", action="store_true", help="Expand chunk content")
    parser.add_argument("--enable_hyde", action="store_true", help="Enable Hyde")
    parser.add_argument("--config_file", type=str, required=True, help="Config file name")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    config_path = os.getenv(
        "CONFIG_PATH", os.path.join(project_root, "config", args.config_file)
    )
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    persist_directory = os.path.join(config["persist_directory"], "chroma")
    if "finder" in config_path.lower():
        collection_name = "final_base_jsons"
    elif "zeekr" in config_path.lower():
        collection_name = "zeekr" 
    elif "finqa" in config_path.lower():
        collection_name = "jsondata_for_database"
    else:
        collection_name = "lotus"
    embeddings = HuggingFaceEmbeddings(model_name=config["embeddings_model_name"])

    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        relevance_score_fn="l2",
    )

    ts_chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.path.join(config["persist_directory"], "ts_chroma"),
        relevance_score_fn="l2",
    )

    bm25_dir = os.path.join(config["persist_directory"], "bm25_index", collection_name)
    retriever = EnsembleRetriever(
        bm25_dir,
        chroma,
        ts_chroma,
        10,
        embeddings,
        faiss_k=args.faiss_k,
        bm25_k=args.bm25_k,
        faiss_ts_k=args.faiss_ts_k,
        enable_expand=args.enable_expand,
    )

    results = collect_query_chunks(
        file_path=args.input,
        retriever=retriever,
        args=args,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_results_as_dataset(results, args.output)
    print(f"Saved retrieval results to {args.output}")


if __name__ == "__main__":
    main()
