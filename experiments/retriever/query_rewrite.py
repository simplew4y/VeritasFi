#!/usr/bin/env python3
import ast
import openai
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from datasets import load_dataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_if_query_rag_prompt(question, qa_history):
    nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    year = 2024
    return f"""
    You are a smart assistant designed to categorize and rewrite questions. Your task contains 3 steps:

1. **Split and rewrite the input query into self-contained questions in English.**
   - Determine if the user's query contains multiple distinct questions, if so, separate them.
   - If the query is in any non-English language (Chinese), translate it to English first
   - Make each question standalone by:
     * Including complete context/subject in every question
     * Replacing pronouns (it, they, these) with specific subjects
     * Repeating full subject names in each question
   - Rewrite questions IN ENGLISH and incorporate relevant context from previous interactions
   - Clarify vague or unclear questions
   - Default to including the subject company from the QA history (previous conversations) as the subject when no specific subject is mentioned.
   - Output a string list containing all rewritten questions, even if there is only one.
   - Add the time information selectively to rewritten question:
     * Only add time information of latest available data (such as "in {year}") for questions related to financial metrics, sales and store data, market performance, or other time-sensitive business metrics.
     * Do NOT add time information for general questions about company attributes that are relatively stable, such as user profiles, business models, company history, or strategic positioning.

     The latest available data is based on year {year}.

     Examples (here <company> is a placeholder to the subject company in the query):
     "<公司>的季度营收是多少？" should be rewritten as "What is <company>'s quarterly revenue in {year}?".
     "<公司>的用户画像是什么?" should be rewritten as "What is the user profile for <company>?" (without adding year information)

2. **Identify the relevant date or any explicit or implied time reference based on the user's question and the conversation history.**
   - If no specific time is mentioned, use the current date as the default reference time.
   - Output the single value representing date in the format YYYY-MM-DD.
          
Here is the Q&A history:
{qa_history}

Question: {question}

Current time: {nowtime}

Respond in the following format:
Line 1: A JSON array of strings representing all sub-questions, each enclosed in double quotes and separated by commas. Example: ["question1", "question2", "question3"].
Line 2: The relevant date or time reference in YYYY-MM-DD format (e.g., 2022-01-01).
Line 3: "YES" - Always output "YES"

Please strictly adhere to this 3-line format with no additional text, explanations, or commentary.
    """

class QueryRewriter:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        """
        初始化查询重写器

        Args:
            base_url: LLM API的基础URL
            api_key: API密钥
            model_name: 模型名称
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

        # 初始化OpenAI客户端
        self.llm = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(f"QueryRewriter initialized with model: {model_name}")

    def rewrite_query(self, question: str, qa_history: str = "", max_retry: int = 3) -> Tuple[List[str], datetime, bool]:
        """
        重写查询，将复杂问题拆解为子查询

        Args:
            question: 用户的原始问题
            qa_history: 历史对话记录，默认为空
            max_retry: 最大重试次数

        Returns:
            Tuple[List[str], datetime, bool]: (重写后的问题列表, 查询时间, 是否需要RAG)
        """
        logger.info(f"Original question: {question}")

        for attempt in range(max_retry):
            try:
                completion = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": get_if_query_rag_prompt(question, qa_history)},
                        {"role": "user", "content": question}
                    ],
                    temperature=0,
                    top_p=0.8,
                    stream=False,
                )

                response_content = completion.choices[0].message.content.strip()
                response_lines = response_content.split("\n")

                # 验证响应格式
                if len(response_lines) < 3:
                    raise ValueError(f"Not enough lines in response, expected 3, got {len(response_lines)}")

                # 解析响应
                rewritten_questions = ast.literal_eval(response_lines[0].strip())
                query_time = datetime.strptime(response_lines[1].strip(), "%Y-%m-%d")
                need_rag = "yes" in response_lines[2].strip().lower()

                # 验证重写后的问题格式
                if not isinstance(rewritten_questions, list):
                    raise ValueError("Rewritten questions must be a list")

                logger.info(f"Rewritten questions: {rewritten_questions}")
                logger.info(f"Query time: {query_time}")
                logger.info(f"Need RAG: {need_rag}")

                return rewritten_questions, query_time, need_rag

            except Exception as e:
                logger.warning(f"Error in rewrite_query (attempt {attempt + 1}/{max_retry}): {str(e)}")

                if attempt == max_retry - 1:
                    # 最后一次重试失败，返回默认值
                    logger.error("All retry attempts failed, returning default values")
                    return [question], datetime.now(), False

                continue


def process_dataset(input_file: str, output_file: str, config: dict, question_field: str = "question"):
    """
    处理JSONL数据集，为每个问题添加重写结果

    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        config: QueryRewriter配置字典
        question_field: 问题字段名称，默认为"question"
    """
    # 加载数据集
    logger.info(f"Loading dataset from {input_file}")
    dataset = load_dataset('json', data_files=input_file, split='train')
    logger.info(f"Loaded {len(dataset)} examples")

    # 定义处理函数 - 每个进程内初始化rewriter以避免pickle错误
    def rewrite_example(example, idx):
        """处理单个样本"""
        # 在每个进程中初始化rewriter（避免pickle SSLContext）
        if not hasattr(rewrite_example, 'rewriter'):
            rewrite_example.rewriter = QueryRewriter(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model_name=config["model_name"]
            )
        
        try:
            question = example.get(question_field, "")
            qa_history = example.get("qa_history", "")
            
            if not question:
                logger.warning("Empty question found, skipping rewrite")
                example['rewritten'] = [question]
                example['query_time'] = datetime.now().strftime("%Y-%m-%d")
                example['need_rag'] = False
                return example
            
            # 执行重写
            rewritten_questions, query_time, need_rag = rewrite_example.rewriter.rewrite_query(
                question=question,
                qa_history=qa_history
            )
            
            # 添加结果到样本
            example['rewritten'] = rewritten_questions
            example['query_time'] = query_time.strftime("%Y-%m-%d")
            example['need_rag'] = need_rag
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            example['rewritten'] = [example.get(question_field, "")]
            example['query_time'] = datetime.now().strftime("%Y-%m-%d")
            example['need_rag'] = False
        
        return example

    # 使用map处理数据集（单进程避免pickle问题）
    logger.info("Processing dataset...")
    processed_dataset = dataset.map(rewrite_example, with_indices=True, num_proc=16)

    # 保存处理后的数据集
    logger.info(f"Saving processed dataset to {output_file}")
    processed_dataset.to_json(output_file, orient='records', lines=True, force_ascii=False)
    logger.info("Processing complete!")

    return processed_dataset


def main():
    """主函数"""
    
    # 配置参数 - 请根据实际情况修改
    CONFIG = {
        "base_url": "https://api.lkeap.cloud.tencent.com/v1",
        "api_key": "your_api_key_here",
        "model_name": "deepseek-v3"
    }

    # 文件路径 - 请根据实际情况修改
    input_file = ""
    output_file = ""
    
    # 处理数据集
    try:
        processed_dataset = process_dataset(
            input_file=input_file,
            output_file=output_file,
            config=CONFIG,
            question_field="question"  # 根据实际JSONL中的字段名修改
        )
        
        print(f"\n=== Processing Summary ===")
        print(f"Total examples processed: {len(processed_dataset)}")
        print(f"Output saved to: {output_file}")
        
        # 显示第一个示例
        if len(processed_dataset) > 0:
            print(f"\n=== First Example ===")
            first_example = processed_dataset[0]
            print(f"Original: {first_example.get('question', 'N/A')}")
            print(f"Rewritten: {first_example.get('rewritten', 'N/A')}")
            print(f"Query Time: {first_example.get('query_time', 'N/A')}")
            print(f"Need RAG: {first_example.get('need_rag', 'N/A')}")
            
    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found. Please create it first.")
        print(f"\nPlease create '{input_file}' with the following format:")
        print('{"question": "your question here"}')
        print('{"question": "another question", "qa_history": "previous context"}')
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
