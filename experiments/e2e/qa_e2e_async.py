import sys
import os
import time
import yaml
import logging
import json
logging.basicConfig(
    filemode='w',
    filename='qa_e2e.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.vllmChatService import ChatService
from utils.ragManager import RAGManager

def read_questions_from_md(md_file_path):
    questions = []
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                question = line.strip()
                if question:
                    questions.append(question)
    except FileNotFoundError:
        print(f"'{md_file_path}' not found")
    except Exception as e:
        print(f"Error：{e}")
    return questions

def load_questions_file(file_path):
    """Load questions from either JSON or text files"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension in ['.txt', '.md']:
            # Load text file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                # Convert to same format as JSON data
                data = [{'question': line, 'answer': ''} for line in lines]
        elif file_extension == '.json':
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        raise

if __name__ == "__main__":
    
    config_path = os.getenv('CONFIG_PATH', '../../config/production.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    logger = logging.getLogger(__name__)

    #collections = {'lotus': 10}
    collections = {'zeekr': 10}
    logger.warning("Before loading: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
    rag_manager = RAGManager(config=config, collections=collections)
    logger.warning("Load retriever: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
    chat_service = ChatService(config=config, rag_manager=rag_manager, rerank_topk=5)
    logger.warning("Load Reranker: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    questions_folder_path = "./test_questions/zeekr_questions/"
    questions_file = "question_batch2.md"
    
    
    questions_file_path = questions_folder_path + questions_file
    
    # Create output directory based on markdown file name
    base_name = os.path.splitext(os.path.basename(questions_file_path))[0]
    DIR_PATH = os.path.join(os.path.dirname(questions_file_path), f'{base_name}')

    BATCH_SIZE = 1
    show_rag_info = True
    show_history_summary = True
    show_rewritten_question = True
    show_if_rag = False
    show_input = False
    show_total_input = False
    judge_answer = False

    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    try:
        data = load_questions_file(questions_file_path)
        num_questions = len(data)
        sum_score = 0
    except Exception as e:
        logging.error(f"Failed to load questions: {str(e)}")
        sys.exit(1)
    
    for i in range(0, num_questions, BATCH_SIZE):
        batch_questions = data[i:i+BATCH_SIZE]
        
        # Create a dictionary to store results
        results = {
            "metadata": {
                "generated_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "database": config["persist_directory"],
                "batch_index": f"{i}_{i+BATCH_SIZE-1}"
            },
            "questions": []
        }
        
        session_id = time.time()
        for idx, item in enumerate(batch_questions):
            question = item['question']
            expected_answer = item['answer']
            
            # Process question
            st_time = time.time()
            answer, rag_context, rag_info, rewritten_question, hypo_chunk_content, all_retrieved_content, history = chat_service.generate_response_async(
                question, session_id, internal_input=None, interrupt_index=None
            )
            duration = time.time() - st_time
            
            complete_input, need_rag = chat_service.get_test_info(session_id)
            
            # Create question result dictionary
            question_result = {
                "question_index": idx,
                "original_question": question,
                "answer": answer,
                "duration": duration
            }
            
            question_result["rewritten_question"] = rewritten_question
            question_result["need_rag"] = need_rag
            # question_result["history_summary"] = history_summary
            question_result["rag_info"] = rag_info.to_dict('records')
            question_result["all_retrieved_content"] = all_retrieved_content
            # question_result["complete_input"] = complete_input
                
            if judge_answer:
                judge_score, reason = chat_service.api_chat_manager[session_id]['manager'].evaluate(
                    answer, expected_answer
                )
                sum_score += judge_score
                question_result.update({
                    "expected_answer": expected_answer,
                    "score": judge_score,
                    "evaluation_reason": reason
                })
            
            results["questions"].append(question_result)
            
        chat_service.api_chat_manager[session_id]['manager'].clear_chat_history()
        
        # Save results to JSON file
        output_file = os.path.join(DIR_PATH, f'question_{i}_{i+BATCH_SIZE-1}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if judge_answer:
        accuracy = sum_score / num_questions
        print(f'Average Score: {accuracy:.2f}')
