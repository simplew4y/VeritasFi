import re
import sys
import os
import time
import json
import logging
logger = logging.getLogger(__name__)

import torch
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd
from FlagEmbedding import FlagLLMReranker

from .vllmManager import ChatManager
from .ragManager import RAGManager
from .profiler import profiler
from .frequentQA import QuestionSimilarityFinder
from .QARetriever import QAChromaLoader


def select_most_recent_time(time_info):

    time_info_as_dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in time_info]
    most_recent_date = max(time_info_as_dates)
    
    return most_recent_date.strftime("%Y-%m-%d")
    
@profiler.profile_function(name="rerank")
def get_rag_content(chat_manager: ChatManager, chunks, rewritten_question: str, query_time: datetime, retriever):
    # The rag context given to LLM, with the format as below:
    # "Date Published: 2024-01-02; Chunk Content: Cash Flow of Lotus Technology... 
    #  Date Published: 2024-01-04; Chunk Content: Lotus Technology recorded revenue of ..."
    rag_content = "" 
    time_info_list = []
    selected_chunks = []

    top_bundle_id = chat_manager.rank_chunk(chunks, rewritten_question, query_time, retriever)
    # logger.warning("rank chunks Max CUDA memory allocated: {}".format(torch.cuda.max_memory_allocated() / (1024 * 1024)))

    
    for bundle_id in top_bundle_id:
        bundle_chunks = [chunk for chunk in chunks if chunk['bundle_id'] == bundle_id]
        # Concatenate all chunks within the same bundle into one single paragraph;
        # if the overall length of the bundle paragraph is shorted than 50, ignore that bundle
        page_content = " ".join(chunk['page_content'] for chunk in bundle_chunks)
        if len(page_content) < 50:
            continue

        selected_chunks.extend(bundle_chunks)
    
    # Sort chunks by global_id to preserve their original sequence in the document
    # and maintain contextual flow
    selected_chunks = sorted(selected_chunks, key=lambda x: x['metadata']['global_id'])

    # Form rag_content, the overall context given to LLM, by concatenating selected chunks' content
    rag_content = "\n".join([f"Date Published: {chunk['metadata'].get('date_published', 'N/A')}; Chunk Content: {chunk['page_content']}" for chunk in selected_chunks])
    time_info_list = [chunk['metadata'].get('date_published', "N/A") for chunk in selected_chunks]
    # The rag info of each sub-question is added to the rag_info of the original question  
    sub_question_info = pd.DataFrame({
        'sub_query': [rewritten_question] * len(selected_chunks),
        'timeinfo': time_info_list,
        'chunk_id': [chunk['metadata']['doc_id'] for chunk in selected_chunks],
        'chunk_content': [chunk['page_content'] for chunk in selected_chunks],
        'chunk_bundle_id': [chunk['bundle_id'] for chunk in selected_chunks],
    })
    chat_manager.rag_info = pd.concat([chat_manager.rag_info, sub_question_info], ignore_index=True) 
    return rag_content, time_info_list



class ChatService:

    def __init__(self, config, rag_manager: RAGManager, rerank_topk: int, session_timeout: int = 1800):
        self.api_chat_manager: Dict[str, ChatManager] = {}
        self.rag_manager: RAGManager = rag_manager
        self.base_url: str = config.get('llm_base_url')
        self.model_name: str = config.get('llm_model_name')
        self.api_key: str = config.get('llm_api_key', 'EMPTY')
        self.rerank_topk = rerank_topk
        self.session_timeout = session_timeout
        # Lock to access the api_chat_manager dict
        self.api_chat_manager_lock = threading.Lock()

        # Lock to ensure only one chat_manager can call self.reranker.compute_score at the same time
        self.reranker_lock = threading.Lock()
        
        self.reranker = FlagLLMReranker(config.get('rerank_model'), devices='cuda',use_fp16 = True)
        self.frequent_qa_db = config.get('frequent_qa_directory')
        self.qa_table_directory = config.get('qa_table_directory')
        self.question_similarity_finder = QuestionSimilarityFinder(self.frequent_qa_db,self.qa_table_directory)
        self.qa_loader = QAChromaLoader(persist_directory = config.get("qa_table_persist_directory"), collection_name = "zeekr_qa")
        

        logger.warning("Load Reranker: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

        if not self.model_name or not self.base_url:
            logging.error("LLM model name/base_url is not configured.")
            sys.exit(1)
        logging.info(f"Using model: {self.model_name}, URL: {self.base_url}")

    def __del__(self):
        """
        Cleanup method that gets called when the ChatService object is being garbage collected.
        This ensures all ChatManager objects and their threads are properly cleaned up.
        """
        try:
            logger.info("ChatService cleanup started")
            session_ids = list(self.api_chat_manager.keys())
            # Clean up each ChatManager
            for session_id in session_ids:
                del self.api_chat_manager[session_id]
            logger.info("ChatService cleanup completed")
        except Exception as e:
            logger.error(f"Error during ChatService cleanup: {str(e)}")

    def cleanup_old_sessions(self):
        """Remove chat managers that are older than the timeout period"""
        current_time = datetime.now()
        expired_sessions = []
        
        # Convert session_timeout (in seconds) to timedelta
        timeout_delta = timedelta(seconds=self.session_timeout)
        
        with self.api_chat_manager_lock:
            for session_id, session_data in self.api_chat_manager.items():
                if current_time - session_data['timestamp'] > timeout_delta:
                    expired_sessions.append(session_id)
                    
            for session_id in expired_sessions:
                del self.api_chat_manager[session_id]
                logger.info(f"Removed expired session {session_id}")

    def get_or_create_chat_manager(self, session_id: str) -> ChatManager:
        with self.api_chat_manager_lock:
            if session_id not in self.api_chat_manager:
                chat_manager = ChatManager(session_id, self.base_url, self.api_key, self.model_name, 
                                        self.reranker, chunk_topk=self.rerank_topk, reranker_lock=self.reranker_lock)
                self.api_chat_manager[session_id] = {
                    'manager': chat_manager,
                    'timestamp': datetime.now()
                }
            else:
                # Update timestamp on access
                self.api_chat_manager[session_id]['timestamp'] = datetime.now()
            
        return self.api_chat_manager[session_id]['manager']

    def get_similar_questions_db(self, question: str, top_n: int = 5, threshold: float = 0.55, 
                              bm25_threshold: float = 3.0) -> tuple[bool, list[dict]]:
        try:
            # Find similar questions from BM25Matcher and from SequenceMatcher
            sequence_results_db = self.question_similarity_finder.find_similar_questions_db(
                question, top_n=2, threshold=threshold
            )
            
            bm25_results_db = self.question_similarity_finder.find_similar_questions_bm25_db(
                question, top_n=2, threshold=bm25_threshold
            )

            combined_results = []
            seen_ids = set() # remove duplicates from 2 matchers
            
            # BM25 results have the priority
            for match in bm25_results_db:
                row_id = match[0]
                seen_ids.add(row_id)
                logger.info(f"Found similar question via BM25 with score {match[4]}: '{match[1]}'")
                combined_results.append(match)     
                
            for match in sequence_results_db:
                row_id = match[0]
                if row_id not in seen_ids:
                    seen_ids.add(row_id)
                    logger.info(f"Found similar question via SequenceMatcher with score {match[4]}: '{match[1]}'")
                    combined_results.append(match)                  

            top_results = combined_results[:top_n]

            if top_results and len(top_results) > 0:
                qa_pairs_for_llm = []
                for match in top_results:
                    # logger.info(f"Found similar question in database with score {match[4]}: '{match[1]}'")
                    qa_pairs_for_llm.append(
                        {
                            "question": match[2],
                            "answer": match[3]
                        }
                    )
                if qa_pairs_for_llm:
                    return True, qa_pairs_for_llm
            return False, []
        except Exception as e:
            logger.error(f"Error searching for similar questions: {str(e)}")
            return False, []    

    def get_similar_questions_table(self, question: str, top_n: int = 5, threshold: float = 0.55, 
                            bm25_threshold: float = 3.0) -> tuple[bool, list[dict]]:
        try:
            # Find similar questions from BM25Matcher and from SequenceMatcher
            sequence_results_table = self.question_similarity_finder.find_similar_questions_table(
                question, top_n=3, threshold=threshold
            )
            
            bm25_results_table = self.question_similarity_finder.find_similar_questions_bm25_table(
                question, top_n=3, threshold=bm25_threshold
            )
            
            # print(sequence_results_table)

            combined_results = []
            seen_ids = set() # remove duplicates from 2 matchers
            
            # BM25 results have the priority
            for match in bm25_results_table:
                row_id = match[0]
                seen_ids.add(row_id)
                logger.info(f"Found similar question via BM25 with score {match[4]}: '{match[1]}'")
                combined_results.append(match)     
                
            for match in sequence_results_table:
                row_id = match[0]
                if row_id not in seen_ids:
                    seen_ids.add(row_id)
                    logger.info(f"Found similar question via SequenceMatcher with score {match[4]}: '{match[1]}'")
                    combined_results.append(match)                  

            top_results = combined_results[:top_n]

            if top_results and len(top_results) > 0:
                qa_pairs_for_llm = []
                for match in top_results:
                    # logger.info(f"Found similar question in database with score {match[4]}: '{match[1]}'")
                    qa_pairs_for_llm.append(
                        {
                            "question": match[2],
                            "answer": match[3]
                        }
                    )
                if qa_pairs_for_llm:
                    return True, qa_pairs_for_llm
            return False, []
        except Exception as e:
            # logger.error(f"Error searching for similar questions: {str(e)}")
            logger.error("Error searching for similar questions", exc_info=True)
            return False, []        

    
    def generate_response_with_rag(self, question: str, session_id: str, internal_input=None, interrupt_index=None):
        chat_manager = self.get_or_create_chat_manager(session_id)
        lang = '中文' if bool(re.search(r'[\u4e00-\u9fff]', question)) else 'English'
        user_input = question
        qa_history = chat_manager.get_qa_history()
        rewrite_start_time = time.perf_counter()
        rewritten = chat_manager.if_query_rag(user_input, qa_history)
        rewrite_end_time = time.perf_counter()
        logger.info("The time for rewrite: {:.2f}".format(rewrite_end_time-rewrite_start_time))

        # Clear the rag_info of the last question before processing the incoming question
        chat_manager.reset_rag_info()
        answer = ""
        answers = []
        all_retrieved_content = []
        hypo_chunks_list = []

        for rewritten_question in rewritten:
            logger.info(f"Processing sub-question: {rewritten_question}")
            user_input = rewritten_question
            rag_context = ""
            used_time = None

            if chat_manager.need_rag:
                # log_gpu_usage('rag started')
                timeinfo_list = []
                
                for retriever in self.rag_manager._retrievers:
                    hypo_chunks = chat_manager.generate_hypo_chunks(rewritten_question)
                    # hypo_chunks = []
                    hypo_chunks_list.append(hypo_chunks)

                    retriever_content = retriever.invoke(user_input, hypo_chunks)
                    all_retrieved_content.append(retriever_content)
                    rerank_start_time = time.perf_counter()
                    current_context, timeinfo_list = get_rag_content(chat_manager, retriever_content, rewritten_question, chat_manager.query_time, retriever)
                    rerank_end_time = time.perf_counter()
                    logger.info(f"Reranking sub-question {rewritten_question}")
                    logger.info("The time for rerank: {:.2f}".format(rerank_end_time-rerank_start_time))
                    # log_gpu_usage('rag finished')
                    rag_context += current_context + '\n'
                    logger.info(f'Input Rag Context is: \n {rag_context}')

                # time of chunks in metadata['date_published']
                used_time = select_most_recent_time(timeinfo_list)
            response_start_time = time.perf_counter()
            response = chat_manager.chat_internal(user_input, rag_context, used_time, lang, False, 
                                                internal_input=internal_input, 
                                                interrupt_index=interrupt_index)
            response_end_time = time.perf_counter()
            logger.info("The time for response: {:.2f}".format(response_end_time - response_start_time))

            answer = response.choices[0].message.content
            answers.append(answer)

            logger.info(f"Rewritten Sub-Question: {rewritten_question}")
            logger.info(f"Sub-Question Answer: {answer}")

            # chat_manager.save_chat_history(answer)
            # history = chat_manager.get_hisotry_summary()
            # history = "Using Deep seek chat history instead"

        if len(rewritten) > 1:
            modify_start_time = time.perf_counter()
            logger.info("Start to merge the answer")
            answer = chat_manager.modify_answer(answers, question, rewritten, stream=False, lang=lang)
            modify_end_time = time.perf_counter()
            logger.info("The time for modify: {:.2f}".format(modify_end_time-modify_start_time))
            logger.info(f"Final answer: {answer}")
        else:
            answer = answers[0]
        
        # Update chat logs
        chat_manager.add_to_qa_history(user_input, answer)
        chat_manager.all_retrieved_content = all_retrieved_content
        chat_manager.hypo_chunks = hypo_chunks_list
        logger.info(f"Current QA History after processing the question '{user_input}': \n {chat_manager.qa_history}")
        
        # Start a thread to generate chat summary
        #summary_thread = threading.Thread(
        #    target=self.generate_chat_summary,
        #    args=(session_id,)
        #)
        #summary_thread.daemon = True  # do not block application shutdown
        #summary_thread.start()
        
        return answer, rag_context, chat_manager.rag_info, rewritten, chat_manager.hypo_chunks, all_retrieved_content, chat_manager.get_qa_history()

    def generate_response_async(self, question: str, session_id: str, internal_input: str = None, interrupt_index: int = None):
        chat_manager = self.get_or_create_chat_manager(session_id)
        lang = '中文' if bool(re.search(r'[\u4e00-\u9fff]', question)) else 'English'
        qa_history = chat_manager.get_qa_history()
        rewrite_start_time = time.perf_counter()
        rewrittens = chat_manager.if_query_rag(question, qa_history)
        rewrite_end_time = time.perf_counter()
        logger.info("The time for rewrite: {:.2f}".format(rewrite_end_time-rewrite_start_time))
        


        chat_manager.reset_rag_info()
        answer = ""
        all_retrieved_content = []
        chat_tasks = []
        hypo_chunks_list = []

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # hyde_tasks = []
            # profiler.start("hyde_async")
            # for rewritten in rewrittens:
            #     if chat_manager.need_rag:
            #         for retriever in self.rag_manager._retrievers:
            #             task = loop.create_task(chat_manager.generate_hypo_chunks_async(rewritten))
            #             hyde_tasks.append(task)
            # hyde_resps = loop.run_until_complete(asyncio.gather(*hyde_tasks))
            # profiler.end("hyde_async")

            chat_tasks = []
            hyde_resps = [[] for _ in range(len(rewrittens))]
            for rewritten, hypo_chunks in zip(rewrittens, hyde_resps):
                logger.info(f"Searching for similar question: '{rewritten}'")
                found_matches, qa_pairs_db = self.get_similar_questions_db(rewritten)
                found_matches, qa_pairs_table = self.get_similar_questions_table(rewritten)

                # qa_pairs_table = []
                seen_questions = {row["question"] for row in qa_pairs_table}
                results = self.qa_loader.query_qa(rewritten, n_results=2)
                for result in results:
                    q = result["question_rewritten"]
                    if q in seen_questions:
                        continue

                    qa_pairs_table.append(
                        {
                            "question": q,
                            "answer": result['data']
                        }
                    )
                    seen_questions.add(q)
    
                qa_pairs_for_llm = qa_pairs_db + qa_pairs_table

                

                logger.info(f"Found similar questions in database: {qa_pairs_for_llm}")

                if chat_manager.need_rag:
                    hypo_chunks_list.append(hypo_chunks)

                    retriever = self.rag_manager._retrievers[0]

                    retriever_content = retriever.invoke(rewritten, hypo_chunks)
                    all_retrieved_content.append(retriever_content)

                    rag_context, timeinfo_list = get_rag_content(chat_manager, retriever_content, rewritten, chat_manager.query_time, retriever)
                    rag_docu_time = select_most_recent_time(timeinfo_list)
                else:
                    rag_context = ""
                    rag_docu_time = ""

                task = loop.create_task(chat_manager.chat_async(rewritten, rag_context, rag_docu_time, lang, qa_pairs_for_llm))
                chat_tasks.append(task)

            profiler.start("answer_sub")
            chat_resps = loop.run_until_complete(asyncio.gather(*chat_tasks))
            profiler.end("answer_sub")
        except Exception as e:
            logger.error(f"Error during task execution: {str(e)}")
            raise
        
        # If there are multiple subquestions, collect information for final answer
        if len(rewrittens) > 1:
            chat_answers = []
            chat_questions = []
            for rewritten, response in chat_resps:
                answer = response.choices[0].message.content
                chat_answers.append(answer)
                chat_questions.append(rewritten)

            merge_answer = chat_manager.modify_answer(chat_answers, question, chat_questions, stream=False, lang=lang)
            final_answer = merge_answer
        else:
            final_answer = chat_resps[0][1].choices[0].message.content
        # logger.info(f"Final answer: {final_answer}")
        chat_manager.add_to_qa_history(question, final_answer)
        chat_manager.all_retrieved_content = all_retrieved_content
        chat_manager.hypo_chunks = hypo_chunks_list

        qa_history = chat_manager.qa_history

        # Start a thread to generate chat summary
        # summary_thread = threading.Thread(
        #     target=self.generate_chat_summary,
        #     args=(session_id,)
        # )
        # summary_thread.daemon = True  # do not block application shutdown
        # summary_thread.start()
        
        return final_answer, "", chat_manager.rag_info, rewrittens, chat_manager.hypo_chunks, all_retrieved_content, qa_history

    def generate_response_async_stream(self, question: str, session_id: str, internal_input: str = None, interrupt_index: int = None):
        profiler.start("answer_stream")
        
        chat_manager = self.get_or_create_chat_manager(session_id)
        lang = '中文' if bool(re.search(r'[\u4e00-\u9fff]', question)) else 'English'
        qa_history = chat_manager.get_qa_history()
        rewrittens = chat_manager.if_query_rag(question, qa_history)

        chat_manager.reset_rag_info()
        answer = ""
        all_retrieved_content = []
        hypo_chunks_list = []
            

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # hyde_tasks = []
            # profiler.start("hyde_async")
            # for rewritten in rewrittens:
            #     if chat_manager.need_rag:
            #         for retriever in self.rag_manager._retrievers:
            #             task = loop.create_task(chat_manager.generate_hypo_chunks_async(rewritten))
            #             hyde_tasks.append(task)
            # hyde_resps = loop.run_until_complete(asyncio.gather(*hyde_tasks))
            # profiler.end("hyde_async")

            chat_tasks = []
            hyde_resps = [[] for _ in range(len(rewrittens))]
            for rewritten, hypo_chunks in zip(rewrittens, hyde_resps):
                logger.info(f"Searching for similar question: '{rewritten}'")
                found_matches, qa_pairs_db = self.get_similar_questions_db(rewritten)
                found_matches, qa_pairs_table = self.get_similar_questions_table(rewritten)

                # qa_pairs_table = []
                seen_questions = {row["question"] for row in qa_pairs_table}
                results = self.qa_loader.query_qa(rewritten, n_results=2)
                for result in results:
                    q = result["question_rewritten"]
                    if q in seen_questions:
                        continue

                    qa_pairs_table.append(
                        {
                            "question": q,
                            "answer": result['data']
                        }
                    )
                    seen_questions.add(q)

                
                qa_pairs_for_llm = qa_pairs_db + qa_pairs_table

                logger.info(f"Found similar questions in database: {qa_pairs_for_llm}")

                hypo_chunks_list.append(hypo_chunks)

                rag_context = ""
                rag_docu_time = None

                if chat_manager.need_rag:
                    retriever = self.rag_manager._retrievers[0]
                    retriever_content = retriever.invoke(rewritten, hypo_chunks)
                    all_retrieved_content.append(retriever_content)

                    rag_context, timeinfos = get_rag_content(chat_manager, retriever_content, rewritten, chat_manager.query_time, retriever)
                    rag_docu_time = select_most_recent_time(timeinfos)

                if len(rewrittens) == 1:
                    answer = chat_manager.chat_internal(rewritten, rag_context, rag_docu_time, lang, qa_pairs_for_llm, True)
                else:
                    task = loop.create_task(chat_manager.chat_async(rewritten, rag_context, rag_docu_time, lang, qa_pairs_for_llm))
                    chat_tasks.append(task)

            if len(rewrittens) > 1:
                profiler.start("answer_sub")
                chat_resps = loop.run_until_complete(asyncio.gather(*chat_tasks))
                profiler.end("answer_sub")

                chat_answers = []
                chat_questions = []
                for rewritten, response in chat_resps:
                    answer = response.choices[0].message.content
                    chat_answers.append(answer)
                    chat_questions.append(rewritten)

                answer = chat_manager.modify_answer(chat_answers, question, chat_questions, stream=True, lang=lang)
        except Exception as e:
            logger.error(f"Error during task execution: {str(e)}")
            raise
        finally:
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending_tasks:
                loop.run_until_complete(asyncio.gather(*[task.cancel() for task in pending_tasks]))
            loop.close()

        final_answer = ""
        try:
            first_flag = True
            for chunk in answer:
                if first_flag:
                    first_flag = False
                    profiler.end("answer_stream")
                    # profiler.log_profiling_results('tmp.json')
                if chunk.choices[0].delta.content:
                    final_answer += chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'response': chunk.choices[0].delta.content})}\n\n"
        except Exception as e:
            logger.error(f"Error during stream response: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
        chat_manager.add_to_qa_history(question, final_answer)
        chat_manager.all_retrieved_content = all_retrieved_content
        chat_manager.hypo_chunks = hypo_chunks_list

        # Start a thread to generate chat summary
        # summary_thread = threading.Thread(
        #     target=self.generate_chat_summary,
        #     args=(session_id,)
        # )
        # summary_thread.daemon = True  # do not block application shutdown
        # summary_thread.start()
        
    def generate_chat_summary(self, session_id: str):
        '''
            Generate chat summary for a ChatManager given session_id
        '''
        chat_manager = self.get_or_create_chat_manager(session_id)
        try:
            # Set the summarizing flag to true and clear the event
            with chat_manager.summary_lock:
                chat_manager.is_summarizing = True
                chat_manager.summary_event.clear()
            
            qa_history = chat_manager.get_qa_history()
            new_summary = chat_manager.summarize_chat_history(qa_history)
            
            # Update the summary and signal completion
            with chat_manager.summary_lock:
                chat_manager.history_summary = new_summary
                chat_manager.is_summarizing = False
                chat_manager.summary_event.set()

            logger.info(f"Chat summary generated: {new_summary}")

        except Exception as e:
            # Make sure to reset flags even on error
            with chat_manager.summary_lock:
                chat_manager.is_summarizing = False
                chat_manager.summary_event.set()
            logging.error(f"An error occurred while generating summary: {str(e)}")

    def get_test_info(self, session_id: str):
        # use this function only in the testing scripts [qa_e2e_*.py], otherwise it will slow down the process
        chat_manager = self.get_or_create_chat_manager(session_id)
        if chat_manager.is_summarizing:
            logger.info(f"Waiting for summary generation to complete for session {session_id}, should ONLY be invoked from testing scripts")
            chat_manager.summary_event.wait(timeout=10)

        return  None, chat_manager.need_rag
