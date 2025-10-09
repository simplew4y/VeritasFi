from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import ast
import json
import os
import openai
import torch
import threading
import pandas as pd
from typing import List, Dict, Tuple
from pydantic import BaseModel
import prompts
from .profiler import profiler
import asyncio
from .tools import get_stock_price, get_ipo_info

class IfQueryRagResp(BaseModel):
    need_rag: bool
    mult_question: bool
    rewritten: List[str]

class ChatManager:
    def __init__(self, session_id, base_url, api_key, model_name, reranker, chunk_topk = 5, history_limit=20, reranker_lock=None):
        assert history_limit % 2 == 0, "history_limit must be an even number"
        self.session_id = session_id
        self.base_url = base_url
        self.model_name = model_name
        self.llm = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.async_llm = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.reranker = reranker
        
        # Store the lock for synchronizing reranker access
        self.reranker_lock = reranker_lock
        self.similar_threshhold = 0.9
        self.internal_assistant_message = []

        self.chat_history = []
        self.all_chat_history = [{
            "role": "system", "content": prompts.get_sys_prompt()
        }]

        # Intermediate variables, logged for debugging
        self.need_rag = False
        self.rewrittens = []
        self.query_time = datetime.now()
        self.hypo_chunks = []
        self.rag_info = pd.DataFrame({
            'timeinfo': [],
            'chunk_id': [],
            'chunk_content': [],
            'chunk_bundle_id': []
        })
        self.all_retrieved_content = []

        #This is the history of user and the rag llm
        self.qa_history = []
        self.time_info = ""
        self.history_limit = history_limit
        self.chunk_topk = chunk_topk

        # Chat summary
        self.summary_lock = threading.Lock()
        self.is_summarizing = False
        self.summary_event = threading.Event()
        self.history_summary = ""

        # Set logger formatter with [ChatManager] prefix
        formatter = logging.Formatter('[ChatManager] %(asctime)s %(levelname)s %(message)s')
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False

        # Load available tools
        tools_schema_path = os.path.join(os.path.dirname(__file__), "tools_schema.json")
        try:
            with open(tools_schema_path, "r", encoding="utf-8") as f:
                self.tools_schema = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load tools_schema.json: {e}")
            self.tools_schema = []


    def __del__(self):
        """
        Destructor that gets called when the ChatManager object is being garbage collected.
        This handles cleaning up any threads or resources that need proper disposal.
        """
        try:
            logger.info(f"ChatManager cleanup started for session {self.session_id}")
            
            # Signal any running summary threads to exit
            with self.summary_lock:
                if self.is_summarizing:
                    self.summary_event.set()
                    logger.info(f"Signaled summary thread to exit for session {self.session_id}")
                    # Wait a short time for the thread to finish
                    self.summary_event.wait(timeout=1.0)
            
            # Clear any large data structures to help with garbage collection
            self.chat_history.clear()
            self.all_chat_history.clear()
            self.qa_history.clear()
            self.rag_info = pd.DataFrame()
            
            logger.info(f"ChatManager cleanup completed for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error during ChatManager cleanup for session {self.session_id}: {str(e)}")

    def get_history_summary(self):
        # Wait for summary generation to complete if it's running
        if self.is_summarizing:
            self.summary_event.wait()
        return self.history_summary
        
    @profiler.profile_function(name="rewrite")
    def if_query_rag(self, question, qa_history, max_retry=3):
        logger.info(f"Original question: {question}")
        for _ in range(max_retry):
            try:
                completion = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompts.get_if_query_rag_prompt(qa_history=qa_history, question=question)},
                        {"role": "user", "content": question}
                    ],
                    temperature=0,
                    top_p=0.8,
                    stream=False,
                    # extra_body={"guided_json": IfQueryRagResp.model_json_schema()}
                )

                try:
                    profiler.add_metric("rewrite_total_tokens", completion.usage.total_tokens)
                except:
                    pass

                response_lines = completion.choices[0].message.content.strip().split("\n")
                assert len(response_lines) == 3, "Not enough lines in response, expected 3, get %d" % len(response_lines)

                self.rewrittens = ast.literal_eval(response_lines[0].strip())
                self.query_time = datetime.strptime(response_lines[1].strip(), "%Y-%m-%d")
                self.need_rag = "yes" in response_lines[2].strip().lower()

                assert isinstance(self.rewrittens, list), "Rewritten question must be a list"
                break

                # data = IfQueryRagResp.model_validate_json(completion.choices[0].message.content)
                # print(data)
                # self.need_rag = data.need_rag
                # self.mult_question = data.mult_question
                # rewritten = data.rewritten
                # break
                
            except Exception as e:
                logger.warning(f"Error in if_query_rag: {str(e)}. Retry...")
                self.need_rag = False
                self.rewrittens = [question]
                self.query_time = datetime.now().strftime("%Y-%m-%d")
                continue
            
        logger.info(f"Rewritten question: {self.rewrittens}")
        logger.info(f"Query Time: {self.query_time}")
        logger.info(f"Need RAG: {self.need_rag}")
        return self.rewrittens
    
    def if_query_rag_financebench(self, question, qa_history, max_retry=1):
        prompt_template = """
        You are a smart assistant designed to categorize and rewrite questions. Your task contains 2 steps:

        1. **Determine if the user's question includes more than one subquestions**:
            - If the input contains more than one distinct question, respond with True.
            Split user's question into individual sub-questions and do step 2 for each individual sub-questions.
            The output for step 2 in this case should be a string list that contains all sub-questions after rewriting.

            - If the input contains only one question, respond with False.
            Go to step 2 and rewrite the question.
            The output for step 2 in this case should just be a string list with 1 element (the rewritten question).

        2. **Rewrite the question according to the Q&A history**:
            - Rewritten question is in English.
            - If related to previous interactions, it should be rewritten to incorporate the relevant context.

        Here is the Q&A history:
        {qa_history}

        Question: {question}

        Respond in the following json format:
        {{
            "mult_question": True or False (if the input includes multiple questions),
            "rewritten": List[str] (rewritten questions)
        }}
        """
        
        logger.info(f"Original question: {question}")
        for _ in range(max_retry):
            try:
                completion = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt_template.format(qa_history=qa_history, question=question)},
                        {"role": "user", "content": question}
                    ],
                    temperature=0,
                    top_p=0.8,
                    stream=False,
                    extra_body={"guided_json": IfQueryRagResp.model_json_schema()}
                )
                data = IfQueryRagResp.model_validate_json(completion.choices[0].message.content)
                self.need_rag = True
                self.rewrittens = data.rewritten
                break
                
            except Exception as e:
                logger.warning(f"Error in if_query_rag: {str(e)}")
                continue
            
        logger.info(f"Rewritten question: {self.rewrittens}")
        return self.rewrittens

    @profiler.profile_function(name="hyde")
    def generate_hypo_chunks(self, question: str, max_retry=3) -> List[str]:
        chunk_list = []
        for retry in range(max_retry):
            try:
                completion = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompts.get_hypo_sys_prompt(num_hypo=3)},
                        {"role": "user", "content": question}
                    ],
                    temperature=0,
                    top_p=0.8,
                    stream=False,
                )
                try:
                    profiler.add_metric("hyde_tokens", completion.usage.total_tokens)
                except:
                    pass
                hypothetical_context = completion.choices[0].message.content
                chunk_list = [chunk.strip() for chunk in hypothetical_context.split("ANSWER:")[1:]]
                break
            except Exception as e:
                logger.warning(f"Error while generating hypothetical chunks: {e}")
        return chunk_list

    async def generate_hypo_chunks_async(self, question: str, max_retry=3):
        chunk_list = []
        for attempt in range(max_retry):
            try:
                completion = await asyncio.wait_for(
                    self.async_llm.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": prompts.get_hypo_sys_prompt(num_hypo=3)},
                            {"role": "user", "content": question}
                        ],
                        temperature=0,
                        top_p=0.8,
                        stream=False,
                    ),
                    timeout=30  # 30-second timeout
                )
                
                try:
                    profiler.add_metric("hyde_total_tokens", completion.usage.total_tokens)
                except:
                    pass
                hypothetical_context = completion.choices[0].message.content
                chunk_list = [chunk.strip() for chunk in hypothetical_context.split("ANSWER:")[1:]]
                break
            except asyncio.TimeoutError:
                logger.warning(f"Request timed out (attempt {attempt+1}/{max_retry})")
                await asyncio.sleep(0.5)  # Brief pause before retry
            except Exception as e:
                logger.warning(f"Error while generating hypothetical chunks: {e}")
                await asyncio.sleep(0.5)  # Brief pause before retry
                
        return chunk_list
    
    def summarize_chat_history(self, chat_history, max_retry=3):
        summary = ""
        for i in range(max_retry):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompts.get_summary_prompt(chat_history)},
                    {"role": "user", "content": "Summarize the conversation history."}
                ],
                temperature=0,
                top_p=0.8,
                stream=False,
            )
            summary = completion.choices[0].message.content.strip()
            if summary:
                break

        return summary

    @profiler.profile_function(name="final_answer")
    def modify_answer(self, answers: List[str], question: str, rewrittens: List[str], stream: bool, lang: str, max_retry=3):
        # Create pairs of rewritten questions and their corresponding answers
        qa_pairs = []
        for i in range(len(rewrittens)):
            if i < len(answers):
                qa_pairs.append(f"Question: {rewrittens[i]}\nAnswer: {answers[i]}")
        
        qa_pairs_text = "\n\n".join(qa_pairs)

        for i in range(max_retry):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompts.get_modify_answer_prompt(question, qa_pairs_text, lang)},
                    {"role": "user", "content": "Create a comprehensive answer based on the provided information."}
                ],
                temperature=0,
                top_p=0.8,
                stream=stream
            )
            
            if not stream:
                merged_answer = completion.choices[0].message.content.strip()
                try:
                    profiler.add_metric("final_answer_tokens", completion.usage.total_tokens)
                except:
                    pass
                if merged_answer:
                    logger.info(f"Sub-Questions: {answers}")
                    logger.info(f"Sub-Question Answers: {answers}")
                    logger.info(f"Merged Answers: {merged_answer}")
                    return merged_answer
                logger.warning(f"Empty response for modifying answer")
            else:
                return completion

        # If we reach here in non-streaming mode, return the first answer or empty string
        return answers[0] if answers else ""
    
    def evaluate(self, answer, expected_answer) -> Tuple[float, str]:
        # Use LLM to evaluate the answer, return the score [0-1], 1 means the answer totally match the expected answer and include all the information. 0 means totally different.
        prompt_template = f"""
        You are a smart assistant designed to evaluate answers provided. Your task is to compare the given answer with the expected answer and assign a score ranging from 0 to 1 based on its relevance and accuracy. The evaluation must consider whether the given answer includes all the numbers and points in the expected answer.
        • A score of 1 indicates that the given answer includes all the numbers and points in the expected answer.
        • A score of 0 indicates that the given answer is irrelevant, inaccurate, or does not include any of the key information from the expected answer.
        • Scores between 0 and 1 reflect partial relevance and accuracy, based on how much of the expected answer’s information is included.

        In addition to the score, provide a brief explanation of the reasoning behind the assigned score.

        Output your response in the following format:
        
        Score: [score]
        Reason: [brief explanation, focusing on whether the given answer includes all or part of the expected answer and the overall relevance and accuracy.]
        """

        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": f"Answer: {answer}\nExpected Answer: {expected_answer}"}
            ],
            temperature=0,
            top_p=0.8,
            stream=False
        )
        resp = completion.choices[0].message.content.strip()
        score = float(resp.split("Score:")[1].split("Reason:")[0].strip())
        reason = resp.split("Reason:")[1].strip()
        return score, reason

    def evaluate_chunk(self, chunk: str, question: str, exp_answer: str) -> bool:
        # Use LLM to evaluate the chunk, return True if the chunk is inclusive to get the expected answer for the question, otherwise return False
        prompt_template = """
        You are a smart assistant whose task is to determine whether the provided chunks of text are relevant for answering the 'Question', and whether they contain one OR more key information necessary to produce the 'Expected Answer'

        Criteria:
        1. Consider the overall context and how chunks may complement each other to form a complete answer.
        2. A chunk should be marked as relevant if it:
        - Contains direct information needed for the answer
        - Answers part (aspect) of the question
        3. For questions requiring multiple aspects, mark chunks as relevant if they address any of:
        - Financial metrics 
        - Strategic planning 
        - Business positioning 
        - Operational aspects 
        - Future outlook 
        - Historical context 
        - Industry relationships 

        Response format:
        Relevance: [YES or NO]
        Reason: [One sentence explains why this chunk contributes to the answer or why it doesn't]
        """

        resp = ""
        try_cnt = 3
        while try_cnt > 0 and (resp != "YES" and resp != "NO"):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template},
                    {
                        "role": "user", 
                        "content": f"Question: {question}\nExpected Answer: {exp_answer}\nChunk: {chunk}"
                    }
                ],
                temperature=0,
                top_p=0.8,
                stream=False
            )
            resp = completion.choices[0].message.content.strip()
            flag = resp.split("Relevance:")[1].split("Reason:")[0].strip()
            reason = resp.split("Reason:")[1].strip()
            try_cnt -= 1

        print(f"Question: {question}\nExpected Answer: {exp_answer}\nChunk: {chunk}\nResponse: {flag}\nReason: {reason}")
        return flag == "YES"

    def rank_chunk(self, chunks: List[Dict], question: str, query_time: datetime, retriever):
        
        bundle_map = {}
        for idx, chunk in enumerate(chunks):
            bundle_map.setdefault(chunk['bundle_id'], []).append(idx)

        pairs = [[question, chunk['page_content']] for chunk in chunks]
        time_scores = []

        #只有chunk content的 list
        chunk_content_list = []
        chunk_content_list.extend(chunk['page_content'] for chunk in chunks)

        for chunk in chunks:
            # time score = max(0, 1 - |query reference date - date of chunk| / 365)
            score = abs((query_time - datetime.strptime(chunk['metadata']['date_published'], "%Y-%m-%d")).days)
            score = max(0, 1 - score / 365)
            time_scores.append(score)

        # Use the reranker_lock if available to ensure only one compute_score call at a time
        with self.reranker_lock, torch.no_grad():
            reranker_scores = self.reranker.compute_score(pairs, batch_size=8)
            reranker_scores = torch.tensor(reranker_scores)
        
        time_scores = torch.tensor(time_scores)
        scores = reranker_scores + time_scores

        ranked_indices = torch.argsort(scores, descending=True).tolist()

        # 根据 chunks_num 选择合适数量的 chunk，确保总大小不超过 topk
        selected_indices = []
        current_size = 0
        similar_mtx = retriever.compute_similarity_mtx(chunk_content_list)

        for idx in ranked_indices:
            bundle_id = chunks[idx]['bundle_id']
            bundle = bundle_map[bundle_id]
            # if bunleid is selected, skip
            if bundle_id in selected_indices or current_size + len(bundle) > self.chunk_topk:
                continue

            # remove similar chunks
            # similarity = retriever.compute_similarity(chunk_content_list, selected_indices, idx)
            # if torch.any(similarity > self.similar_threshhold):
            #     print(f"chunk{idx} is skip due to similarity")
            #     continue
            if torch.any(similar_mtx[idx, selected_indices] > self.similar_threshhold):
                # logger.info(f"chunk{idx} is skip due to similarity")
                continue

            selected_indices.append(bundle_id)
            current_size += len(bundle)
            
        return selected_indices[::-1]

    # sync streaming/ non-streaming chat
    def chat_internal(self, user_input, rag_context='', rag_docu_time=None, lang: str='en', potential_qa=[{}], stream=False, internal_input=None, interrupt_index=None):
        # Handle modification of the previous assistant message if the interrupt index is provided
        if interrupt_index is not None:
            self.modify_previous_assistant_message(interrupt_index)

        # Prepend the internal assistant input to the user message if provided
        if internal_input:
            # Prepend the internal input to the user's message
            user_input = f"[Internal Assistant Information]: {internal_input}\n\nUser Input: {user_input}"

        # Now create the user message and append it to the chat history
        user_message = {"role": "user", "content": prompts.get_qa_template(user_input, rag_context, lang, self.get_internal_assitant_message(), potential_qa)}
        time_info = f"\nAt the end of your response, include only one sentence stating that the information is based on knowledge available before {rag_docu_time}, and ensure that the language used remains consistent with previous responses." if rag_docu_time else ""

        # print(user_message)

        self.chat_history.append(user_message)
        self.all_chat_history.append(user_message)

        messages = [{
            "role": "system", "content": prompts.get_sys_prompt() + time_info
        }]
        messages.extend(self.form_chat_history())
        messages.append(user_message)

        response = self.llm.chat.completions.create(
            messages=messages,
            model=self.model_name,
            stream=stream,
            temperature=0,
            top_p=0.8,
        )

        return response
    
    async def process_tool_calls(self, messages, tools_schema):
        """
        Process tool calls in the chat messages.
        """

        # STEP 1: Send the user message and tools to the model
        response = await self.async_llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message) # Append the tool call response to history
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {
                "get_stock_price": get_stock_price,
                "get_ipo_info": get_ipo_info
            }

            # STEP 2: Execute the function and get its output
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                tool_output = function_to_call(**function_args)
                
                # STEP 3: Append the tool output to the message history
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(tool_output),
                    }
                )

        # logger.info(f"Messages after tool execution: {messages}")
        return messages

    # async non-streaming chat 
    async def chat_async(self, user_input, rag_context='', rag_docu_time=None, lang: str='en', potential_qa=[{}]):
        try:
            # user_message = {"role": "user", "content": prompts.get_qa_template(user_input, rag_context, lang, self.get_internal_assitant_message(), potential_qa)}
            # time_info = f"\nAt the end of your response, include only one sentence stating that the information is based on knowledge available before {rag_docu_time}, and ensure that the language used remains consistent with previous responses." if rag_docu_time else ""
            
            # messages = [{
            #     "role": "system", "content": prompts.get_sys_prompt() + time_info
            # }]

            # messages.extend(self.form_chat_history())
            # messages.append(user_message)
            
            # logger.info(f"Messages: {messages}")

            messages = [{
                "role": "system", "content": prompts.get_sys_prompt()
            }]
            messages.extend(self.form_chat_history())
            messages.append({"role": "user", "content": user_input})

            messages = await self.process_tool_calls(
                messages.copy(),
                self.tools_schema
            )

            # rag_context = f"Real-time Market Data:\n{tool_info}\n\n{rag_context}" if rag_context else f"Real-time Market Data:\n{tool_info}"
            messages.append({"role": "user", "content": prompts.get_qa_template(user_input, rag_context, lang, self.get_internal_assitant_message(), potential_qa)})
            
            logger.debug(f"Messages: {messages}")

            # Use 60-second timeout to prevent hanging connections
            response = await asyncio.wait_for(
                self.async_llm.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    stream=False,
                    temperature=0,
                    top_p=0.8,
                ),
                timeout=60 
            )

            try:
                profiler.add_metric("answer_sub_tokens", response.usage.total_tokens)
            except:
                pass
            return user_input, response
        except asyncio.TimeoutError:
            logger.error(f"Chat request timed out for input: {user_input[:50]}...")
            # Return a minimal response object to prevent errors downstream
            return user_input, None
        except Exception as e:
            logger.error(f"Error in chat_async: {str(e)}")
            return user_input, None

    def add_internal_assitant_message(self, internal_input):
        self.internal_assistant_message.append({
            "Critical information": internal_input, 
            "Time": datetime.now().strftime("%H-%M-%S")})
        if len(self.internal_assistant_message) > 5:
            self.internal_assistant_message.pop(0)

        
    def add_to_qa_history(self, user_input, llm_response):
        # 将新的问答对添加到qa_history
        self.qa_history.append({
            "user": user_input,
            "assistant": llm_response
        })
        # 保证qa_history中最多有5个问答对
        if len(self.qa_history) > self.history_limit:
            self.qa_history.pop(0)  # 删除最早的一对

    def get_internal_assitant_message(self):
        return str(self.internal_assistant_message)

    def get_qa_history(self):
        # 将qa_history格式化为字符串，作为大模型的上下文
        qa_context = ""
        for qa in self.qa_history:
            qa_context += f"{{'User Question': '{qa['user']}'; 'LLM Answer': '{qa['assistant']}'}}\n"
        return qa_context
    
    def form_chat_history(self):
        # Use Deepseek format to add chat history
        chat_history = []
        for qa in self.qa_history:
            chat_history.append({"role": "user", "content": qa["user"]})
            chat_history.append({"role": "assistant", "content": qa["assistant"]})
        return chat_history  

    def modify_previous_assistant_message(self, interrupt_index):
        """
        Modify the last assistant message by truncating it after the given interrupt index.
        """
        # Find the previous assistant message (which should be the last assistant response in chat history)
        for message in reversed(self.chat_history):
            if message['role'] == 'assistant':
                # Truncate the assistant's message after the interrupt index
                modified_message = message['content'][:interrupt_index]
                
                # Update the message in chat history
                message['content'] = modified_message
                break  # We only modify the last assistant message

          
    def save_chat_history(self, response):
        assistant_message = {"role": "response", "content": response}
        self.chat_history.append(assistant_message)
        self.all_chat_history.append(assistant_message)
        self._trim_chat_history()

    def _trim_chat_history(self):
        # Keep the system message and the last `self.history_limit` user and assistant messages
        non_system_messages = [msg for msg in self.chat_history if msg['role'] != 'system']
        if len(non_system_messages) > self.history_limit:
            self.chat_history = [self.chat_history[0]] + non_system_messages[-self.history_limit:]

    def get_chat_history(self):
        # 将 chat_history 格式化为字符串，作为大模型的上下文
        chat_context = ""
        for entry in self.chat_history:
            chat_context += f"{entry['role']}: {entry['content']}\n"
        return chat_context

    def get_all_chat_history(self):
        return self.all_chat_history

    def clear_chat_history(self):
        self.qa_history = []

    def reset_rag_info(self):
        """
        reset the rag info after each Q&A
        """
        self.rag_info = pd.DataFrame({
            'timeinfo': [],
            'chunk_id': [],
            'chunk_content': [],
            'chunk_bundle_id': []
        })

    def get_runtime_log(self):
        return {
            'session_id': self.session_id,
            'need_rag': self.need_rag,
            'rewrittens': self.rewrittens,
            'hypo_chunks': self.hypo_chunks,
            'rag_info': self.rag_info.to_json(orient='records'),
            'qa_history': self.qa_history,
            'all_retrieved_content': self.all_retrieved_content,
        }
