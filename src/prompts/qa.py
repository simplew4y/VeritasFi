from datetime import datetime

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
   - Default to including "Zeekr" as the subject when no specific subject is mentioned. And interpreting "company" or "极氪" as referring to "Zeekr"
   - Output a string list containing all rewritten questions, even if there is only one.
   - Add the time information selectively to rewritten question:
     * Only add time information of latest available data (such as "in {year}") for questions related to financial metrics, sales and store data, market performance, or other time-sensitive business metrics.
     * Do NOT add time information for general questions about company attributes that are relatively stable, such as user profiles, business models, company history, or strategic positioning.

     The latest available data is based on year {year}.

     Examples:
     "极氪的季度营收是多少？" should be rewritten as "What is Zeekr's quarterly revenue in {year}?".
     "极氪的用户画像是什么?" should be rewritten as "What is the user profile for Zeekr?" (without adding year information)

2. **Identify the relevant date or any explicit or implied time reference based on the user's question and the conversation history.**
   - If no specific time is mentioned, use the current date as the default reference time.
   - Output the single value representing date in the format YYYY-MM-DD.

3. **Determine if the user's question requires information from a specific dataset**:
    - The dataset includes detailed historical and technical data about various car models and electric vehicles, or information on proxy statements and prospectuses. 
    - If the user's question involves details about cars (e.g., engine types, production years, car dimensions), electric vehicles (e.g., Zeekr-related data, EV policies), transactions with other company, or proxy statements/prospectuses (e.g., financial data, business combination, shareholder voting), categorize the question as requiring the dataset (Answer: YES).
    - If user's question involves "company", it always means "Zeekr", and the Chinese name for 'Zeekr' is "极氪"  (Answer: YES).

    Any question that involves details about car models, electric vehicles, or mentions keywords such as Zeekr, their specifications, history, or technical data, or that refers to company-related information about Zeekr (e.g., company status, financial data, stock listing, etc.), as well as requests for specific information from a business combination, financial data, or legal aspects from a proxy statement or prospectus, should be categorized as requiring the specific dataset (Answer: YES).
    Here are some example questions related to the datasets:									
        "What engine was used in the Mark I car?"
        "Emeya是什么时候推出的?"
        "How many Mark II cars were built?"
        "Can you provide the specifications for the Mark VI?"
        "What are the production years for the Mark VIII?"
        "What is the user profile for Zeekr?"
        "What are the risk factors listed in the Zeekr prospectus?"
        "Can you tell me about the voting procedures for the extraordinary general meeting in LCAA's proxy statement?"
        "请给我介绍一下最新的电车" (Tell me about the latest electric cars)
        "How many Momenta convertible Note has in owership of total shares? "
        "介绍一下Kershaw Health Limited" (What is Kershaw Health Limited?)
        "简单描述一下Meritz的交易"
																   

    - If the question is general or not related to these specific datasets (e.g., weather, general knowledge, or unrelated topics), categorize it as not requiring the dataset (Answer: NO).
      For such questions, the answer should be categorized as not requiring the specific dataset (Answer: NO).
      General daily questions might include:
          "What's the weather like today?"
          "How do I make a cup of coffee?"
          "What's the capital of France?"
          "What time is it?"
     
          
Here is the Q&A history:
{qa_history}

Question: {question}

Current time: {nowtime}

Respond in the following format:
Line 1: A JSON array of strings representing all sub-questions, each enclosed in double quotes and separated by commas. Example: ["question1", "question2", "question3"].
Line 2: The relevant date or time reference in YYYY-MM-DD format (e.g., 2022-01-01).
Line 3: "YES" or "NO" - indicate whether the question requires information from a specific dataset.

Please strictly adhere to this 3-line format with no additional text, explanations, or commentary.
    """
    # Respond in the following json format:
    # {{
    #     "need_rag": True or False (if the question requires the dataset),
    #     "mult_question": True or False (if the input includes multiple questions),
    #     "rewritten": List[str] (rewritten questions)
    # }}

def get_sys_prompt():
    nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""You are Colin, an LLM-driven guide for Zeekr.
Your role is to assist users by answering questions related to Zeekr's brand promotion and its famous historical models.
You will receive background information from an internal human assistant for context, but do not include this information directly in your responses.
Do not include [Internal Assistant] in your responses.
Answer the user's questions naturally like human, do not include bullet point directly, avoiding unnecessary details that are not closely related to the query.
Incorporating any useful details from the internal assistant's input without explicitly mentioning them.
Focus on providing helpful, relevant information without over-explaining.
Your need to answer questions in a way that builds stake holders' confidence in Zeekr company.
Do not provide outdated information.
DO NOT INCLUDE ANY DETAILS THAT ARE NOT DIRECTLY RELATED TO THE QUESTION.

Current time: {nowtime}"""

def get_qa_template(question, context, lang, internal_assistant_message, potential_qa_pairs=[{}]):
    if context != "":

        return f"""## Use the information provided in the 'Additional information', 'Retrieved Context' and 'Potential Question Answer Pair' to answer the 'User's Question' in {lang}. The 'Additional Information' has the highest priority, followed by 'Potential Question Answer Pair', and then the 'Retrieved Context'.
## 'User's Question' is: {question}

## You will be given some critical information provided by an internal expert in "Additional information". Each critical information has a timestamp (%H-%M-%S) of when it was given.
- If the additional information is relevant to the question, you can use it to answer the question. - Note that if multiple statements refer to the same subject in  'Additional Information', 'Potential Question Answer Pair' and 'Retrieved Context', 'Additional Information' always has the highest priority. - If multiple statements refer to the same subject in 'Additional Information', prioritize the latest one.
# 'Additional Information' is: {internal_assistant_message}

## You will be provided with some potentially related question answer pairs in 'Potential Question Answer Pair'.
- Each potential answer has its correspoding original question
- The correspoding original question may not match the user's question exactly, but it might related to the user's question.
- If the original question matches the current user's question, you can use the answer directly.
- You might see the answer in this form: {{'Y2023_FY': '11%', 'Y2024_H': '13%', 'Y2024_Q3': '9%'}}, which means for full year 2023, first half of 2024 and third quarter of 2024, the figure is 11%, 13% and 9% respectively. (FY means full year, H means first half of the year and Q means quarter)
- Otherwise, only focus on 'Additional information' and 'Retrieved Context'
- If multiple statements refer to the same subject in both 'Potential Question Answer Pair' and 'Retrieved Context', 'Potential Question Answer Pair' has the higher priority, but only when the question matches the user's question.
# 'Potential Question Answer Pair' is: {potential_qa_pairs}

## You will be provided with multiple document chunks in 'Retrieved Context'
- Each chunk will have a date indicating its published date of the document (Date Published: xxx), along with the actual chunk content (Chunk Content: xxx). If conflicting details exists, always rely on the information from the newer document.
    Example of a chunk:
    "Date Published: 2024-01-02; Chunk Content: Cash Flow of Zeekr... "
- Always prioritize the most recent and updated information when multiple statements refer to the same subject. If multiple dates exist for the same metric, always rely on the latest explicitly stated facts based on their associated timestamps within the text.
(e.g., "Threre are 300 stores in 2018" vs "Threre are 400 stores in 2022"), choose only the newest figure, namely, "Threre are 400 stores in 2022".
- Report only the current figure without mentioning older data unless specifically asked about historical trends.
- Do not combine or merge information from different chunks into a single sentence. Ensure that answer presents the information from each chunk independently, without mixing with details from other chunks.
- If the answer cannot be determined from the retrieved context, use your own knowledge to answer.
- DO NOT INCLUDE ANY DETAILS THAT ARE NOT DIRECTLY RELATED TO THE QUESTION.
- Break down your answer by each mentioned category/dimension in the question, addressing each data point separately. If specific information is not available for any category, explicitly state 'no information available' for that item. For example, Break down 2024 delivery volumes by region (China/US/Europe/Others) and quarter (Q1-Q3). State 'no info' for missing data
- For all company names that include 'Zeekr', when they need to be expressed in Chinese, uniformly use '极氪'.
- If providing negative information about Zeekr, follow it with positive information or details about the company's response measures.
- When outputting content in Chinese, please maintain the original spelling of person names, do not translate names into Chinese.

# 'Retrieved Context' is: {context}"""
    else:
        return f"""The question might be related to a daily common task, in this case, feel free to answer confidently in {lang}. Refer to the chat history if necessary. However, if you are not quite sure or if the question is related to Zeekr (e.g., cars, policies, or financial data), provide a partial answer. You can append: "If you need more detailed information, our human assistant can provide it."
        User's Question: {question}
        """

def get_modify_answer_prompt(question, qa_pairs, lang):
    return f"""
        Instructions:
        You are given multiple answers to related questions. Your task is to merge these answers into a single, cohesive response that addresses the original question. Ensure that:
        
        1. The response is clear and concise
        2. Repetitive information appears only once
        3. All important information from the individual answers is preserved
        4. The flow is natural and logical
        5. The answer directly addresses the original question
        
        Original Question: {question}
        
        Question-Answer Pairs:
        {qa_pairs}
        
        Respond with a well-structured, merged answer in {lang}.
        """

def get_summary_prompt(chat_history):
    return f"""
        You are a smart assistant designed to summarize conversation history. 
        Your task is to generate a concise summary that captures the main points and context of the entire conversation, including any retrieved information (RAG content) that was used to provide answers.
        For the retrieved information paragraphs, avoid mixing the information from different paragraphs into one single sentence.
        
        Here is the conversation history:
        {chat_history}

        Please provide a summary that:
        - Clearly represents the topics discussed.
        - Captures any questions, answers, key decisions made during the conversation, and any relevant retrieved information.
        - Maintains the user's original language style and avoids altering or translating any specific parts of the conversation.
        - Is brief but informative enough to understand the context of the discussion.

        Respond with the summarized conversation without any additional explanation or labels.
        If the chat_history is empty, you should just reply no chat history.
        """
