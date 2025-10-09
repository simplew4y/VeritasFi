import os
import sqlite3
import re
from difflib import SequenceMatcher
from collections import Counter
import math

def normalize_question(question):
    # Normalize a question by removing punctuation and standardizing spacing.
    q = question.strip()
    q = re.sub(r'[，。、？！：；""''（）【】［］｛｝《》〈〉「」『』〔〕…—－～]', ' ', q)
    # q = re.sub(r'(?i)lotus\s+technology', '', q)
    #q = re.sub(r'(?i)lotus\s+technology(?:\'s)?', '', question)
    q = re.sub(r'(?i)zeekr(?:\'s)?', '', question)
    return q

def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def compare_questions(question1, question2, use_normalization=True):
    if use_normalization:
        q1_normalized = normalize_question(question1)
        q2_normalized = normalize_question(question2)
        similarity = calculate_similarity(q1_normalized, q2_normalized)
    else:
        similarity = calculate_similarity(question1, question2)
    return similarity, [q1_normalized, q2_normalized]

def periods_to_dict(row_id, db_path):
    FIXED_COLS = {"question", "question_rewritten", "category", "metadata","id"}
    META_COLS  = {"last_updated", "updated_by", "is_active"}

    # 用集合做过滤，O(1) 查找
    skip_cols = FIXED_COLS | META_COLS        # 并集
    
   

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute(f"PRAGMA table_info(qa_table)")
    columns = [row[1] for row in cur.fetchall()]
    period_columns = [col for col in columns if col not in skip_cols]
    

    # 1️⃣ 只把所有 period 列选出来
    col_list = ",".join(period_columns)
    cur.execute(f"SELECT {col_list} FROM qa_table WHERE id = ?", (row_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        print(f"id = {row_id} 不存在")
        return


    # non_empty = {col: val for col, val in zip(period_columns, row) if val not in (None, "", "NULL")}
    data_dict = {}
    for time, val in zip(period_columns, row):
        # print(f"{col}: {val}")
        if val in (None,"NULL"):
            val = ""
        data_dict[time] = val
    return data_dict    
     


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.initialize(corpus)

    def initialize(self, corpus):
        for document in corpus:
            self.doc_len.append(len(document))
            freq = Counter(document)
            self.doc_freqs.append(freq)
            
            # Add to the global counts for IDF calculation
            for word, count in freq.items():
                if word not in self.idf:
                    self.idf[word] = 0
                self.idf[word] += 1
        
        # Calculate IDF scores
        for word, doc_freq in self.idf.items():
            self.idf[word] = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + self.epsilon)

    def score(self, query, index):
        score = 0.0
        doc_len = self.doc_len[index]
        frequencies = self.doc_freqs[index]
        
        for word in query:
            if word not in frequencies:
                continue
                
            freq = frequencies[word]
            numerator = self.idf[word] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += numerator / denominator
            
        return score

    def get_scores(self, query):
        scores = []
        for i in range(self.corpus_size):
            score = self.score(query, i)
            scores.append(score)
        return scores

class QuestionSimilarityFinder:
    def __init__(self, db_path=None, table_path=None):
        if db_path is None:
            # self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log', 'frequent_qa.db')
            self.db_path = '/root/autodl-tmp/dir_tzh/dev/RAG_Agent/log/frequent_qa.db'
            print(f"Using default database path: {self.db_path}")
        else:
            self.db_path = db_path
            self.table_path = table_path
        print(self.db_path)
        print(self.table_path)    
    

    def find_similar_questions_db(self, input_question, top_n=5, threshold=0.55, use_normalization=True):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, question_rewritten, answer FROM frequent_qa_pairs WHERE is_active = TRUE")
        rows = cursor.fetchall()
        results = []
        for row_id, question, question_rewritten, answer in rows:
            similarity, q_normalized = compare_questions(input_question, question_rewritten, use_normalization)
            if similarity >= threshold:
                results.append((row_id, question, question_rewritten, answer, similarity, q_normalized))
        results.sort(key=lambda x: x[4], reverse=True)
        top_results = results[:top_n]
        conn.close()
        return top_results

    def find_similar_questions_table(self, input_question, top_n=5, threshold=0.55, use_normalization=True):
        conn = sqlite3.connect(self.table_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, question_rewritten FROM qa_table WHERE is_active = TRUE")
        rows = cursor.fetchall()
        results = []
        for row_id, question, question_rewritten in rows:
            similarity, q_normalized = compare_questions(input_question, question_rewritten, use_normalization)
            if similarity >= threshold:
                results.append((row_id, question, question_rewritten, periods_to_dict(row_id,self.table_path),similarity, q_normalized))
        results.sort(key=lambda x: x[4], reverse=True)
        top_results = results[:top_n]
        conn.close()
        return top_results    

    def find_similar_questions_bm25_db(self, input_question, top_n=5, threshold=3.0):
        normalized_input = normalize_question(input_question)
        tokenized_input = normalized_input.split()
        
        # get all questions 
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, question_rewritten, answer FROM frequent_qa_pairs WHERE is_active = TRUE")
        rows = cursor.fetchall()
        
        corpus = []
        db_questions = []
        
        for row_id, question, question_rewritten, answer in rows:
            normalized_question = normalize_question(question_rewritten)
            tokenized_question = normalized_question.split()
            corpus.append(tokenized_question)
            db_questions.append((row_id, question, question_rewritten, answer, normalized_question))
        
        bm25 = BM25(corpus)
        scores = bm25.get_scores(tokenized_input)
        
        # Combine scores with question data
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                row_id, question, question_rewritten, answer, normalized_question = db_questions[i]
                results.append((row_id, question, question_rewritten, answer, score, [normalized_input, normalized_question]))

        results.sort(key=lambda x: x[4], reverse=True)
        conn.close()
        return results[:top_n]

    def find_similar_questions_bm25_table(self, input_question, top_n=5, threshold=3.0):
        normalized_input = normalize_question(input_question)
        tokenized_input = normalized_input.split()
        
        # get all questions 
        conn = sqlite3.connect(self.table_path)
        
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, question_rewritten FROM qa_table WHERE is_active = TRUE")
        rows = cursor.fetchall()
        
        corpus = []
        db_questions = []
        
        for row_id, question, question_rewritten in rows:
            normalized_question = normalize_question(question_rewritten)
            tokenized_question = normalized_question.split()
            corpus.append(tokenized_question)
            db_questions.append((row_id, question, question_rewritten, normalized_question))
        
        bm25 = BM25(corpus)
        scores = bm25.get_scores(tokenized_input)
        
        # Combine scores with question data
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                row_id, question, question_rewritten, normalized_question = db_questions[i]
                results.append((row_id, question, question_rewritten, periods_to_dict(row_id,self.table_path), score, [normalized_input, normalized_question]))

        results.sort(key=lambda x: x[4], reverse=True)
        conn.close()
        return results[:top_n]    
    
    def get_full_qa_by_id(self, question_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM frequent_qa_pairs WHERE id = ?
        """, (question_id,))
        
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
        else:
            result = None
        
        conn.close()
        return result


if __name__ == "__main__":
    finder = QuestionSimilarityFinder('/root/autodl-tmp/hyc_production/RAG_Agent/log/frequent_qa.db','/root/autodl-tmp/hyc_production/RAG_Agent/log/qa_table.db')

    #searching_question = "Who are the board members of Lotus Technology?"
    #searching_question = "What is the gross margin of Lotus Technology?"
    #searching_question = "What is the gross margin of Lotus Technology in 2024?"
    #searching_question = "What was Lotus Technology's sales volume in Europe in the first quarter of 2023?"
    # searching_question = "What is the sports car sales volume for Lotus Technology in the second quarter of 2024?"
    #searching_question = "What was the number of Lotus Technology stores in Q1 2023?"
    #searching_question = "What was the number of Lotus Technology stores in the first quarter of 2023?"
    
    searching_question = "Who are the board members of Zeekr?"
    searching_question = "What is the gross margin of Zeekr?"
    searching_question = "What is the gross margin of Zeekr in 2024?"
    searching_question = "What was Zeekr's sales volume in Europe in the first quarter of 2023?"
    # searching_question = "What is the sports car sales volume for Zeekr in the second quarter of 2024?"
    searching_question = "What was the number of Zeekr stores in Q1 2023?"
    searching_question = "What was the number of Zeekr stores in the first quarter of 2023?"

    results = finder.find_similar_questions_db(searching_question,top_n=3)
    # print(results)
    print("-"*60)
    print(f"Searching for: {searching_question}")
    print("-"*60)
    print(f"Found {len(results)} using sequence matcher")
    for r in results:
        # print(f"ID: {r[0]}, Score: {r[4]:.4f}, Question: {r[1]}, Normalizaed: {r[5][1]}")
        print(f"ID: {r[0]}, Score: {r[4]:.4f}, Question: {r[1]}")
    print("-"*60)
    bm25_results = finder.find_similar_questions_bm25_db(searching_question,top_n=3)
    print(f"Found {len(bm25_results)} using BM25")
    for i, r in enumerate(bm25_results, 1):  
        print(f"{i}. ID: {r[0]}, Score: {r[4]:.4f}, Question: {r[1]}")

    bm25_results_table = finder.find_similar_questions_bm25_table(searching_question,top_n=3)
    print(f"Found {len(bm25_results_table)} using BM25 table")
    for i, r in enumerate(bm25_results_table, 1):  
        # print(f"{i}. ID: {r[0]}, Score: {r[4]:.4f}, Question: {r[1]}")
        print(r)

    sequence_results_table = finder.find_similar_questions_table(searching_question,top_n=3)
    print(f"Found {len(sequence_results_table)} using sequence table")
    for i, r in enumerate(sequence_results_table, 1):  
        # print(f"{i}. ID: {r[0]}, Score: {r[4]:.4f}, Question: {r[1]}")
        print(r)    

