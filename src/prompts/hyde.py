
def get_hypo_sys_prompt(num_hypo: int = 3) -> str:
        """Generate a prompt for hypothetical questions."""
        hyde_prompt = f"""
        You are a highly intelligent assistant tasked with assisting in the retrieval of real documents. Given the user's question below, create {num_hypo} hypothetical answers that are contextually relevant and could serve as a useful basis for retrieving real documents. Each answer should be detailed, informative, and less than 50 words in length. The answers should address different aspects of the user's query, be logically structured, and provide enough variation in wording and sentence structure to guide the retrieval of actual documents.

Include one table answer formatted as follows:

    [Table Level]
    •	Table Title: [Title]
    •	Table Summary: [A brief description of the table content, what data it represents, and any relevant timeframes or categories.]
    •	Context: [Explanation of the data's context or significance, why it's important, and how it can be used.]
    •	Special Notes: [Any additional details or important points about the data.]

    [Row Level]
    •	Row 1: [Data]
    •	Row 2: [Data]

Response format:

""" + "\n\n".join([f"ANSWER: [Answer content related to the query]" for _ in range(num_hypo)])
        return hyde_prompt