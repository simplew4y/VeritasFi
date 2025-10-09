def get_prompt(query: str, chunk: str) -> str:
        return f"""
        # Role
        You are an expert financial document annotation specialist, specializing in providing high-quality relevance annotations for question-answering systems focused on financial regulatory filings (such as SEC filings).

        # Task
        Given a financial domain user query and a document chunk, determine the relevance of the document chunk to the query.

        # Relevance Assessment Criteria
        **High Relevance (Relevant):**
        1. **Direct Answer Match**: The chunk directly contains specific data, figures, or explicit answers required by the query
        2. **Contextual Support**: The chunk provides essential background information or calculation basis necessary to answer the query, even if it doesn't contain the final answer itself
        3. **Fuzzy Time Period Match**: The chunk discusses the same topic but within a different time frame, still providing useful context

        **Low Relevance (Irrelevant):**
        1. **Generic Discussion**: Only provides industry background or company overview without supporting the specific query
        2. **Incidental Mention**: Only mentions query elements incidentally in footnotes, disclaimers, or unrelated paragraphs

        # Annotation Examples
        ---
        **Example 1: Direct Data Match**
        Query: What was Lotus Technology's Q4 2023 revenue?
        Chunk: "Lotus Technology (LOT) reported Q4 2023 revenue of $750 million, representing a 45% increase year-over-year, primarily driven by strong sales of the Eletre electric vehicle model."
        Analysis: The chunk directly provides the required Q4 2023 revenue data ($750 million), completely matching the query requirements.
        Relevance: yes
        ---
        **Example 2: Keywords Present but Content Mismatched**
        Query: What were the main risks disclosed in Tesla's 2023 10-K filing?
        Chunk: "Tesla's 2023 annual shareholder meeting highlighted the company's commitment to sustainable transportation and discussed upcoming product launches for 2024."
        Analysis: Although Tesla and 2023 are mentioned, this discusses shareholder meeting and product launches, not risk disclosures from 10-K filing.
        Relevance: no
        ---
        **Example 3: Time Period Mismatch (still relevant)**
        Query: Analyze Apple's iPhone revenue trend in Q1 2024.
        Chunk: "Apple's iPhone segment generated $65.8 billion in Q1 2023, showing resilience despite global supply chain challenges."
        Analysis: While it involves Apple iPhone revenue, the time period doesn't match (Q1 2023 vs required Q1 2024).
        Relevance: yes
        ---
        # Begin Annotation
        Query: {query}
        Chunk: {chunk}

        Respond in the following format:
        Line 1: "YES" or "NO" - indicate whether the chunk is relevant to the query.
        Line 2: Your analysis reasoning, briefly explaining why it is relevant or irrelevant.

        Please strictly adhere to this 2-line format with no additional text, explanations, or commentary.
        """