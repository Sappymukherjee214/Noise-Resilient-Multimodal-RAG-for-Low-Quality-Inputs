class PromptTemplates:
    """Versioned prompt templates for multimodal RAG."""

    RAG_SYSTEM_PROMPT = """
    You are an AI Fashion Assistant trained to assist users with product queries.
    You will be provided with context retrieved from a database and a user query (which might be noisy).
    
    GUIDELINES:
    1. Base your answer ONLY on the provided context.
    2. If the context is empty or irrelevant, politely inform the user.
    3. Acknowledge any noise or ambiguity in the user's input if it affects your confidence.
    4. Maintain a professional, helpful tone.
    """

    RAG_USER_PROMPT_V1 = """
    USER QUERY: {query}
    
    RETRIEVED CONTEXT:
    {context}
    
    Based on the above, provide a detailed response:
    """

    RAG_USER_PROMPT_V2 = """
    You are processing a potentially noisy multimodal input.
    CLEANED USER INTENT: {query}
    
    RELEVANT PRODUCTS FOUND:
    {context}
    
    Synthesize a response that solves the user's request, focusing on the specific product attributes mentioned.
    """

    @classmethod
    def get_prompt(cls, version: str = "v1", **kwargs):
        if version == "v1":
            return cls.RAG_USER_PROMPT_V1.format(**kwargs)
        elif version == "v2":
            return cls.RAG_USER_PROMPT_V2.format(**kwargs)
        return cls.RAG_USER_PROMPT_V1.format(**kwargs)
