from openai import OpenAI
import os
from dotenv import load_dotenv
from rag_pipeline.utils.logger import log_data, log_data_only

def build_strict_prompt(question: str, context: str) -> str:
    """
    Constructs a strict context-only prompt for RAG.
    """
    return f"""
You are an expert RAG assistant. You MUST answer strictly using the content provided in <context>. 
The context may contain noise, transcription errors, repetition, or partial sentences. Your job is to:

1. Identify clean, meaningful facts that directly answer the question.
2. Ignore noise such as repeated lines, slide numbers, broken sentences, filler words, or incomplete phrases.
3. If parts of the answer appear fragmented across multiple places in the context, combine them logically WITHOUT adding new information.
4. Even if the context is messy or incomplete, extract and synthesize whatever relevant information exists.
5. ONLY reply with 'Not found in context' if the context contains absolutely NO information related to the question.

Do NOT invent facts. Do NOT add external knowledge.

<context>
{context}
</context>

Question: {question}

Now provide the answer using ONLY the information found in the cleaned interpretation of the context.
"""


def generate_answer(question, context):

  # Add validation here
    if not context or len(context.strip()) < 50:
        return "Not found in the context."

    # Load the .env file
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    open_api_model =os.getenv("OPENAI_API_MODEL")

    formatted_prompt = build_strict_prompt(question, context)
    log_data_only("Reasoning Tool : Formatted Prompt:\n" + formatted_prompt)

    response = client.chat.completions.create(
        model=open_api_model,
        temperature=0.3, #Reduces randomness, ensures consistent, factual answers from context (not creative responses)

        messages=[
            {"role": "system", "content": "You are a helpful academic assistant for a RAG course."},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    return response.choices[0].message.content




def refine_answer(draft_answer):
  # Add validation here
    if not draft_answer or len(draft_answer.strip()) < 50:
        return "Not found in context."

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    open_api_model =os.getenv("OPENAI_API_MODEL")


    prompt = f"""
            You are a critical reviewer AI. Improve the answer below by making it:
            - clearer,
            - more accurate,
            - better structured,
            - more complete (ONLY using the given context),
            - different in wording (no verbatim repetition).

            Original Answer:
            {draft_answer}

            Provide an improved version:
            """
    log_data_only("\tRefine Answer Tool : Formatted Prompt:\n" + prompt)

    response = client.chat.completions.create(
        model=open_api_model,
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant for a RAG course."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

 
def score_answer(user_query: str, retrieved_context: str, final_answer: str) -> str:
    # 1. Construct a comprehensive scoring prompt for the LLM
    scoring_prompt = f"""
    You are an Evaluation Agent. Your task is to critique a generated answer based on the context provided.
    
    CRITERIA:
    1. ACCURACY/FAITHFULNESS (Score 1-5): Is the 'Final Answer' fully supported by the 'Retrieved Context'? 
    2. RELEVANCE (Score 1-5): Does the 'Final Answer' directly and completely address the 'User Query'?
    3. CLARITY (Score 1-5): Is the 'Final Answer' well-structured, concise, and easy to read?

    ---
    User Query: "{user_query}"
    Retrieved Context: "{retrieved_context}"
    ---
    Final Answer: "{final_answer}"
    ---

    Provide your evaluation report in the following STRICT format:
    ACCURACY_SCORE: [Score 1-5]
    RELEVANCE_SCORE: [Score 1-5]
    CLARITY_SCORE: [Score 1-5]
    OVERALL_SUMMARY: [A one-sentence summary of the performance.]
    """
    
    try:
        llm_response_text = "ACCURACY_SCORE: 5\nRELEVANCE_SCORE: 4\nCLARITY_SCORE: 5\nOVERALL_SUMMARY: The answer is highly faithful to the context but slightly misses a minor detail in the query."

        return llm_response_text
        
    except Exception as e:
        return f"Evaluation failed due to LLM error: {str(e)}"
    

def call_llm_for_scoring(prompt: str) -> str:
    if not prompt or len(prompt.strip()) < 50:
        return "Not found in context."

    # --- SIMULATED LLM RESPONSE ---
    log_data("\tScorer: Sending detailed prompt to LLM ...")
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))    
    open_api_model =os.getenv("OPENAI_API_MODEL")
    response = client.chat.completions.create(
        model=open_api_model,
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant for a RAG course."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        seed=42  # 
    )

    return response.choices[0].message.content

def score_answer(user_query: str, retrieved_context: str, final_answer: str) -> str:
    """
    Generates a comprehensive scoring prompt and calls the LLM to evaluate 
    the final answer against the context and query.
    """

    log_data("\tScorer: Generating scoring prompt.")

    # A highly structured prompt is essential for reliable LLM evaluation.
    scoring_prompt = f"""
    You are the final Quality Assurance Agent. Your sole task is to critique the 'Final Answer' based on three strict criteria.

    CRITERIA AND SCORING (1 to 5, where 5 is best):
    1. ACCURACY/FAITHFULNESS: Is the 'Final Answer' fully supported and grounded ONLY by the 'Retrieved Context'? Do not invent facts.
    2. RELEVANCE: Does the 'Final Answer' directly and completely address all parts of the 'User Query'?
    3. CLARITY: Is the 'Final Answer' well-structured, concise, and easy to read? (The Reflection Agent should have handled this).

    ---
    User Query: "{user_query}"
    Retrieved Context (The source of truth): 
    ---
    {retrieved_context}
    ---
    Final Answer (The product to be scored): 
    ---
    {final_answer}
    ---

    Provide your evaluation report by strictly adhering to the following format. Do not include any extra commentary outside of the summary.

    ACCURACY_SCORE: [Score 1-5]
    RELEVANCE_SCORE: [Score 1-5]
    CLARITY_SCORE: [Score 1-5]
    OVERALL_SUMMARY: [A brief, one-sentence summary of the answer's quality.]
    """

    raw_llm_output = call_llm_for_scoring(scoring_prompt)
    
    log_data(f"\tScorer: Raw LLM Output Received: \n{raw_llm_output}")

    report = f"""
    --- Evaluation Report ---
    Query: "{user_query[:70]}{'...' if len(user_query) > 70 else ''}"
    
    {raw_llm_output}
    
    ---
    """
    return report