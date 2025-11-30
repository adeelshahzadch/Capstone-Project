from crewai import Agent, Task, LLM
from agents.tasks.evaluation import evaluate_answer_task
from rag_pipeline.utils.logger import log_data

# Define the Evaluation Agent
llm = LLM(model="gemini/gemini-2.0-flash")
evaluation_agent = Agent(
    role="Evaluation and Quality Assurance Agent",
    goal="Critique the final answer from the Reflection Agent to ensure it meets quality standards for Accuracy, Relevance, and Clarity, and output a detailed score report.",
    backstory="You are the final quality check in the RAG pipeline. Your expertise is in assessing the performance of the overall crew, ensuring the final deliverable is accurate, fully grounded in the retrieved context, and highly relevant to the user's query.",
    tasks=[evaluate_answer_task],
    llm="gemini/gemini-2.0-flash",
    function_calling_llm=llm,
    max_rpm=15,
    verbose=True,
    max_retry_limit=1, 
    max_iter=3 
)
evaluate_answer_task.agent = evaluation_agent
