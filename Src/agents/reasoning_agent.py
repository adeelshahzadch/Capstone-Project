from crewai import Agent, LLM
from agents.tasks.reasoning  import generate_answer_task
 
# Define the Reasoning Agent
llm = LLM(model="gemini/gemini-2.0-flash")

reasoning_agent = Agent(
    role="Reasoning", 
    goal="Analyze the retrieved context from previous tasks (Retrieval Agent's retrieve context task), synthesize relevant information, and generate a coherent, context-based answer to the user's query using a custom reasoning method.",
    backstory="You specialize in analyzing the retrieved context, synthesizing relevant information, and generating coherent, context-based answers. Your reasoning method uses a custom-built LLM approach that processes the context and generates precise answers, ensuring that all responses are grounded in the provided context and do not rely on external knowledge.",
    llm="gemini/gemini-2.0-flash",
    function_calling_llm=llm,   
    max_rpm=10,
    verbose=True,
    max_retry_limit=2,  # Default is 2, reduce if needed
    max_iter=5,  # Also reduce max iterations
    tasks=[generate_answer_task],
)

generate_answer_task.agent = reasoning_agent
 
