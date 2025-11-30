from crewai import Agent, Task, LLM
from agents.tasks.retrieval import retrieve_context_task
# from agents.tasks.retrieval import insert_embeddings_task, retrieve_context_task
from rag_pipeline.utils.logger import log_data

 

# Log agent creation
 
llm = LLM(model="gemini/gemini-2.0-flash")
retrieval_agent = Agent( 
    role="Retrieval Agent", 
    goal="Retrieve the most relevant text chunks from vector stores to address the user's query.", 
    backstory="You specialize in searching through vector stores and retrieving semantic information. "
    "Your role is to gather and present the most relevant data from large sets of information, " 
    "ensuring it aligns with the user's query and provides meaningful insights. " 
    "Your work supports the overall process of knowledge extraction and contextual understanding, "
    "serving as a foundation for other agents to process the information further.",
    llm=None, 
    max_rpm=0,
    verbose=True,
    tasks=[retrieve_context_task],
    max_retry_limit=2,  
    max_iter=5 ,
    # function_calling_llm=llm, 

    )
 
retrieve_context_task.agent = retrieval_agent
 
