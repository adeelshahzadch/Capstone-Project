from crewai import Agent, LLM
from agents.tasks.rag_initialize import insert_embeddings_task
from rag_pipeline.utils.logger import log_data

 

# Log agent creation
 
Initialize_agent = Agent( 
    role="Initialization Agent", 
    goal="Retrieve the most relevant text chunks from vector stores to Pinecone.", 
    backstory="You specialize in searching through vector stores and retrieving semantic information. "
    "Your role is to gather and present the most relevant data from large sets of information, " 
    "ensuring it aligns with the user's query and provides meaningful insights. " 
    "Your work supports the overall process of knowledge extraction and contextual understanding, "
    "serving as a foundation for other agents to process the information further.",
    llm="gemini/gemini-2.0-flash", 
    max_rpm=0,
    verbose=True,
    tasks=[insert_embeddings_task],
    max_retry_limit=2,  
    max_iter=5 ,
    )
 
insert_embeddings_task.agent = Initialize_agent

 
