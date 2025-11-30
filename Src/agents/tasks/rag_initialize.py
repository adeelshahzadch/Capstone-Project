from crewai import Agent, Task
from agents.tools.rag_initialize import InsertEmbeddingsTool

# Instantiate the tools
insert_embeddings_tool = InsertEmbeddingsTool()

# Define tasks for the Retrieval Agent
insert_embeddings_task = Task(
    description="Fetch text and audio data from documents (PDF, Audio), " \
    "Break them into chunks, Convert those chunks into embeddings, " \
    "Store them in Pinecone vector store for efficient retrieval.",
    expected_output="Vector store in Pinecone populated with embeddings from text and audio data, " \
    "where each chunk of data is represented as an embedding vector for efficient similarity-based retrieval.",
    tools=[insert_embeddings_tool]   
)

