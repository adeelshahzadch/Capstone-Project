from crewai import Agent, Task
from agents.tools.retrival import RetrieveContextTool


# Instantiate the tools
retrieve_context_tool = RetrieveContextTool(result_as_answer=True)


retrieve_context_task = Task(
    description="Retrieve the most relevant context for a given query ({user_query}) by searching the vector store (Pinecone) for matching embeddings.",
    # expected_output="Relevant context retrieved for the query '{user_query}' from Pinecone, consisting of the most relevant text chunks or embedding vectors.",
    expected_output="Raw context text only, no formatting",
    tools=[retrieve_context_tool],
    verbose = True,
    output_file = "retrieved_context.md"
)


