# file: agents/tools/retrival_tools.py
from crewai.tools import BaseTool
from rag_pipeline.storage.pinecone_db import prepare_data_for_rag_pipeline, fetch_pinecone_context
from rag_pipeline.utils.logger import log_data
from pydantic import BaseModel, Field
from typing import Type

class InsertEmbeddingsTool(BaseTool):
    name: str = "InsertEmbeddingsTool"
    description: str = "Stores PDF and audio embeddings and returns the index."

    def _run(self):
        log_data(f"Insert Embeddings Tool calling...")

        log_data("Insert Embeddings Tool: Storing PDF and Audio embeddings in the vector store and returning the index.")
        prepare_data_for_rag_pipeline() 
        log_data("Insert Embeddings Tool: Pinecone index has been updated successfully.")
        return "Pinecone index has been updated" 

class RetrieveContextToolInput(BaseModel):
    user_query: str = Field(..., description="Query from the user")

class RetrieveContextTool(BaseTool):
    name: str = "RetrieveContextTool"
    description: str = "Fetches relevant context from Pinecone for a user query."
    args_schema: Type[BaseModel] = RetrieveContextToolInput

    def _run(self, user_query) -> str:
        log_data(f"Retrieve Context Tool: Searching for relevant context for query: ({user_query})")
        context = fetch_pinecone_context(user_query)  # Assuming this function returns context
        log_data(f"Retrieve Context Tool: Context : \n {context}")
        return context  # Return the context so the task has output
