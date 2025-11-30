# file: agents/tools/retrival_tools.py
from crewai.tools import BaseTool
from rag_pipeline.storage.pinecone_db import prepare_data_for_rag_pipeline
from rag_pipeline.utils.logger import log_data
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

 