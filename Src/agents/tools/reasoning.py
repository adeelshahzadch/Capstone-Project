from crewai.tools import BaseTool
from rag_pipeline.generation.answer_generation import generate_answer 
from pydantic import BaseModel, Field
from rag_pipeline.utils.logger import log_data
from typing import Type

class GenerateAnswerToolInput(BaseModel):
    user_query: str = Field(..., description="The user's question")
    context: str = Field(default="", description="Retrieved context from previous task")

class GenerateAnswerTool(BaseTool):
    name: str = "GenerateAnswerTool"
    description: str = "Generates answers based on user query and retrieved context"
    args_schema: Type[BaseModel] = GenerateAnswerToolInput
    def _run(self, user_query: str, context: str) -> str:
        log_data(f"Reasoning Tool : Received context from vector database \n {context}")
        answer = generate_answer(user_query, context)
        log_data(f"Reasoning Tool : Generated Answer from LLM > \n {answer}")
        return answer
