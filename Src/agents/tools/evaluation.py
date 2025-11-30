from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from rag_pipeline.utils.logger import log_data

from rag_pipeline.generation.answer_generation import score_answer 


class ScoreAnswerToolInput(BaseModel):
    user_query: str = Field(..., description="The original query from the user.")
    retrieved_context: str = Field(..., description="The relevant context retrieved by the Retrieval Agent.")
    final_answer: str = Field(..., description="The final, refined answer from the Reflection Agent.")

class ScoreAnswerTool(BaseTool):
    name: str = "ScoreAnswerTool"
    description: str = "Calculates and returns evaluation metrics (Accuracy, Relevance, Clarity) for the final answer generated Answer based on previous (Reason Agent's retrieve context task) based on the query and retrieved context (Retrieval Agent's retrieve context task)."
    args_schema: Type[BaseModel] = ScoreAnswerToolInput

    def _run(self, user_query: str, retrieved_context: str, final_answer: str) -> str:
        log_data("Score Answer Tool: Initializing evaluation...")
        
        report = score_answer(user_query, retrieved_context, final_answer)
        
        log_data(f"Score Answer Tool: Generated Report> \n{report}")
        
        return report