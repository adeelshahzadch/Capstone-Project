from crewai.tools import BaseTool
from rag_pipeline.generation.answer_generation import refine_answer 
from pydantic import BaseModel, Field
from typing import Type
from rag_pipeline.utils.logger import log_data

class RefinementAnswerToolInput(BaseModel):
    context: str = Field(..., description="Generated Answer based on previous Reasoning task/agent")

class RefinementAnswerTool(BaseTool):
    name: str = "RefinementAnswerTool"
    description: str = "Generated Answer based on previous Reasoning task/agent"
    args_schema: Type[BaseModel] = RefinementAnswerToolInput  
    
    def _run(self, context: str) -> str:
        log_data(f"Refine Answer Tool: Initialed.")
        answer = refine_answer(context)
        log_data(f"Refine Answer Tool: Refined answer >  \n {answer}.")
        return answer

 