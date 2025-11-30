from crewai import Agent, Task
from agents.tools.reflection import RefinementAnswerTool
from agents.tasks.reasoning import generate_answer_task
from crewai.tasks.task_output import TaskOutput
from crewai.tasks.conditional_task import ConditionalTask
from rag_pipeline.utils.logger import log_data
refine_answer_tool = RefinementAnswerTool(result_as_answer=True)

# Condition function to check if answer was found
def answer_found(output: TaskOutput) -> bool:
    """Skip refinement if answer not found"""
    log_data(f"Reflection Agent: Checking if answer found in output: {output.raw}")
    return "Not found in context" not in output.raw

# Make refinement task conditional
refine_answer_task = ConditionalTask(
    description="Refine the answer from reasoning agent",
    expected_output="A refined, well-structured answer",
    condition=answer_found,   
    tools=[refine_answer_tool],
    context=[generate_answer_task],
    verbose = True,
    output_file = "refined_answer.md"     
)

 

