from crewai import Task
from agents.tools.evaluation import ScoreAnswerTool
from agents.tasks.reflection  import refine_answer_task
from agents.tasks.retrieval  import retrieve_context_task
from agents.tasks.reasoning  import generate_answer_task
from crewai.tasks.task_output import TaskOutput
from crewai.tasks.conditional_task import ConditionalTask
from rag_pipeline.utils.logger import log_data

score_answer_tool = ScoreAnswerTool(result_as_answer=True)

# def answer_found(output: TaskOutput) -> bool:
#     """Skip refinement if answer not found"""
#     log_data(f"Evaluation Agent: Checking if answer found in output: {output.raw}")
#     return "Not found in context" not in output.raw

# evaluate_answer_task = ConditionalTask(
#     description="Score the retrieved_context answer: Retrieval Agent's retrieve context task"
#     "against the original user query ({user_query}) "
#     "and the final answer (Reflection Agent's retrieve context task)" \
#     "Based on metrics like Accuracy, Relevance, and Clarity.",
#     expected_output="A brief evaluation report detailing the scores for Accuracy, Relevance, and Clarity, and a summary confidence score (e.g., 5/5).",
#     tools=[score_answer_tool],
#     context=[retrieve_context_task, generate_answer_task , refine_answer_task],
#     output_file = "evaluation_report.md",
#     condition=answer_found
# )


def answer_found_for_eval(output: TaskOutput) -> bool:
    """Skip evaluation if reasoning found no answer"""
    # Check the reasoning task output from context
    log_data(f"Evaluation Agent: Checking context for answer")
    return "Not found in context" not in output.raw

evaluate_answer_task = ConditionalTask(
    description="Score the retrieved_context answer: Retrieval Agent's retrieve context task"
    "against the original user query ({user_query}) "
    "and the final answer (Reflection Agent's retrieve context task)" \
    "Based on metrics like Accuracy, Relevance, and Clarity.",
    expected_output="A brief evaluation report detailing the scores for Accuracy, Relevance, and Clarity, and a summary confidence score (e.g., 5/5).",
    tools=[score_answer_tool],
    context=[retrieve_context_task, generate_answer_task, refine_answer_task],  # Add reasoning task
    condition=answer_found_for_eval,  # Check reasoning output
    output_file="evaluation_report.md"
)


 
# evaluate_answer_task = Task(
#     description=(
#         "You are the final Quality Assurance Agent. Your sole task is to critique the 'Final Answer' based on three strict criteria: Accuracy, Relevance, and Clarity. "
#         "The input to this task is the combined output from the Reflection Agent, which includes the 'Final Answer'. "
#         "Your final output MUST strictly adhere to the following evaluation format. "
#         "Do not include any extra commentary outside of the summary."
#         "--- CRITERIA AND SCORING (1 to 5, where 5 is best): ---"
#         "1. ACCURACY/FAITHFULNESS: Is the 'Final Answer' fully supported and grounded ONLY by the 'Final Answer'? "
#         "2. RELEVANCE: Does the 'Final Answer' directly and completely address all parts of the 'User Query'? "
#         "3. CLARITY: Is the 'Final Answer' well-structured, concise, and easy to read?"
#         "--- REQUIRED OUTPUT FORMAT ---"
#         "ACCURACY_SCORE: [Score 1-5]"
#         "RELEVANCE_SCORE: [Score 1-5]"
#         "CLARITY_SCORE: [Score 1-5]"
#         "OVERALL_SUMMARY: [A brief, one-sentence summary of the answer's quality.]"
#         "--- START EVALUATION HERE ---"
#     ),
#     expected_output="A four-line report following the exact format: ACCURACY_SCORE, RELEVANCE_SCORE, CLARITY_SCORE, OVERALL_SUMMARY.",
#     context=[refine_answer_task],
#     verbose = True,
#     outout_file = "evaluation_report.md"
# )