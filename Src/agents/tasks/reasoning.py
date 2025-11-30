from crewai import Task
from agents.tools.reasoning import GenerateAnswerTool
from agents.tasks.retrieval import retrieve_context_task

# Instantiate the tool for generating answers
generate_answer_tool = GenerateAnswerTool()
 

generate_answer_task = Task(
    description="""
    Using the retrieved context from the previous retrieval task, 
    generate an answer for the user query: {user_query}.
    
    Use the GenerateAnswerTool and pass both the user_query and the 
    context that was retrieved in the previous task.
    """,
    expected_output="The direct, unedited string output of the generate_answer_tool.",
    verbose=True,
    context=[retrieve_context_task],
    tools=[generate_answer_tool],
    output_file="reasoning_context.md"
)
 
