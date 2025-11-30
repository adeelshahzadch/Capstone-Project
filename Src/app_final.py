import sys

# Third-party imports
from crewai import Crew, LLM
# Local application imports
from rag_pipeline.utils.logger import log_data,log_data_only
from rag_pipeline.utils.menu import display_menu, get_choice

# Agent imports
from agents.initialize_agent import Initialize_agent
from agents.retrieval_agent import retrieval_agent
from agents.reasoning_agent import reasoning_agent
from agents.reflection_agent import reflection_agent
from agents.evaluation_agent import evaluation_agent  
# Task imports
from agents.tasks.rag_initialize import insert_embeddings_task
from agents.tasks.retrieval import retrieve_context_task
from agents.tasks.reasoning import generate_answer_task
from agents.tasks.reflection import refine_answer_task
from agents.tasks.evaluation import evaluate_answer_task  

def start_interactive_agent_crew():
    while True:
        llm = LLM(
    model="openai/gpt-4o" 
  
    )
        # Get the user query from input
        user_query = input("Enter your query (type 'exit' to quit): ")
        # Exit condition
        if user_query.lower() == 'exit':
            log_data("Exiting the program.")
            break

        
        log_data(f"Setting-up the Crew for Retrieval, Reasoning and Reflection Agents...")
        crew = Crew(agents=[retrieval_agent,reasoning_agent, reflection_agent, evaluation_agent],
                        tasks=[retrieve_context_task, generate_answer_task, refine_answer_task, evaluate_answer_task],
                        verbose=True, 
                        function_calling_llm=llm
        ) 
                     
        inputs = {
            'user_query': user_query
        }


        # Execute the crew and get the output
        crew_output = crew.kickoff(inputs=inputs)

        # log_data(f"Generated Answer: {crew_output}")


def main():
    """Main function to run the interactive script."""
    

    while True:
        display_menu()
        choice = get_choice()

        if choice == '1':
            db_name = "Pinecone"
            log_data(f"You selected **{db_name}** setup.")
            log_data(f"RAG Pipeline is executing for each step.")

            initialize_db_crew = Crew(agents=[Initialize_agent],
                         tasks=[insert_embeddings_task], verbose=False, function_calling_llm=None,  planning=False, memory=False)
            
            initialize_db_crew_output = initialize_db_crew.kickoff()  
            
            start_interactive_agent_crew()

 
        elif choice == '2':
            log_data("Exiting...!")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        log_data("Script interrupted. Exiting!")
        sys.exit(0)
