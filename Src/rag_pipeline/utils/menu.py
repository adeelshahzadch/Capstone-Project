
from rag_pipeline.utils.logger import log_data, log_data_only

def display_menu():
    """Displays the main RAG vector database selection menu."""
    log_data("="*40)
    log_data("        RAG CHATBOT SETUP MENU        ")
    log_data("="*40)
    log_data("Select your Vector Database option:")
    # log_data("  [1] ChromaDB   - Local Setup")
    log_data("  [1] Pinecone   - Cloud Setup")
    log_data("  [2] Exit       - Quit the setup")
    log_data("="*40)

def get_choice():
    """Gets and validates the user's database choice."""
    while True:
        log_data_only("Enter your choice (1, 2): ")
        choice = input("Enter your choice (1, 2): ").strip()
        if choice in ['1', '2']:
            return choice
        else:
            log_data("Invalid choice. Please enter 1, 2.")
