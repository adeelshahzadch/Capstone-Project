import logging

logging.basicConfig(
    filename='chatbot.log',      # Log file to store logs
    filemode='a',                 # Append mode
    level=logging.INFO,           # Log level
    format='%(asctime)s | %(levelname)-8s | %(message)s',  # FIXED alignment
    encoding='utf-8'
)

def silence_external_loggers():
    for logger_name in ["httpx", "openai", "urllib3", "pinecone", "langchain", "asyncio", 
                    "sentence_transformers", "transformers", "torch"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# Call immediately after basicConfig to remove verbose logs
silence_external_loggers()
# Logs and Prints the message
def log_data(message):
    print(message)   
    logging.info(message)   

# Logs the message only
def log_data_only(message):
    logging.info(message)       