import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from rag_pipeline.utils.logger import log_data 
from rag_pipeline.embedding.embeddings import generate_query_embedding, generate_embeddings
from rag_pipeline.generation.answer_generation import generate_answer
from rag_pipeline.utils.cleanup import clean_context_to_bullets_pinecone
from rag_pipeline.chunking.chunking import chunk_text_with_metadata
from rag_pipeline.data_loader.document_loading import Initialize
from rag_pipeline.utils.logger import log_data



# Initialize Pinecone
def initialize_pinecone(retrieval=False):
    load_dotenv()
    """Initialize Pinecone and create an index if it doesn't exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # Pinecone index name
    index_name = os.getenv("INDEX_NAME")

    if not retrieval:
    
        # Delete the index if it already exists
        if index_name in pc.list_indexes().names():
            log_data(f"\tIndex '{index_name}' already exists. Deleting it just for assignment purpose.")
            pc.delete_index(index_name)
        
        # Create a fresh index
        log_data(f"\tCreating new index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,  
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # Get the Pinecone index
    return pc.Index(index_name)

def handle_query_pinecone(db_name, index =None):
    """Asks the user for a query and simulates LLM processing."""
    log_data(f"--- {db_name} Chatbot Query Interface ---")
    
    while True:
        user_query = input("Your LLM Query (Type 'exit' to go back): ")
        
        if user_query.lower() in ['exit', 'quit', '3']:
            log_data("Returning to main menu.")
            break

        if not user_query.strip():
            log_data("Query cannot be empty.")
            continue

        log_data(f"Searching {db_name} for context.")
                
        
        final_answer = answer_query(user_query, index)
        log_data(f"LLM Response:\n{final_answer}")

        
def answer_query(user_query: str, index) -> str:
 
    """Answers a user query using the Pinecone index for context retrieval."""
    log_data(f"Generating embedding for the user query : {user_query}")   

    embedding_vector = generate_query_embedding(user_query)
    

    # Query the Pinecone index for similar vectors
    results = index.query(
        vector=embedding_vector, 
        top_k=15, 
        include_metadata=True
    )

    log_data(f"Query vector dimension: {len(embedding_vector)}")

    matches = results.get('matches', []) 

    if not matches:
        # If no matches are found, return a default answer or log a failure.
        log_data("Warning: Pinecone returned no relevant matches.")
        context = "No relevant context found in the knowledge base."
    else:
        # Extract the 'text' metadata from each match
        context_list = [match.metadata.get('text', '') for match in matches]
        context = " ".join(context_list)
    
    filtered_context = clean_context_to_bullets_pinecone(context)    
    return generate_answer(user_query, filtered_context)

def fetch_pinecone_context(user_query: str) -> str:
     # Initialize Pinecone
    index = initialize_pinecone(True)
    """Answers a user query using the Pinecone index for context retrieval."""
    log_data(f"Retrieve Context Tool: Generating embedding for query: {user_query}")

    embedding_vector = generate_query_embedding(user_query)
    

    # Query the Pinecone index for similar vectors
    results = index.query(
        vector=embedding_vector, 
        top_k=3, 
        include_metadata=True
    )

    log_data(f"Retrieve Context Tool: Query vector dimension {len(embedding_vector)}")

    matches = results.get('matches', []) 

    if not matches:
        # If no matches are found, return a default answer or log a failure.
        log_data("Retrieve Context Tool: Warning (Pinecone returned no relevant matches.)")
        context = "No relevant context found in the knowledge base."
    else:
        # Extract the 'text' metadata from each match
        context_list = [match.metadata.get('text', '') for match in matches]
        context = " ".join(context_list)
    
    filtered_context = clean_context_to_bullets_pinecone(context)    
    return filtered_context





def prepare_data_for_rag_pipeline():
    """Load data, chunk, embed, and store once in Pinecone."""
    # Initialize Pinecone
    index = initialize_pinecone()

    # Step 1: Load and process data
    log_data("Insert Embeddings Tool: 1 -  Data loading and processing started.")
    pdf_documents, audio_text = Initialize()

    # Chunk text
    log_data("Insert Embeddings Tool: 2 -  Chunking documents.")
    pdf_chunks = []
    for i, doc in enumerate(pdf_documents):
        page_number = doc.metadata.get("page", None)
        pdf_chunks.extend(chunk_text_with_metadata(doc.page_content, page_number))

    # Print number of chunks
    log_data(f"Insert Embeddings Tool: Total numbers of PDF chunks: {len(pdf_chunks)}")
    audio_chunks = chunk_text_with_metadata(audio_text, page_number=0)
    log_data(f"Insert Embeddings Tool: Total numbers of Audio chunks: {len(audio_chunks)}")
    
    # Generate embeddings
    log_data("Insert Embeddings Tool: 3 -  Embedding.")

    pdf_embeddings = generate_embeddings(pdf_chunks)
    print(f"Insert Embeddings Tool: Numbers of PDF embeddings generated: {len(pdf_embeddings)}") 
    audio_embeddings = generate_embeddings(audio_chunks)
    print(f"Insert Embeddings Tool: Numbers of Audio embeddings generated: {len(audio_embeddings)}") 


    log_data("Insert Embeddings Tool: 4 -  Prepare vectors to store in Pinecone.")

    #Create vectors with metadata for traceability
    pdf_vectors = []
    for i, (chunk, emb) in enumerate(zip(pdf_chunks, pdf_embeddings)):
        vector = {
            "id": f"pdf-{i}",
            "values": emb,
            "metadata": {
                "source": "Rag.pdf",
                "type": "pdf",
                "page_number": chunk.get("page", None),  
                "text": chunk.get("text", ""),   
                "length": len(chunk.get("text", "")),   
            }
        }
        pdf_vectors.append(vector)  # Make sure to app
    
    audio_vectors = []
    for i, (chunk, emb) in enumerate(zip(audio_chunks, audio_embeddings)):
        # Ensure 'page_number' is not None; default to "unknown" if it is None
        page_number = chunk.get("page", "unknown")  # Default to "unknown" if 'page' is None
        
        # Ensure 'text' is a valid string and slice it to 500 characters
        text_snippet = chunk.get("text", "")  # Store a snippet of the text (first 500 chars)

        audio_vector = {
            "id": f"audio-{i}",  # Unique ID for each vector
            "values": emb,
            "metadata": {
                "source": "Rag.mp3",
                "type": "mp3",
                "page_number": i,   
                "text": text_snippet,
                "length": len(chunk.get("text", "")),   
            }
        }
        # Add the vector to the list if it's valid
        audio_vectors.append(audio_vector)


    # Insert embeddings into Pinecone
    index.upsert(vectors=pdf_vectors)
    index.upsert(vectors=audio_vectors)
    log_data("Insert Embeddings Tool: 4 -  Pinecone Index has been updated with new embeddings.")
    return index
