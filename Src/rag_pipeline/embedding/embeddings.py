from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

 
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Wrap text chunks into Document objects
def to_documents(chunks):
    return [Document(page_content=chunk) for chunk in chunks]

def generate_embeddings_basic(chunks):
    embeddings = embedding_model.embed_documents(chunks)
    return embeddings

def generate_query_embedding(query: str):
    return embedding_model.embed_query(query)

def generate_embeddings(chunks):
    """Generate embeddings for the text chunks."""
    text_chunks = [chunk["text"] for chunk in chunks]  
    embeddings = embedding_model.embed_documents(text_chunks)
    return embeddings

 