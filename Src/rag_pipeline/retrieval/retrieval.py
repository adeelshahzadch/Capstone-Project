from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def retrieve_relevant_chunks(query, vector_store):
    if vector_store is None:
        raise ValueError("Vector store is not initialized properly.")
    query_embedding = embedding_model.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=8)
  
    return results