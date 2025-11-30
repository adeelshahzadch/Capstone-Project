from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=800):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100  # Overlap for maintaining context
    )
    chunks = text_splitter.split_text(text)
    return chunks

 
def chunk_text_with_metadata(text, page_number, chunk_size=800):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    
    # Return chunks with associated metadata
    return [{"text": chunk, "page": page_number} for chunk in chunks]

