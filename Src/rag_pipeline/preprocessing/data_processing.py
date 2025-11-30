from langchain_community.document_loaders import PyPDFLoader
import whisper
from rag_pipeline.utils.logger import log_data

def extract_text_from_pdf(pdf_path):
    log_data("Insert Embeddings Tool: Loading and extracting text from PDF...")

    loader = PyPDFLoader(pdf_path)
    log_data("Insert Embeddings Tool: PDF loaded successfully.")

    documents = loader.load()
    return documents

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # You can choose a larger model if needed
     # Simulating a loading process with some time delays
    log_data("Insert Embeddings Tool: Loading Whisper model...")

    log_data("Insert Embeddings Tool: Model loaded. Transcribing audio.")
    result = model.transcribe(audio_path)
    log_data("Insert Embeddings Tool: Transcription completed.")
    return result['text']