from rag_pipeline.preprocessing.data_processing import extract_text_from_pdf , transcribe_audio
import os
from dotenv import load_dotenv
from pathlib import Path

# def Initialize():
#     load_dotenv()


#     pdf_documents = extract_text_from_pdf(os.getenv("PDF"))

#     audio_text = transcribe_audio(os.getenv("AUDIO"))

#     return pdf_documents, audio_text

def Initialize():
   
    load_dotenv()

    # --- 1. Process PDF Document ---
    pdf_path = os.getenv("PDF")
    pdf_documents = None
    
    if pdf_path and os.path.exists(pdf_path):
        print(f"Found PDF file: {pdf_path}. Extracting text...")
        pdf_documents = extract_text_from_pdf(pdf_path)
    else:
        print(f" PDF file not found or path not set (Path: {pdf_path}). Skipping PDF processing.")

    # --- 2. Process Audio File ---
    audio_path = os.getenv("AUDIO")
    audio_text = None
    
    if audio_path and os.path.exists(audio_path):
        print(f"Found Audio file: {audio_path}. Transcribing audio...")
        audio_text = transcribe_audio(audio_path)
    else:
        print(f"Audio file not found or path not set (Path: {audio_path}). Skipping audio transcription.")

    return pdf_documents, audio_text

