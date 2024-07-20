from src.helper import load_pdf, text_split, download_hugging_face_embeddings, query_openai
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load and process data
try:
    extracted_data = load_pdf("data/")  # Ensure the path to your PDF file is correct
    text_chunks = text_split(extracted_data)
    text_chunks = text_chunks[:10]  # Limit the number of text chunks for debugging
    embeddings = download_hugging_face_embeddings()
except Exception as e:
    print(f"Error in data processing: {e}")
    raise

# Initialize Chroma
chroma_instance = Chroma()
#index_name ='soraka-healthcare'

# Create embeddings and store them in Chroma
try:
    docsearch = chroma_instance.from_texts([t.page_content for t in text_chunks], embeddings)#index_name=index_name
except Exception as e:
    print(f"Error in creating index: {e}")
    raise
