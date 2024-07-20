from flask import Flask, render_template, request
from src.helper import load_pdf, text_split, download_hugging_face_embeddings, query_openai
from langchain_community.vectorstores import Chroma  # Correct import
from langchain_community.llms import CTransformers  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import prompt_template
import os

app = Flask(__name__)

# Load environment variables from .env file
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
# Create the index with text chunks and embeddings
try:
    docsearch = chroma_instance.from_texts([t.page_content for t in text_chunks], embeddings)#index_name
except Exception as e:
    print(f"Error in creating index: {e}")
    raise

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model\\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
