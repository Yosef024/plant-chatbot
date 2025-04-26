from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import Chunks
try:
    from chunks import DOCUMENT_CHUNKS
except ImportError:
    raise ImportError("Could not import DOCUMENT_CHUNKS from chunks.py. Make sure it exists.")

# Constants
GOOGLE_API_KEY = ""
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
TOP_K = 15

# FastAPI App
app = FastAPI(title="Plant Disease RAG Assistant")

# Prompt Template
template = """
You are an assistant for question-answering tasks for plant diseases.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and relevant to the question.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Models and Chain Initialization
embeddings_model = None
llm = None
chunk_embeddings = None
rag_chain = None

# ----------------- Initialization Functions -----------------
def load_models():
    global embeddings_model, llm
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY)

def embed_chunks():
    global chunk_embeddings
    chunk_embeddings = embeddings_model.embed_documents(DOCUMENT_CHUNKS)

def create_rag_chain():
    global rag_chain
    rag_chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )

# ----------------- RAG Helper Functions -----------------
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K):
    query_embedding = embeddings_model.embed_query(query)
    similarities = cosine_similarity(
        np.array(query_embedding).reshape(1, -1),
        np.array(chunk_embeddings)
    )[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return [DOCUMENT_CHUNKS[i] for i in top_k_indices]

def format_docs(docs):
    return "\n\n---\n\n".join(docs)

def generate_answer(query: str) -> str:
    retrieved_chunks = retrieve_relevant_chunks(query)
    context = format_docs(retrieved_chunks)
    return rag_chain.invoke({"context": context, "question": query})

# ----------------- FastAPI Endpoints -----------------
class QueryInput(BaseModel):
    message: str

@app.post("/query")
def query_llm(input_data: QueryInput):
    try:
        answer = generate_answer(input_data.message)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Run Setup on Startup -----------------
@app.on_event("startup")
def startup_event():
    load_models()
    embed_chunks()
    create_rag_chain()
