import os # Still needed for file path operations if any, but not for getenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings # Use the new package
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Try to import chunks ---
try:
    # Make sure chunks.py is in the same directory
    from chunks import DOCUMENT_CHUNKS
except ImportError:
    print("Error: Could not import DOCUMENT_CHUNKS from chunks.py.")
    print("Please ensure chunks.py exists in the same directory and")
    print("you have run the PDF processing script first.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred importing chunks: {e}")
    exit()

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
TOP_K = 20

# --- Load API Key (Hardcoded) ---
# !!! WARNING: Hardcoding keys is a security risk. Consider environment variables for shared code. !!!
GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY_HERE" # <--- PASTE YOUR GEMINI API KEY HERE

# Basic check if the key was actually pasted
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_ACTUAL_API_KEY_HERE":
     print("Error: Please replace 'YOUR_ACTUAL_API_KEY_HERE' with your actual Google API Key in the script.")
     exit()

# --- Initialize Components ---

# 1. Embedding Model (BERT via HuggingFace - Updated)
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
try:
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
       )
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

# 2. LLM (Gemini) - Initialize the llm variable
print(f"Initializing LLM: {GEMINI_MODEL_NAME}...")
try:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    print("LLM initialized.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Please ensure your GOOGLE_API_KEY is correct and activated.")
    exit()


# --- Pre-compute Chunk Embeddings ---
print(f"Generating embeddings for {len(DOCUMENT_CHUNKS)} chunks...")
try:
    chunk_embeddings = embeddings_model.embed_documents(DOCUMENT_CHUNKS)
    print(f"Embeddings generated successfully. Shape: {np.array(chunk_embeddings).shape}")
except Exception as e:
    print(f"Error generating chunk embeddings: {e}")
    exit()

# --- Define Retrieval Function (Manual Similarity Search) ---
def retrieve_relevant_chunks(query, top_k=TOP_K):
    """
    Embeds the query and finds the top_k most similar chunks.
    """
    print(f"\nEmbedding query: '{query}'")
    query_embedding = embeddings_model.embed_query(query)

    # Calculate cosine similarity
    similarities = cosine_similarity(
        np.array(query_embedding).reshape(1, -1),
        np.array(chunk_embeddings)
    )[0]

    # Get indices of top_k chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    print(f"Retrieving top {top_k} chunks with indices: {top_k_indices}")
    relevant_chunks = [DOCUMENT_CHUNKS[i] for i in top_k_indices]
    return relevant_chunks

# --- Define the RAG Chain ---

template = """
You are an assistant for question-answering tasks for plant diseases.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and relevant to the question.
understand the question and the retrieved context, and generate a smart answer from it.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n---\n\n".join(doc for doc in docs)

# Create the RAG chain using Langchain Expression Language (LCEL)
rag_chain = (
    RunnablePassthrough()
    | prompt
    | llm # This now correctly refers to the initialized ChatGoogleGenerativeAI instance
    | StrOutputParser()
)

# --- Querying Loop ---
print("\n--- RAG System Ready ---")
while True:
    query = input("\nEnter your question (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    if not query:
        continue

    # 1. Retrieve relevant chunks manually
    retrieved_chunks = retrieve_relevant_chunks(query)
    context_string = format_docs(retrieved_chunks)

    print("\n--- Sending to LLM ---")
    print(f"Context provided:\n{context_string[:500]}...")
    print(f"Question: {query}")

    # 2. Generate answer using the RAG chain
    try:
        answer = rag_chain.invoke({
            "context": context_string,
            "question": query
        })
        print("\n--- Answer ---")
        print(answer)
        print("--------------")
    except Exception as e:
        print(f"\nError during LLM invocation: {e}")
        if "API key not valid" in str(e):
             print("Check if your GOOGLE_API_KEY is correct and has Gemini API enabled.")
        # You might encounter other API errors like quota issues depending on usage
        elif "permission" in str(e).lower() or "quota" in str(e).lower():
             print("Check API key permissions and quota limits in your Google Cloud/AI Studio project.")

print("\nExiting RAG system.")

