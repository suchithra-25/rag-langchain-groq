import os

# =========================================================
# 1. LOAD PDF
# =========================================================
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "data/sample.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError("❌ sample.pdf not found inside data/ folder")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"✅ Loaded {len(documents)} pages from PDF")


# =========================================================
# 2. SPLIT TEXT INTO CHUNKS
# =========================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Split into {len(chunks)} chunks")


# =========================================================
# 3. EMBEDDINGS (LOCAL – HUGGINGFACE)
# =========================================================
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("✅ Local HuggingFace embeddings ready")


# =========================================================
# 4. CREATE FAISS VECTOR STORE
# =========================================================
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

print("✅ FAISS index created")


# =========================================================
# 5. GROQ LLM
# =========================================================
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

print("✅ Groq LLM ready")


# =========================================================
# 6. RAG QUERY
# =========================================================
query = "Summarize the main idea of the document."

docs = retriever.invoke(query)

context = "\n\n".join(doc.page_content for doc in docs)

prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

response = llm.invoke(prompt)

print("\n================ ANSWER ================\n")
print(response.content)
print("\n========================================\n")


