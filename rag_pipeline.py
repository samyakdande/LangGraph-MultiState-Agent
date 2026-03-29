from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from langchain_groq import ChatGroq
import os
os.environ["GROQ_API_KEY"] = "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"

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    ) # or define again

# 🔥 LOAD + BUILD ONCE
loader = DirectoryLoader(
    "data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Dense
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Sparse
sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 3


# 🔥 HYBRID
def hybrid_retriever(query):
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)

    docs = dense_docs + sparse_docs

    seen = set()
    unique_docs = []

    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:5]


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# 🔥 FINAL RAG
def rag_with_sources(query):
    docs = hybrid_retriever(query)

    context = format_docs(docs)

    llm = get_llm()

    response = llm.invoke(
        f"""
Answer the question using the context below.

Question:
{query}

Context:
{context}

Also mention sources briefly.
"""
    )

    sources = [
        {
            "file": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in docs
    ]

    return {
        "answer": response.content,
        "sources": sources
    }