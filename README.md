# 🤖 LangGraph Multi-Tool AI Agent

A modular AI agent built with **LangGraph**, **LangChain**, and **Groq LLM** that dynamically routes user queries to the right tool — math evaluation, general knowledge, or PDF-based RAG retrieval.

---

## System Architecture

```mermaid
graph TB
    subgraph Client["🖥️ Client Layer"]
        UI["Streamlit Chat UI\n(app.py)"]
        API["FastAPI REST\n(main.py)\nPOST /ask"]
    end

    subgraph Agent["🧠 LangGraph Agent (agent.py)"]
        IN["input_node\nWraps query as HumanMessage"]
        RT["router_node\nDecides route via LLM"]
        TL["tool_node\nMath eval()"]
        GN["general_node\nGroq LLM answer"]
        CB["combine_node\nFormats final answer"]
    end

    subgraph RAG["📚 RAG Pipeline (sam.py)"]
        PD["PDF Ingestion\n(data/ folder)"]
        CH["Chunking\nRecursiveCharacterTextSplitter"]
        EM["Embeddings\nSentence Transformers"]
        VDB["ChromaDB\nVector Store"]
        RET["Retriever\nTop-K chunks"]
        GR["Groq LLM\nAnswer + Summarize"]
        LD["Language Detection\nlangdetect"]
    end

    subgraph LLM["⚡ Groq Cloud"]
        M1["openai/gpt-oss-120b"]
        M2["llama3-8b-8192"]
    end

    UI -->|query| IN
    API -->|query| IN
    IN --> RT
    RT -->|math| TL
    RT -->|general| GN
    TL --> CB
    CB -->|final_answer| UI
    CB -->|final_answer| API
    GN -->|final_answer| UI
    GN -->|final_answer| API

    PD --> CH --> EM --> VDB
    VDB --> RET --> GR
    LD --> GR
    GR --> CB

    GN --> M1
    GR --> M1
    GR --> M2
```

---

## LangGraph Workflow

```mermaid
flowchart LR
    A([🟢 START]) --> B[input_node]
    B --> C[router_node]
    C -->|route = math| D[tool_node\neval arithmetic]
    C -->|route = general| E[general_node\nGroq LLM]
    D --> F[combine_node\nformat result]
    F --> G([🔴 END])
    E --> G
```

---

## RAG Pipeline

```mermaid
flowchart TD
    A[📄 PDF Files\nin data/ folder] --> B[PDF Loader\npypdf]
    B --> C[Text Chunker\nRecursiveCharacterTextSplitter]
    C --> D[Sentence Transformer\nEmbeddings]
    D --> E[(ChromaDB\nVector Store)]

    Q[❓ User Query] --> LD[Language Detection\nlangdetect]
    Q --> E
    E -->|Top-K chunks| CTX[Build Context\n+ Conversation Memory]
    LD --> CTX
    CTX --> G[Groq LLM\ngpt-oss-120b]
    G --> ANS[📝 Answer]
    ANS --> SUM[Summarizer\nllama3-8b-8192]
    SUM --> MEM[Conversation Memory\nsummaries list]
    ANS --> OUT[Return answer\n+ sources + summary]
```

---

## Routing Decision Flow

```mermaid
flowchart TD
    Q[User Query] --> CHK{Contains\n+ - * / ?}
    CHK -->|Yes| MATH[🧮 Math Route\neval the expression]
    CHK -->|No| LLM[🤖 LLM Route\nGroq general answer]
    MATH --> RES1[Result: 96]
    LLM --> RES2[Natural language answer]
```

---

## Features

- **Dynamic Routing** — automatically routes queries to:
  - Math tool (arithmetic evaluation)
  - General LLM (open-ended questions)
  - Vector DB / RAG (PDF document Q&A)

- **RAG Pipeline** (`sam.py`)
  - PDF ingestion and chunking via `RecursiveCharacterTextSplitter`
  - Embeddings with `sentence-transformers`
  - Retrieval via ChromaDB
  - Multi-language support (English, Hindi, Malayalam, Marathi, Urdu)
  - Conversation memory with summarization

- **Streamlit Chat UI** — ChatGPT-style interface showing route decisions and answers

- **FastAPI Backend** — REST endpoint at `/ask` for programmatic access

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (LLaMA3 / GPT-OSS) |
| Agent Framework | LangGraph, LangChain |
| Vector DB | ChromaDB |
| Embeddings | Sentence Transformers |
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Language Detection | langdetect |

---

## Project Structure

```
├── agent.py          # LangGraph agent — routing, tool, and general nodes
├── app.py            # Streamlit chat UI
├── main.py           # FastAPI server with /ask endpoint
├── sam.py            # RAG core — Groq answers, summarization, source extraction
├── data/             # PDF documents for ingestion
├── chroma_db_1/      # Persisted ChromaDB vector store
├── requirements.txt
├── pyproject.toml
└── .env              # API keys (not committed)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/samyakdande/LangGraph-MultiState-Agent.git
cd LangGraph-MultiState-Agent
```

### 2. Install dependencies

Using `uv` (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

### 5. Run the FastAPI server (optional)

```bash
uvicorn main:app --reload
```

---

## API Usage

**POST** `/ask`

```json
{
  "query": "What is 12 * 8?"
}
```

Response:
```json
{
  "answer": "Result: 96",
  "route": "math"
}
```

---

## Notes

- `chroma_db_1/` and `.env` are excluded from version control via `.gitignore`
- Place PDF files in the `data/` folder before ingestion
- The RAG pipeline in `sam.py` auto-detects the query language and responds accordingly
