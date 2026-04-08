# 🤖 LangGraph Multi-Tool AI Agent

A modular AI agent built with **LangGraph**, **LangChain**, and **Groq LLM** that dynamically routes user queries to the right tool — math evaluation, Wikipedia lookup, or PDF-based RAG retrieval.

---

## System Architecture

```mermaid
graph TB
    subgraph Clients["🖥️ Client Layer"]
        UI["Streamlit Chat UI\napp.py"]
        API["FastAPI REST API\nmain.py · POST /ask"]
    end

    subgraph Agent["🧠 LangGraph Agent · agent.py"]
        IN["input_node\nWrap query → HumanMessage"]
        RT["router_node\nRule-based + LLM fallback"]
        TL["tool_node\nMath · eval()"]
        WK["wiki_node\nWikipedia lookup"]
        VD["vector_node\nChromaDB RAG"]
        CB["combine_node\nMerge + format answer"]
    end

    subgraph RAG["📚 RAG Core · sam.py"]
        PD["PDF Ingestion\ndata/ folder"]
        CH["Chunking\nRecursiveCharacterTextSplitter"]
        EM["Embeddings\nSentence Transformers"]
        VDB[("ChromaDB\nVector Store")]
        LD["Language Detection\nlangdetect"]
        GR["Groq LLM Answer\ngpt-oss-120b"]
        SUM["Summarizer\nllama3-8b-8192"]
        MEM["Conversation Memory\nsummaries list"]
    end

    subgraph Groq["⚡ Groq Cloud"]
        M1["openai/gpt-oss-120b\nMain reasoning model"]
        M2["llama3-8b-8192\nSummarization model"]
    end

    UI -->|query| IN
    API -->|query| IN
    IN --> RT
    RT -->|math| TL
    RT -->|wikipedia| WK
    RT -->|vector_db| VD
    TL --> CB
    WK --> CB
    VD --> CB
    CB -->|final_answer + route| UI
    CB -->|final_answer + route| API

    PD --> CH --> EM --> VDB
    VDB -->|top-K chunks| GR
    LD --> GR
    GR --> SUM --> MEM
    GR --> CB

    RT --> M1
    GR --> M1
    SUM --> M2
```

---

## LangGraph Node Flow

```mermaid
flowchart LR
    A([🟢 START]) --> B[input_node]
    B --> C[router_node]
    C -->|math_tool| D[tool_node\neval arithmetic]
    C -->|wikipedia| E[wiki_node\nWikipedia API]
    C -->|vector_db| F[vector_node\nChromaDB RAG]
    D --> G[combine_node]
    E --> G
    F --> G
    G --> H([🔴 END])
```

---

## Routing Decision Logic

```mermaid
flowchart TD
    Q[User Query] --> R1{Contains\n+ - * / or\ncalculate/solve?}
    R1 -->|Yes| MATH[🧮 math_tool\neval the expression]
    R1 -->|No| R2{Contains\nnotes/pdf/document?}
    R2 -->|Yes| VEC[📄 vector_db\nChromaDB RAG]
    R2 -->|No| LLM[🤖 LLM Router\nGroq decides]
    LLM -->|wikipedia| WIKI[🌍 wikipedia\nWikipedia API]
    LLM -->|vector_db| VEC
    LLM -->|math_tool| MATH
    MATH --> ANS[Final Answer]
    WIKI --> ANS
    VEC --> ANS
```

---

## RAG Pipeline

```mermaid
flowchart TD
    A[📄 PDF Files\nin data/ folder] --> B[pypdf Loader]
    B --> C[RecursiveCharacterTextSplitter\nchunk_size / overlap]
    C --> D[Sentence Transformer\nEmbeddings]
    D --> E[(ChromaDB\nPersisted Vector Store)]

    Q[❓ User Query] --> LD[langdetect\nLanguage Detection]
    Q --> E
    E -->|Top-5 chunks| CTX[Build Prompt\n+ last 3 summaries from memory]
    LD -->|target language| CTX
    CTX --> G["Groq · gpt-oss-120b\nNCERT-strict answer\n200–300 words"]
    G --> ANS[📝 Answer]
    ANS --> SUM["Groq · llama3-8b-8192\nOne-line summary"]
    SUM --> MEM[Conversation Memory]
    ANS --> OUT["Return\nanswer + summary + sources"]
```

---

## Conversation Memory Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant R as RAG Core (sam.py)
    participant M as Memory Store

    U->>A: Ask question (any language)
    A->>R: query + top-K chunks
    R->>M: Read last 3 summaries
    M-->>R: Previous context
    R->>R: Build prompt with context
    R-->>A: answer + sources
    R->>R: Summarize answer (llama3)
    R->>M: Append new summary
    A-->>U: Final answer + route label
```

---

## Multi-Language Support

```mermaid
graph LR
    Q[User Query] --> LD[langdetect]
    LD -->|en| EN[English]
    LD -->|hi| HI[Hindi · हिंदी]
    LD -->|ml| ML[Malayalam · മലയാളം]
    LD -->|mr| MR[Marathi · मराठी]
    LD -->|ur| UR[Urdu · اردو]
    EN & HI & ML & MR & UR --> LLM[Groq LLM\nResponds in detected language]
```

---

## Features

- **3-way Dynamic Routing** — math, Wikipedia, or vector DB, chosen automatically
- **Rule-based + LLM fallback router** — fast rules first, Groq decides on ambiguous queries
- **RAG Pipeline** (`sam.py`) — PDF ingestion, chunking, ChromaDB retrieval, Groq answers
- **Multi-language responses** — auto-detects query language and answers in kind
- **Conversation memory** — rolling 3-summary context window across turns
- **Streamlit Chat UI** — shows answer and route decision per query
- **FastAPI backend** — REST endpoint for programmatic access

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`gpt-oss-120b`, `llama3-8b-8192`) |
| Agent Framework | LangGraph, LangChain |
| Vector DB | ChromaDB |
| Embeddings | Sentence Transformers |
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Language Detection | langdetect |
| PDF Parsing | pypdf |

---

## Project Structure

```
├── agent.py          # LangGraph agent — nodes, routing, graph builder
├── app.py            # Streamlit chat UI
├── main.py           # FastAPI server — GET / and POST /ask
├── sam.py            # RAG core — Groq answers, summarization, memory
├── data/             # Drop PDF files here for ingestion
├── chroma_db_1/      # Persisted ChromaDB vector store (git-ignored)
├── local.ipynb       # Prototype notebook (3-route agent with Wikipedia)
├── requirements.txt
├── pyproject.toml
└── .env              # API keys — never committed
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/samyakdande/LangGraph-MultiState-Agent.git
cd LangGraph-MultiState-Agent
```

### 2. Install dependencies

```bash
# recommended
uv sync

# or with pip
pip install -r requirements.txt
```

### 3. Configure environment

```env
# .env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add PDFs (optional, for RAG)

Drop any PDF files into the `data/` folder before running.

### 5. Run Streamlit UI

```bash
streamlit run app.py
```

### 6. Run FastAPI server (optional)

```bash
uvicorn main:app --reload
```

---

## API Reference

### `GET /`
Health check.
```json
{ "message": "Server running ✅" }
```

### `POST /ask`
```json
{ "query": "What is 12 * 8?" }
```
```json
{
  "answer": "Result: 96",
  "route": "math"
}
```

```json
{ "query": "Who is Albert Einstein?" }
```
```json
{
  "answer": "Albert Einstein was a German-born theoretical physicist...",
  "route": "general"
}
```

---

## Example Queries

| Query | Route | Source |
|---|---|---|
| `12 * 8 + 4` | `math_tool` | Python `eval()` |
| `Who invented the telephone?` | `wikipedia` | Wikipedia API |
| `What does my PDF say about photosynthesis?` | `vector_db` | ChromaDB RAG |
| `प्रकाश संश्लेषण क्या है?` | `vector_db` | ChromaDB RAG (Hindi) |

---

## Notes

- `chroma_db_1/` and `.env` are excluded via `.gitignore`
- The router uses rule-based matching first (fast), then falls back to Groq LLM for ambiguous queries
- RAG answers are strictly grounded in the provided PDF context — no hallucination
- Conversation memory keeps the last 3 one-line summaries for context continuity
