# =========================
# rag_core.py (PART 1)
# =========================

import os
from groq import Groq
from dotenv import load_dotenv
from langdetect import detect

load_dotenv()

# =========================
# GROQ CLIENT
# =========================
client = Groq(api_key=os.getenv("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"))

# =========================
# Conversation Memory
# =========================
conversation_memory = {
    "summaries": []
}

# =========================
# Language Detection
# =========================
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# =========================
# GROQ ANSWER (REPLACES OLLAMA)
# =========================
def groq_answer_with_citations(question, retrieved_docs, memory):

    lang = detect_language(question)

    language_map = {
        "hi": "Hindi (हिंदी)",
        "en": "English",
        "ml": "Malayalam (മലയാളം)",
        "mr": "Marathi (मराठी)",
        "ur": "Urdu (اردو)"
    }

    target_language = language_map.get(lang, "the same language as the question")

    context = "\n\n".join(d["text"] for d in retrieved_docs[:5])
    previous = "\n".join(memory["summaries"][-3:])

    prompt = f"""
SYSTEM INSTRUCTION (MANDATORY — DO NOT VIOLATE):
You MUST answer ONLY in {target_language}.
--------------------
STRICT NCERT RULES:
- Use ONLY the provided NCERT context
- Do NOT hallucinate
- Length: 200-300 words
- If answer not found, say:
"I don't know the answer based on the provided context."
--------------------
Conversation context:
{previous if previous else "None"}

NCERT Context:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an NCERT tutor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# =========================
# rag_core.py (PART 2)
# =========================

def groq_summarise_answer(answer, lang):

    language_map = {
        "hi": "Hindi (हिंदी)",
        "en": "English",
        "ml": "Malayalam (മലയാളം)",
        "mr": "Marathi (मराठी)",
        "ur": "Urdu (اردو)"
    }

    target_language = language_map.get(lang, "same language")

    prompt = f"""
Summarize the following answer in ONE line.
Language: {target_language}

Answer:
{answer}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# =========================
# Source Extraction
# =========================
def extract_stable_sources(source_docs, max_sources=3):
    sources = []
    for d in source_docs:
        src = d["metadata"].get("source")
        if src and src not in sources:
            sources.append(src)
        if len(sources) == max_sources:
            break
    return sources


# =========================
# MAIN FUNCTION (USED BY app.py)
# =========================
def generate_answer_from_chunks(question, chunks):

    answer = groq_answer_with_citations(
        question=question,
        retrieved_docs=chunks,
        memory=conversation_memory
    )

    lang = detect_language(question)

    summary = groq_summarise_answer(answer, lang)

    conversation_memory["summaries"].append(summary)

    return {
        "answer": answer,
        "summary": summary,
        "sources": extract_stable_sources(chunks)
    }