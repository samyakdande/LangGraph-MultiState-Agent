from fastapi import FastAPI
from agent import build_graph, AgentState
import os

# 🔑 SET YOUR KEY
os.environ["GROQ_API_KEY"] = "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"

app = FastAPI()

print("🚀 Server starting...")

graph = build_graph()


@app.get("/")
def home():
    return {"message": "Server running ✅"}


@app.post("/ask")
async def ask(data: dict):
    query = data.get("query")

    if not query:
        return {"error": "No query provided"}

    try:
        state = AgentState(query=query)
        result = graph.invoke(state)

        return {
            "answer": result.get("final_answer"),
            "route": result.get("route")
        }

    except Exception as e:
        return {"error": str(e)}