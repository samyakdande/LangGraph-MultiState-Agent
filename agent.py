from typing import List, Optional
from operator import add
from typing_extensions import Annotated

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END


# =============================
# LLM
# =============================
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )


# =============================
# STATE
# =============================
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)

    query: str
    route: Optional[str] = None

    tool_result: Optional[str] = None
    final_answer: Optional[str] = None


# =============================
# NODES
# =============================
def input_node(state):
    return {"messages": [HumanMessage(content=state.query)]}


def router_node(state):
    llm = get_llm()

    res = llm.invoke(
        f"Decide route: math or general\nQuery: {state.query}"
    )

    text = res.content.lower()

    if any(x in state.query for x in ["+", "-", "*", "/"]):
        return {"route": "math"}
    return {"route": "general"}


def tool_node(state):
    try:
        return {"tool_result": str(eval(state.query))}
    except:
        return {"tool_result": None}


def general_node(state):
    llm = get_llm()

    response = llm.invoke(state.query)

    return {"final_answer": response.content}


def combine_node(state):
    if state.tool_result:
        return {"final_answer": f"Result: {state.tool_result}"}

    return {"final_answer": "Could not compute"}


def route_decision(state):
    return state.route


# =============================
# GRAPH
# =============================
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("input", input_node)
    builder.add_node("router", router_node)
    builder.add_node("tool", tool_node)
    builder.add_node("general", general_node)
    builder.add_node("combine", combine_node)

    builder.set_entry_point("input")

    builder.add_edge("input", "router")

    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "math": "tool",
            "general": "general"
        }
    )

    builder.add_edge("tool", "combine")
    builder.add_edge("general", END)
    builder.add_edge("combine", END)

    return builder.compile()