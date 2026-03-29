import streamlit as st
from agent import build_graph, AgentState

import os
os.environ["GROQ_API_KEY"] = "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"
# 🔥 Load graph once
@st.cache_resource
def load_graph():
    return build_graph()

graph = load_graph()

# 🎨 UI CONFIG
st.set_page_config(page_title="LangGraph Agent", layout="centered")

st.title("🤖 LangGraph AI Agent")

# 🧠 Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# 📥 Input
query = st.chat_input("Ask something...")

if query:
    # Add user message
    st.session_state.chat.append(("user", query))

    with st.spinner("Thinking..."):
        try:
            state = AgentState(query=query)
            result = graph.invoke(state)

            answer = result.get("final_answer")
            route = result.get("route")

            response = f"{answer}\n\n🔀 Route: {route}"

        except Exception as e:
            response = f"❌ Error: {str(e)}"

    # Add bot message
    st.session_state.chat.append(("bot", response))

# 💬 Display chat
for role, msg in st.session_state.chat:
    if role == "user":
        with st.chat_message("user"):
            st.write(msg)
    else:
        with st.chat_message("assistant"):
            st.write(msg)