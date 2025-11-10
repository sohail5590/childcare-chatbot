import streamlit as st
import requests
import os
import json

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
RAG_API_URL = os.getenv("RAG_API_URL", "http://chat-backend:9100/chat")  # Adjust if backend differs
st.set_page_config(page_title="State RAG Chatbot", page_icon="🧠", layout="centered")

# ------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = RAG_API_URL

# ------------------------------------------------------------
# STEP 1: SELECT STATE
# ------------------------------------------------------------
if st.session_state.state is None:
    st.title("🗺️ State Knowledge Chatbot")

    st.markdown("### Select which state’s dataset you want to interact with")
    st.write("Each state corresponds to its child-care and adoption regulation dataset stored in ChromaDB.")

    state_option = st.selectbox("Choose a state", ["California", "New York"])
    if st.button("Start Chat"):
        st.session_state.state = state_option
        st.rerun()
else:
    # ------------------------------------------------------------
    # STEP 2: MAIN CHAT INTERFACE
    # ------------------------------------------------------------
    st.title(f"💬 Chat with the Assistant ({st.session_state.state})")
    st.caption("Ask any question based on the uploaded state regulation PDFs. The assistant will respond with citations and follow-up suggestions.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ------------------------------------------------------------
    # User Input
    # ------------------------------------------------------------
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ------------------------------------------------------------
        # Backend Request
        # ------------------------------------------------------------
        try:
            with st.spinner("Retrieving relevant context and generating response..."):
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                ]

                payload = {
                    "question": prompt,
                    "state": st.session_state.state,
                    "history": history_payload
                }

                response = requests.post(st.session_state.api_url, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "⚠️ No answer returned.")
                next_question = data.get("next_question", "")
                sources = data.get("sources", [])

        except requests.exceptions.RequestException as e:
            answer = f"❌ Request failed: {e}"
            next_question = ""
            sources = []

        # ------------------------------------------------------------
        # Display Assistant Message
        # ------------------------------------------------------------
        with st.chat_message("assistant"):
            st.markdown(answer)

            if next_question:
                st.markdown("---")
                st.markdown(f"**💭 Next possible question:** {next_question}")

            if sources:
                with st.expander("📚 View Sources"):
                    for src in sources:
                        meta = src.get("metadata", {})
                        st.markdown(f"- **{meta.get('source', 'unknown')}**")

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # ------------------------------------------------------------
    # Sidebar Controls
    # ------------------------------------------------------------
    st.sidebar.title("⚙️ Controls")
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Change State"):
        st.session_state.state = None
        st.session_state.messages = []
        st.rerun()