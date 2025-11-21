import streamlit as st
import requests
import os
import json

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
RAG_API_URL = os.getenv("RAG_API_URL", "http://chat-backend:9100/chat")
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

if "last_followup_question" not in st.session_state:
    st.session_state.last_followup_question = None

# ------------------------------------------------------------
# STEP 1 — STATE SELECTION
# ------------------------------------------------------------
if st.session_state.state is None:
    st.title("🗺️ State Knowledge Chatbot")
    st.markdown("### Select which state’s regulations you want to explore.")

    state_option = st.selectbox("Choose a state", ["California", "New York"])

    if st.button("Start Chat"):
        st.session_state.state = state_option
        st.session_state.messages = []
        st.session_state.last_followup_question = None
        st.rerun()

# ------------------------------------------------------------
# STEP 2 — MAIN CHAT UI
# ------------------------------------------------------------
else:
    st.title(f"💬 Chat with the Assistant ({st.session_state.state})")
    st.caption("Ask questions about childcare & adoption regulations. All answers are grounded in your state's documents.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ------------------------------------------------------------
    # USER INPUT
    # ------------------------------------------------------------
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Retrieving context and generating answer..."):
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                ]

                payload = {
                    "question": prompt,
                    "state": st.session_state.state,
                    "history": history_payload,
                    "last_followup_question": st.session_state.last_followup_question
                }

                response = requests.post(st.session_state.api_url, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "⚠️ No answer returned.")
                next_q = data.get("next_question", "")
                sources = data.get("sources", [])

        except requests.exceptions.RequestException as e:
            answer = f"❌ Request failed: {e}"
            next_q = ""
            sources = []

        # ------------------------------------------------------------
        # FORMAT ASSISTANT ANSWER
        # ------------------------------------------------------------
        combined = answer
        if next_q:
            combined += f"\n\n---\n\n**💬 Follow-up:** {next_q}"

        with st.chat_message("assistant"):
            st.markdown(combined)

            if sources:
                with st.expander("📚 View Sources"):
                    for src in sources:
                        meta = src.get("metadata", {})
                        st.markdown(f"- **{meta.get('source', 'unknown')}**")

        st.session_state.messages.append({"role": "assistant", "content": combined})
        st.session_state.last_followup_question = next_q

    # ------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------
    st.sidebar.title("⚙️ Controls")

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_followup_question = None
        st.rerun()

    if st.sidebar.button("Change State"):
        st.session_state.state = None
        st.session_state.messages = []
        st.session_state.last_followup_question = None
        st.rerun()
