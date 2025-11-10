import streamlit as st
import requests
import os

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Child Care Chat Assistant",
    page_icon="🧸",
    layout="centered"
)

# Load backend URL
CHAT_BACKEND_URL = os.getenv("CHAT_BACKEND_URL", "http://localhost:9100")
print(f"🔗 Using backend: {CHAT_BACKEND_URL}")

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content}
if "selected_state" not in st.session_state:
    st.session_state.selected_state = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# -------------------------------------------------------------------
# HEADER & STATE SELECTION
# -------------------------------------------------------------------
st.title("🏫 Unified Child Care Chatbot")
st.caption("Ask any question about child care programs — the assistant remembers your context!")

if not st.session_state.initialized:
    try:
        # Optional health check
        health = requests.get(f"{CHAT_BACKEND_URL}/health", timeout=5)
        if health.status_code == 200:
            st.session_state.initialized = True
    except Exception as e:
        st.error(f"❌ Backend not reachable: {e}")
        st.stop()

# Let user pick a state before chatting
states = ["California", "New York"]
st.subheader("Select a State")
selected_state = st.radio(
    "Which state are you asking about?",
    options=states,
    index=0,
    horizontal=True
)
st.session_state.selected_state = selected_state

st.markdown("---")

# -------------------------------------------------------------------
# CHAT UI
# -------------------------------------------------------------------
st.subheader("💬 Chat with the Assistant")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Prompt input box
if prompt := st.chat_input("Type your question here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare payload for backend
    payload = {
        "question": prompt,
        "state": st.session_state.selected_state,
        "history": st.session_state.messages[-10:],  # last 10 for context
    }

    with st.spinner("🤔 Thinking..."):
        try:
            resp = requests.post(f"{CHAT_BACKEND_URL}/chat", json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No answer found.")
                sources = data.get("sources", [])

                # Format answer neatly
                with st.chat_message("assistant"):
                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, src in enumerate(sources, 1):
                                meta = src.get("metadata", {})
                                src_name = meta.get("source", "unknown")
                                st.markdown(f"**{i}. {src_name}**")
                                st.caption(meta)
                # Append assistant response
                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                st.error(f"❌ Backend error: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not reach chat backend. Make sure it’s running.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# -------------------------------------------------------------------
# CLEAR CHAT OPTION
# -------------------------------------------------------------------
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#777;'>"
    "💡 <b>Tip:</b> The assistant remembers the last few messages to keep context!"
    "</div>",
    unsafe_allow_html=True
)