import streamlit as st
import requests
import os
import time
import re

# ------------------------------------------------------------
# PAGE CONFIG — REMOVE ALL TOP RIGHT MENUS
# ------------------------------------------------------------
st.set_page_config(
    page_title="State RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    menu_items={},  # disables Deploy, Feedback, About
)

# FORCE-HIDE TOP-RIGHT ELEMENTS (hamburger, deploy, etc.)
HIDE_TOP_RIGHT = """
<style>
/* Hide generic toolbar / menu / deploy */
header [data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDeployButton"] { display: none !important; }
/* Some Streamlit builds use different classes for the menu */
.css-1rs6os, .st-emotion-cache-1rs6os { display: none !important; }
button[title="View fullscreen"] { display: none !important; }
</style>
"""
st.markdown(HIDE_TOP_RIGHT, unsafe_allow_html=True)

# ------------------------------------------------------------
# API URL
# ------------------------------------------------------------
RAG_API_URL = os.getenv("RAG_API_URL", "http://chat-backend:9100/chat")

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None

if "messages" not in st.session_state:
    # each message: {"role": "user"|"assistant", "content": str, "is_html": bool?}
    st.session_state.messages = []

if "last_followup_question" not in st.session_state:
    st.session_state.last_followup_question = None

# ------------------------------------------------------------
# CSS — Chat UI + Floating Status + Collapsible Sources
# ------------------------------------------------------------
CHAT_CSS = """
<style>

.chat-container {
    padding: 0.5rem 1rem;
}

/* USER BUBBLE (right) */
.user-msg {
    text-align: right;
    margin: 0.35rem 0;
}
.user-bubble {
    display: inline-block;
    padding: 10px 14px;
    border-radius: 18px 4px 18px 18px;
    background: linear-gradient(135deg, #4f8df9, #6fc3ff);
    color: white;
    font-size: 0.95rem;
    max-width: 85%;
}

/* ASSISTANT CONTAINER */
.assistant-msg {
    text-align: left;
    margin: 0.55rem 0;
    max-width: 95%;
}

/* ASSISTANT BUBBLE (for plain text answers) */
.assistant-bubble {
    display: inline-block;
    padding: 10px 14px;
    border-radius: 4px 18px 18px 18px;
    background: linear-gradient(135deg, #ffffff, #fefefe);
    border: 1px solid #e0e0e0;
    color: #333;
    font-size: 0.95rem;
    max-width: 100%;
}

/* Assistant header for HTML-mode answers */
.assistant-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.assistant-avatar {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #e8f0ff;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.assistant-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: #333;
}

.assistant-html {
    font-size: 0.95rem;
}

/* Floating status box */
#floating-status {
    position: fixed;
    bottom: 25px;
    right: 25px;
    background: #ffffffee;
    padding: 14px 16px;
    border-radius: 12px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    max-width: 300px;
    z-index: 9999;
    font-size: 0.9rem;
}

/* Collapsible sources */
details {
    margin: 6px 0;
    padding: 6px 10px;
    background: #fafafa;
    border-left: 3px solid #e0e0e0;
    border-radius: 4px;
}
details summary {
    font-weight: 600;
    cursor: pointer;
    font-size: 0.85rem;
}
details ul {
    margin-top: 6px;
}

</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)

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

    st.stop()

# ------------------------------------------------------------
# MAIN CHAT UI
# ------------------------------------------------------------
st.title(f"💬 Chat with the Assistant ({st.session_state.state})")
st.caption("Ask questions about childcare & adoption regulations. Answers are grounded in your state's documents.")

chat_container = st.container()

# ------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------------------
with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(
                f"<div class='user-msg'><span class='user-bubble'>🙋 {content}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            # Assistant: decide if this turn was HTML or plain text
            is_html_saved = msg.get(
                "is_html",
                bool(re.search(r"<table|<tr|<td|<th|</table", content, re.IGNORECASE)),
            )
            if is_html_saved:
                st.markdown(
                    f"""
                    <div class='assistant-msg'>
                      <div class='assistant-header'>
                        <span class='assistant-avatar'>🤖</span>
                        <span class='assistant-name'>Assistant</span>
                      </div>
                      <div class='assistant-html'>
                        {content}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class='assistant-msg'>
                      <span class='assistant-bubble'>🤖 {content}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
prompt = st.chat_input("Type your question here…")

if prompt:
    # YES logic for follow-up questions
    last_q = st.session_state.last_followup_question or ""
    if last_q and prompt.strip().lower() in {"yes", "yeah", "y", "ok", "sure"}:
        prompt = last_q

    # Show user bubble immediately
    with chat_container:
        st.markdown(
            f"<div class='user-msg'><span class='user-bubble'>🙋 {prompt}</span></div>",
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({"role": "user", "content": prompt})

    # --------------------------------------------------------
    # FLOATING STATUS
    # --------------------------------------------------------
    floating_box = st.empty()
    floating_box.markdown(
        """
        <div id="floating-status">
            <b>🔄 Processing…</b><br>
            Contacting backend…
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # BACKEND REQUEST
    # --------------------------------------------------------
    try:
        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-10:]
        ]

        payload = {
            "question": prompt,
            "state": st.session_state.state,
            "history": history_payload,
        }

        resp = requests.post(RAG_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer", "⚠️ No answer returned.")
        next_q = data.get("next_question", "")
        sources = data.get("sources", [])

        floating_box.markdown(
            """<div id="floating-status"><b>🧠 Generating answer…</b></div>""",
            unsafe_allow_html=True,
        )

    except Exception as e:
        answer = f"<span style='color:red;'>❌ Backend error: {e}</span>"
        next_q = ""
        sources = []

    # --------------------------------------------------------
    # DETECT HTML ANSWER
    # --------------------------------------------------------
    is_html = bool(re.search(r"<table|<tr|<td|<th|</table", answer, re.IGNORECASE))

    # Build assistant text (no HTML tags for follow-up so typewriter still works)
    assistant_text = answer
    if next_q:
        assistant_text += f"\n\n <b>{next_q}</b>"

    # --------------------------------------------------------
    # DISPLAY ASSISTANT MESSAGE (bubbles vs HTML)
    # --------------------------------------------------------
    with chat_container:
        if is_html:
            # Render HTML directly; no typewriter, no bubble
            st.markdown(
                f"""
                <div class='assistant-msg'>
                  <div class='assistant-header'>
                    <span class='assistant-avatar'>🤖</span>
                    <span class='assistant-name'>Assistant</span>
                  </div>
                  <div class='assistant-html'>
                    {assistant_text}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Typewriter effect inside assistant bubble
            placeholder = st.empty()
            typed = ""
            words = assistant_text.split(" ")

            for i, w in enumerate(words):
                typed += (" " if i else "") + w
                bubble_html = (
                    "<div class='assistant-msg'>"
                    f"<span class='assistant-bubble'>🤖 {typed}</span>"
                    "</div>"
                )
                placeholder.markdown(bubble_html, unsafe_allow_html=True)
                time.sleep(0.02)

            final_html = (
                "<div class='assistant-msg'>"
                f"<span class='assistant-bubble'>🤖 {assistant_text}</span>"
                "</div>"
            )
            placeholder.markdown(final_html, unsafe_allow_html=True)

        # ----------------------------------------------------
        # COLLAPSIBLE SOURCES
        # ----------------------------------------------------
        if sources:
            html = "<h4>Sources</h4>"
            for group in sources:
                fname = group.get("source", "unknown")
                chunks = group.get("chunks", [])
                html += f"<details><summary>{fname}</summary><ul>"
                for c in chunks:
                    idx = c.get("chunk_index")
                    snip = c.get("snippet", "")
                    html += f"<li><b>[Chunk {idx}]</b> {snip}</li>"
                html += "</ul></details>"

            st.markdown(html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Remove floating box
    # --------------------------------------------------------
    floating_box.empty()

    # --------------------------------------------------------
    # Save assistant message (with HTML flag)
    # --------------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "is_html": is_html,
        }
    )
    st.session_state.last_followup_question = next_q
