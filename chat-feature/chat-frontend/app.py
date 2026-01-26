import streamlit as st
import requests
import os
import time
import re
import json
import base64


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_icon(path: str) -> str:
    """Load an image and return base64 data URI."""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def strip_html_tags(text: str) -> str:
    """Remove HTML tags only (leave plain text)."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def clean_for_memory(raw_html: str) -> str:
    """
    Clean assistant text for LLM history:
    - Remove HTML tags
    - Remove any base64 image blobs
    """
    if not isinstance(raw_html, str):
        return ""
    # Remove data:image base64 blobs first
    no_b64 = re.sub(
        r"data:image\/[^;]+;base64,[A-Za-z0-9+/=]+",
        "",
        raw_html,
    )
    # Then remove normal tags
    no_tags = re.sub(r"<[^>]+>", "", no_b64)
    return no_tags.strip()


BOT_ICON = load_icon("assets/chat-agent.png")

# ------------------------------------------------------------
# PAGE CONFIG — REMOVE ALL TOP RIGHT MENUS
# ------------------------------------------------------------

st.set_page_config(
    page_title="State RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    menu_items={},  # disables Deploy, Feedback, About
)

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
# API URLs
# ------------------------------------------------------------
RAG_API_URL = os.getenv("RAG_API_URL", "http://chat-backend:9100/chat")
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://chat-agents:9300/analyze_form")

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None

if "messages" not in st.session_state:
    # each message: {"role": "user"|"assistant",
    #                "content": str (DISPLAY TEXT, may be HTML),
    #                "is_html": bool}
    st.session_state.messages = []

if "last_followup_question" not in st.session_state:
    st.session_state.last_followup_question = None

if "lookup_online" not in st.session_state:
    st.session_state.lookup_online = False
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
    width: 30px !important;
    height: 30px !important;
    border-radius: 4px !important;
    overflow: hidden !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: none !important;
}
.assistant-avatar img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
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

/* File uploader tweaks */
.stFileUploader.st-emotion-cache-0 > section{
    background:none !important;
}
.st-emotion-cache-i0nc7r{
    display:none !important;
}
.stFileUploader [data-testid="stFileUploaderDropzone"] span button{
    text-indent: -9999px;
    line-height: 0;
    position: fixed !important;
    bottom: 18px !important;
    left: 40px !important;
    z-index: 9999 !important;
}
/* Replace text with 📎 */
.stFileUploader [data-testid="stFileUploaderDropzone"] span button::after {
    line-height: initial;
    content: "📎" !important;
    text-indent: 0;
    font-size: 1.3rem !important;
}
.stFileUploader div[data-testid="stFileUploaderDropzone"] span button {
    padding: 6px 16px !important;
}
.stFileUploader {
    margin: 0 !important;
    padding: 0 !important;
}
.stFileUploader > div {
    position: fixed !important;
    bottom: 18px !important;
    left: 30px !important;
    z-index: 9999 !important;
}
.stFileUploader div[data-testid="stFileUploaderDropzone"] button {
    width: 42px !important;
    height: 42px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    background: white !important;
    border: 1px solid #d0d0d0 !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.stFileUploader div[data-testid="stFileUploaderDropzone"] button::after {
    content: "📎" !important;
    font-size: 1.4rem !important;
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

st.session_state.lookup_online = st.toggle(
    "🔎 Lookup answers online",
    value=st.session_state.lookup_online,
    help="If enabled, answers come directly from OpenAI (no vector DB / no sources).",
)

chat_container = st.container()

# ------------------------------------------------------------
# DISPLAY CHAT HISTORY (and clean any old garbage)
# ------------------------------------------------------------
with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        is_html_saved = msg.get("is_html", False)

        # 🔧 SAFETY: clean any legacy garbage that still has bubble HTML / base64 avatar
        if role == "assistant" and (
            "assistant-bubble" in content or "assistant-avatar" in content or "data:image" in content
        ):
            content = clean_for_memory(content)
            msg["content"] = content
            msg["is_html"] = False
            is_html_saved = False

        if role == "user":
            st.markdown(
                f"<div class='user-msg'><span class='user-bubble'>🙋 {content}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            if is_html_saved:
                # Answer is HTML from backend
                st.markdown(
                    f"""
                    <div class='assistant-msg'>
                      <div class='assistant-header'>
                        <span class='assistant-avatar'>
                            <img src="data:image/png;base64,{BOT_ICON}">
                        </span>
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
                # Plain text answer
                st.markdown(
                    f"""
                    <div class='assistant-msg'>
                      <span class='assistant-bubble'>
                        <img src="data:image/png;base64,{BOT_ICON}"
                             style="width:30px;height:30px;vertical-align:middle;margin-right:6px;">
                        {content}
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# FILE UPLOAD (paperclip-style area in chat)
# ------------------------------------------------------------

uploaded_form = st.file_uploader(
    "",
    type=["pdf", "docx", "doc", "png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="uploaded_form",
)

if uploaded_form:
    st.caption(
        f"Attached: **{uploaded_form.name}** — the next question will be validated against this form."
    )

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

    st.session_state.messages.append({"role": "user", "content": prompt, "is_html": False})

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
    # PREPARE SHARED HISTORY (CLEAN TEXT ONLY)
    # --------------------------------------------------------
    history_payload = []
    for m in st.session_state.messages[-10:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            content_clean = clean_for_memory(content)
        else:
            content_clean = content
        history_payload.append({"role": role, "content": content_clean})

    # --------------------------------------------------------
    # BRANCH: NORMAL RAG vs AGENT FORM ANALYSIS
    # --------------------------------------------------------
    use_agent = uploaded_form is not None

    try:
        if not use_agent:
            # ------------------ NORMAL RAG /chat ------------------
            payload = {
                "question": prompt,
                "state": st.session_state.state,
                "history": history_payload,
                "lookup_online": st.session_state.lookup_online,
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

        else:
            # ------------------ AGENT /analyze_form ------------------
            files = {
                "file": (
                    uploaded_form.name,
                    uploaded_form.getvalue(),
                    uploaded_form.type or "application/octet-stream",
                )
            }

            form_data = {
                "state": st.session_state.state,
                "user_query": prompt,
                "history_json": json.dumps(history_payload),
            }

            resp = requests.post(
                AGENT_API_URL,
                data=form_data,
                files=files,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer") or data.get("validation_result") or "⚠️ No answer returned."
            next_q = ""
            sources = data.get("sources", []) or []

            floating_box.markdown(
                """<div id="floating-status"><b>🧠 Analyzing uploaded form…</b></div>""",
                unsafe_allow_html=True,
            )

    except Exception as e:
        answer = f"<span style='color:red;'>❌ Backend error: {e}</span>"
        next_q = ""
        sources = []

    # --------------------------------------------------------
    # DETECT HTML ANSWER
    # --------------------------------------------------------
    is_html = bool(re.search(r"<table|<tr|<td|<th|</table|<ul|<ol|<li", answer, re.IGNORECASE))

    # Build assistant text for display (answer + optional follow-up)
    assistant_display = answer
    if next_q:
        assistant_display += f"\n\n<b>{next_q}</b>"

    # --------------------------------------------------------
    # DISPLAY ASSISTANT MESSAGE (bubbles vs HTML)
    # --------------------------------------------------------
    with chat_container:
        if is_html:
            # Render HTML directly (no typewriter)
            st.markdown(
                f"""
                <div class='assistant-msg'>
                  <div class='assistant-header'>
                    <span class='assistant-avatar'>
                        <img src="data:image/png;base64,{BOT_ICON}">
                    </span>
                    <span class='assistant-name'>Assistant</span>
                  </div>
                  <div class='assistant-html'>
                    {assistant_display}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Typewriter effect inside assistant bubble
            placeholder = st.empty()
            typed = ""
            words = assistant_display.split(" ")

            for i, w in enumerate(words):
                typed += (" " if i else "") + w

                bubble_html = f"""
                <div class='assistant-msg'>
                    <span class='assistant-bubble'>
                        <img src="data:image/png;base64,{BOT_ICON}"
                             class="bubble-avatar"
                             style="width:30px;height:30px;vertical-align:middle;margin-right:6px;">
                        {typed}
                    </span>
                </div>
                """
                placeholder.markdown(bubble_html, unsafe_allow_html=True)
                time.sleep(0.02)

            final_html = f"""
            <div class='assistant-msg'>
                <span class='assistant-bubble'>
                    <img src="data:image/png;base64,{BOT_ICON}"
                         class="bubble-avatar"
                         style="width:30px;height:30px;vertical-align:middle;margin-right:6px;">
                    {assistant_display}
                </span>
            </div>
            """
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
    # Save assistant message (for future display + history)
    # --------------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            # Store DISPLAY content (HTML if any)
            "content": answer,
            "is_html": is_html,
        }
    )
    st.session_state.last_followup_question = next_q
