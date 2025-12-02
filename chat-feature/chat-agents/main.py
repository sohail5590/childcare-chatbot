# main.py

import os
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from form_parser import extract_text
from prompts import FORM_CLASSIFICATION_SYSTEM_PROMPT, RAG_DELEGATION_PROMPT

# URL of your existing chat-backend FastAPI
CHAT_BACKEND_URL = os.getenv("CHAT_BACKEND_URL", "http://chat-backend:9100")


class AgentResponse(BaseModel):
    form_type_guess: str
    summary: str
    validation_result: str
    answer: str
    raw_rag_answer: str
    sources: Optional[List[Dict[str, Any]]] = None


app = FastAPI(title="Childcare Chat Agents API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def call_rag_backend(prompt: str, state: str, history: Optional[List[Dict[str, str]]] = None) -> dict:
    """
    Call your existing chat-backend /chat endpoint.

    IMPORTANT: This honors your existing ChatRequest model:
      - question: str
      - state: str
      - history: Optional[List[ChatTurn]] = []
    """
    payload = {
        "question": prompt,
        "state": state,
        "history": history or [],
    }

    try:
        resp = requests.post(
            f"{CHAT_BACKEND_URL}/chat",
            json=payload,
            timeout=120,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat backend unreachable: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Chat backend error: {resp.text}",
        )

    return resp.json()


@app.post("/analyze_form", response_model=AgentResponse)
async def analyze_form(
    state: str = Form(...),
    file: UploadFile = File(...),
    user_query: Optional[str] = Form(None),
    history_json: Optional[str] = Form(None),
):
    """
    Endpoint used by chat-frontend when:
      - user uploads a form (CCP7, incapacity, etc)
      - and then hits Enter / send with a question.

    Flow:
      1. Read file, extract text.
      2. Build a 'mega prompt' that:
         - includes classification instructions
         - includes form text
         - includes user question (or default validation question)
      3. Send that prompt to your existing RAG backend (/chat)
         with the given state.
      4. Wrap and return results for the frontend.
    """
    # 1) Extract text from uploaded file
    raw_bytes = await file.read()
    detected_type, text = extract_text(file.filename, raw_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded document.")

    # 2) Decide what question to ask if user didn't provide one
    default_question = (
        "Please identify what type of childcare assistance form this is, "
        "what information it should contain under program policy, whether "
        "the uploaded form appears complete and consistent, and point out "
        "any missing or inconsistent fields."
    )
    effective_question = (user_query or default_question).strip()

    # 3) Compose question to send into your RAG /chat endpoint
    #    NOTE: we do NOT change your chat-backend prompts;
    #    we just embed our instructions + form text + user question into 'question'.
    composed_question = f"""
{FORM_CLASSIFICATION_SYSTEM_PROMPT}

Detected uploaded file name: {file.filename}
Detected file type (by extension): {detected_type}

FORM_TEXT (truncated if very long):
\"\"\"{text[:15000]}\"\"\"

USER_QUESTION:
\"\"\"{effective_question}\"\"\"

{RAG_DELEGATION_PROMPT}
"""

    # If you later want multi-turn memory for this form, you can decode history_json
    # and pass a non-empty history list into call_rag_backend. For now we ignore it.
    history_list: List[Dict[str, str]] = []

    # 4) Call existing RAG backend
    rag_result = call_rag_backend(composed_question, state=state, history=history_list)

    # Expect the structure from ChatResponse in chat-backend:
    # {
    #   "answer": "...",
    #   "next_question": "...",
    #   "sources": [ ... ]
    # }
    answer_text = rag_result.get("answer", "")
    sources_raw = rag_result.get("sources", []) or []

    # We don't parse out explicit form_type from answer yet,
    # but you CAN later update your prompts/chat-backend to add a JSON header if you want.
    guessed_type = "auto-detected (see explanation in answer)"

    return AgentResponse(
        form_type_guess=guessed_type,
        summary="See validation_result / answer for details.",
        validation_result=answer_text,
        answer=answer_text,
        raw_rag_answer=answer_text,
        sources=sources_raw or None,
    )
