import os
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from form_parser import extract_text
from prompts import (
    DOCUMENT_CLASSIFIER_PROMPT,
    RULE_DISCOVERY_PROMPT,
    build_delegation_prompt,
)

from openai import OpenAI


llm_client = OpenAI()

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CHAT_BACKEND_URL = os.getenv("CHAT_BACKEND_URL", "http://chat-backend:9100")

def call_rag_backend(prompt: str, state: str) -> dict:
    print("\n===== CALLING RAG BACKEND =====")


    payload = {"question": prompt, "state": state, "history": []}

    try:
        resp = requests.post(f"{CHAT_BACKEND_URL}/chat", json=payload, timeout=120)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat backend unreachable: {e}")

    print("Backend Status:", resp.status_code)

    if resp.status_code != 200:
        print("Backend error body:", resp.text)
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    print("\n===== RAW RAG RESPONSE =====")
    print(data)
    return data



def classify_document(text: str) -> str:
    print("\n===== CLASSIFYING DOCUMENT =====")
    preview = text[:2000]
    print("Document preview:\n", preview)

    prompt = f"""
{DOCUMENT_CLASSIFIER_PROMPT}

DOCUMENT TEXT:
\"\"\"{text[:8000]}\"\"\"
"""
    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0,
    )

    doc_type = resp.choices[0].message.content.strip()
    print("DETECTED DOC TYPE:", doc_type, "\n")
    return doc_type



def get_validation_rules_for(doc_type: str, state: str) -> str:
    print("\n===== RULE DISCOVERY =====")
    print("Raw doc_type:", doc_type)

    dt = doc_type.lower()

    if "incap" in dt:
        search_query = (
            "childcare incapacity documentation requirements doctor statement "
            "diagnosis condition affecting ability to care for child expected duration "
            "licensed medical professional extent of incapacity verification"
        )
    elif "attendance" in dt:
        search_query = (
            "monthly attendance record requirements tracking absences parent signature "
            "inconsistent use non use service documentation rules"
        )
    elif "work" in dt and "schedule" in dt:
        search_query = (
            "work schedule verification requirements employer statement hours worked "
            "childcare need linked to schedule"
        )
    else:
        search_query = (
            f"childcare documentation requirements rules for {doc_type} "
            "required fields eligibility verification"
        )

    print("FINAL RULE DISCOVERY QUERY:\n", search_query)

    rag_resp = call_rag_backend(prompt=search_query, state=state)

    rules = rag_resp.get("answer", "").strip()
    print("\nRULES RETRIEVED (first 2000 chars):\n", rules[:2000])
    return rules



@app.post("/analyze_form", response_model=AgentResponse)
async def analyze_form(
    state: str = Form(...),
    file: UploadFile = File(...),
    user_query: Optional[str] = Form(None),
):
    print("\n========== NEW analyze_form REQUEST ==========")
    print("State:", state)
    print("User query:", user_query)


    raw_bytes = await file.read()
    detected_type, text = extract_text(file.filename, raw_bytes)

    print("\n===== EXTRACTED TEXT (first 2000 chars) =====")
    print(text[:2000])

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded document.")


    default_q = (
        "Identify required fields for this document under childcare policy, "
        "check completeness, and list missing or invalid items."
    )
    final_user_q = (user_query or default_q).strip()


    doc_type = classify_document(text)


    rules_text = get_validation_rules_for(doc_type, state)

 
    composed_prompt = build_delegation_prompt(
        doc_type=doc_type,
        rules=rules_text,
        text=text,
        user_q=final_user_q,
    )

    print("\n===== FINAL DELEGATION PROMPT (first 2000 chars) =====")
    print(composed_prompt[:2000])


    rag_result = call_rag_backend(prompt=composed_prompt, state=state)

    print("\n===== FINAL RAG RESULT =====")
    print(rag_result)

    answer_text = rag_result.get("answer", "")
    sources = rag_result.get("sources", []) or []

    return AgentResponse(
        form_type_guess=doc_type,
        summary="Validation completed.",
        validation_result=answer_text,
        answer=answer_text,
        raw_rag_answer=answer_text,
        sources=sources or None,
    )
