import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# -------------------------------------------------------------
# ENVIRONMENT SETUP
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"⚠️ .env not found at {env_path}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

print(f"✅ Loaded environment from: {env_path}")
print(f"OPENAI_MODEL={OPENAI_MODEL}, CHROMA_HOST={CHROMA_HOST}:{CHROMA_PORT}")

STATE_TO_COLLECTION = {
    "california": "california_state",
    "new york": "newyork_state",
}

RETRIEVE_TOP_K_DEFAULT = 15
RERANK_TOP_K_DEFAULT = 6
MAX_HISTORY_MESSAGES = 10

# -------------------------------------------------------------
# INITIALIZE SERVICES
# -------------------------------------------------------------
app = FastAPI(title="Chat Backend — Unified Child Care")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def connect_chroma(retries=10, delay=1.5) -> HttpClient:
    last_err = None
    for _ in range(retries):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            print("✅ Connected to ChromaDB")
            return client
        except Exception as e:
            last_err = e
            print(f"⏳ Waiting for ChromaDB... {e}")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Chroma at {CHROMA_HOST}:{CHROMA_PORT}: {last_err}")

chroma_client = connect_chroma()

# Embedder + Reranker
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# OpenAI client
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required.")
oai = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------
# SCHEMAS
# -------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    question: str
    state: str
    history: Optional[List[ChatTurn]] = []
    top_k: Optional[int] = RETRIEVE_TOP_K_DEFAULT
    rerank_k: Optional[int] = RERANK_TOP_K_DEFAULT

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# -------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------
def normalize_state(state: str) -> str:
    return state.strip().lower()

def pick_collection(state: str):
    key = normalize_state(state)
    name = STATE_TO_COLLECTION.get(key)
    if not name:
        raise ValueError(f"Unsupported state '{state}'. Expected one of: {list(STATE_TO_COLLECTION.keys())}")
    return chroma_client.get_or_create_collection(name=name)

def retrieve(query: str, collection, top_k: int):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids   = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0] if "distances" in results else [None] * len(docs)

    items = []
    for i in range(len(docs)):
        items.append({
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })
    return items

def rerank_passage(query: str, items: List[Dict[str, Any]], rerank_k: int):
    if not items:
        return []
    pairs = [[query, it["document"]] for it in items]
    scores = reranker.predict(pairs)
    arr = []
    for it, s in zip(items, scores):
        row = dict(it)
        row["rerank_score"] = float(s)
        arr.append(row)
    arr.sort(key=lambda x: x["rerank_score"], reverse=True)
    return arr[: min(rerank_k, len(arr))]

def build_system_prompt() -> str:
    return (
        "You are a helpful assistant answering questions about US child care services. "
        "Use the provided context excerpts to answer precisely. "
        "If the answer is unclear, say so. "
        "Cite the program names or document titles when relevant. "
        "Maintain conversational awareness — references like 'it' or 'that program' "
        "should resolve to earlier discussion when possible."
    )

def build_context_block(top_docs: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, it in enumerate(top_docs, 1):
        meta = it.get("metadata", {}) or {}
        src = meta.get("source", "unknown")
        state = meta.get("state", "unknown")
        lines.append(f"[{idx}] Source: {src} · State: {state}\n{it['document']}")
    return "\n\n".join(lines)

def trim_history(history: List[ChatTurn]) -> List[ChatTurn]:
    if not history:
        return []
    return history[-MAX_HISTORY_MESSAGES:]

def openai_answer(model: str, system_text: str, user_text: str, history: List[ChatTurn]) -> str:
    messages = [{"role": "system", "content": system_text}]
    for h in history:
        role = "assistant" if h.role == "assistant" else "user"
        messages.append({"role": role, "content": h.content})
    messages.append({"role": "user", "content": user_text})
    resp = oai.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()

# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.time()
    collection = pick_collection(req.state)

    # 1️⃣ Retrieve top-K from Chroma
    retrieved = retrieve(req.question, collection, top_k=req.top_k)

    # 2️⃣ Rerank
    reranked = rerank_passage(req.question, retrieved, rerank_k=req.rerank_k)

    # 3️⃣ Build context for the LLM
    context_text = build_context_block(reranked)
    user_block = (
        f"User question:\n{req.question}\n\n"
        "Context (top retrieved documents):\n"
        f"{context_text if context_text else '[no context found]'}\n\n"
        "Answer based only on this information, using prior conversation for reference if needed."
    )

    # 4️⃣ Generate final answer with history for indirect references
    answer = openai_answer(
        model=OPENAI_MODEL,
        system_text=build_system_prompt(),
        user_text=user_block,
        history=trim_history(req.history or []),
    )

    sources = [{"id": it["id"], "metadata": it.get("metadata", {}), "document": it.get("document", "")}
               for it in reranked]

    elapsed_ms = round((time.time() - start) * 1000)
    print(f"[chat] state={req.state} retrieved={len(retrieved)} reranked={len(reranked)} time_ms={elapsed_ms}")

    return ChatResponse(answer=answer, sources=sources)