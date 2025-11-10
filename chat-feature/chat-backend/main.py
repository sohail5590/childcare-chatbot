import os
import time
import json
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

# ==========================================================
# 🌍 Environment Setup
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  # go up twice: chat-backend → chat-feature → project root
env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")
else:
    print(f"⚠️ .env not found at {env_path}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

print(f"✅ Loaded environment from {env_path}")
print(f"CHROMA_HOST={CHROMA_HOST}:{CHROMA_PORT}")

# ==========================================================
# 🚀 App + Middleware
# ==========================================================
app = FastAPI(title="Chat Backend — Unified Child Care Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 🔗 Initialize Clients
# ==========================================================
def connect_chroma(retries=10, delay=2.0):
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
    raise RuntimeError(f"❌ Could not connect to ChromaDB: {last_err}")

chroma_client = connect_chroma()
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
oai = OpenAI(api_key=OPENAI_API_KEY)

STATE_TO_COLLECTION = {
    "california": "california_state",
    "new york": "newyork_state",
}

RETRIEVE_TOP_K_DEFAULT = 15
RERANK_TOP_K_DEFAULT = 8
MAX_HISTORY_MESSAGES = 10

# ==========================================================
# 🧩 Models
# ==========================================================
class ChatTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    state: str
    history: Optional[List[ChatTurn]] = []
    top_k: Optional[int] = RETRIEVE_TOP_K_DEFAULT
    rerank_k: Optional[int] = RERANK_TOP_K_DEFAULT

class ChatResponse(BaseModel):
    answer: str
    next_question: str
    sources: List[Dict[str, Any]]

# ==========================================================
# ⚙️ Helper Functions
# ==========================================================
def normalize_state(state: str) -> str:
    return state.strip().lower()

def pick_collection(state: str):
    key = normalize_state(state)
    name = STATE_TO_COLLECTION.get(key)
    if not name:
        raise ValueError(f"Unsupported state '{state}'. Expected one of {list(STATE_TO_COLLECTION.keys())}")
    return chroma_client.get_or_create_collection(name=name)

def retrieve(query: str, collection, top_k: int):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
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
    for it, s in zip(items, scores):
        it["rerank_score"] = float(s)
    items.sort(key=lambda x: x["rerank_score"], reverse=True)
    return items[:min(rerank_k, len(items))]

def prepare_context_for_llm(reranked, retrieved):
    seen_ids = {r["id"] for r in reranked}
    diverse = [r for r in retrieved if r["id"] not in seen_ids][:3]
    combined = reranked[:5] + diverse

    seen_texts = set()
    unique_docs = []
    for r in combined:
        if r["document"] not in seen_texts:
            unique_docs.append(r)
            seen_texts.add(r["document"])

    total_text = ""
    for d in unique_docs:
        meta = d["metadata"]
        src = meta.get("source", "unknown")
        total_text += f"\n[Source: {src}]\n{d['document']}\n"
        if len(total_text) > 12000:
            break

    return total_text.strip(), unique_docs

def trim_history(history: List[ChatTurn]) -> List[ChatTurn]:
    return history[-MAX_HISTORY_MESSAGES:] if history else []

def summarize_context(history: List[ChatTurn]) -> str:
    """Summarize the conversation context (handles ChatTurn or dict objects)."""
    if not history:
        return ""
    parts = []
    for ch in history[-5:]:
        if isinstance(ch, dict):
            content = ch.get("content", "")
        else:
            content = getattr(ch, "content", "")
        if content:
            parts.append(content.strip())
    return " ".join(parts)

# ==========================================================
# 🧠 System Prompt (Enhanced)
# ==========================================================
SYSTEM_PROMPT = """
You are a document-grounded assistant specializing in U.S. child-care, adoption, and administrative regulations.
Your goal is to generate *factually correct, well-cited answers* based **only** on the provided context excerpts.

Your behavior rules:
1. Always stay faithful to the context; do NOT hallucinate.
2. Every factual statement must include a citation like [Source: filename or section title].
3. If multiple snippets support a claim, synthesize them smoothly into a single, coherent explanation.
4. If information is insufficient or ambiguous, explicitly say so and suggest what document or section might clarify it.
5. Prefer structured formatting (numbered steps, bullet points, or short paragraphs).
6. Keep answers concise (≈300 words max).

### Follow-up Guidance
After the main answer, always produce one short **conversational follow-up question** that:
- Is relevant to both the user's query and your answer.
- Encourages continued exploration of the same regulation or process.
- Is phrased naturally (e.g., “Would you like me to explain how the home study is reviewed?”)
- Avoids repetition or trivial prompts.

### Output Format
Return your response strictly in valid JSON with this structure:
{
  "answer": "<well-structured factual answer with inline citations>",
  "next_question": "<1 engaging conversational follow-up question>"
}

If data is missing, state that clearly in the answer (e.g., “The regulation does not specify this detail…”).
Never invent citations, laws, or content not found in the given context.
"""

# ==========================================================
# 🤖 LLM Call
# ==========================================================
def generate_with_next_question(model, query, context_text, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": h.role, "content": h.content})

    context_summary = summarize_context(history) if history else ""
    user_message = (
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Summary of key snippets:\n{context_summary}\n\n"
        "Respond strictly in valid JSON format with 'answer' and 'next_question'."
    )

    messages.append({"role": "user", "content": user_message})
    resp = oai.chat.completions.create(model=model, messages=messages, temperature=0.3)
    return resp.choices[0].message.content.strip()

# ==========================================================
# 🌐 Routes
# ==========================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.time()
    collection = pick_collection(req.state)
    retrieved = retrieve(req.question, collection, top_k=req.top_k)
    reranked = rerank_passage(req.question, retrieved, rerank_k=req.rerank_k)
    context_text, context_docs = prepare_context_for_llm(reranked, retrieved)

    raw_output = generate_with_next_question(
        model=OPENAI_MODEL,
        query=req.question,
        context_text=context_text,
        history=trim_history(req.history or []),
    )

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = {"answer": raw_output, "next_question": "Would you like me to expand on that?"}

    elapsed_ms = round((time.time() - start) * 1000)
    print(f"[chat] state={req.state} retrieved={len(retrieved)} reranked={len(reranked)} time_ms={elapsed_ms}")

    # Debug: top reranked docs
    for i, r in enumerate(reranked[:3]):
        src = r['metadata'].get('source', 'unknown')
        score = r.get('rerank_score', 0)
        print(f"[DEBUG] Top{i+1} ({score:.3f}) → {src}")

    return ChatResponse(
        answer=parsed.get("answer", ""),
        next_question=parsed.get("next_question", ""),
        sources=context_docs
    )