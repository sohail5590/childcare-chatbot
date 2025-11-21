import os
import time
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# =========================================================
# Environment Setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  # chat-backend → chat-feature → project root
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"Loaded .env from {ENV_PATH}")
else:
    print(f".env not found at {ENV_PATH}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# If you had to disable SSL checks for HF previously:
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

print(f"CHROMA_HOST={CHROMA_HOST}:{CHROMA_PORT}")

# =========================================================
# App + Middleware
# =========================================================

app = FastAPI(title="Chat Backend — Unified Child Care Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Dynamic State → Collection Mapping (config.json)
# =========================================================

CONFIG_FILE = PROJECT_ROOT / "config.json"


def load_state_collection_mapping() -> Dict[str, str]:
    """Load state → Chroma collection mapping from config.json (lowercased keys)."""
    if not CONFIG_FILE.exists():
        print("config.json not found; using fallback CA/NY mapping")
        return {
            "california": "california_state",
            "new york": "newyork_state",
        }

    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)

    mapping: Dict[str, str] = {}
    for st in cfg.get("states", []):
        name = st.get("name", "").strip().lower()
        cname = st.get("collection_name")
        if name and cname:
            mapping[name] = cname

    print(f"Loaded states mapping: {mapping}")
    return mapping


STATE_TO_COLLECTION = load_state_collection_mapping()

# =========================================================
# Initialize Clients (Chroma + Embeddings + OpenAI LLM)
# =========================================================


def connect_chroma(retries: int = 12, delay: float = 2.0):
    last_err = None
    for _ in range(retries):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            print("Connected to ChromaDB")
            return client
        except Exception as e:
            last_err = e
            print(f"Waiting for ChromaDB... {e}")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to ChromaDB: {last_err}")


chroma_client = connect_chroma()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
oai = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# Pydantic Models
# =========================================================


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    state: str
    history: Optional[List[ChatTurn]] = []
    top_k: Optional[int] = 15
    rerank_k: Optional[int] = 8


class ChatResponse(BaseModel):
    answer: str
    next_question: str
    sources: List[Dict[str, Any]]

# =========================================================
# Retrieval Helpers
# =========================================================


def normalize_state(s: str) -> str:
    return s.strip().lower()


def pick_collection(state: str):
    key = normalize_state(state)
    if key not in STATE_TO_COLLECTION:
        raise ValueError(f"Unsupported state '{state}'. Available: {list(STATE_TO_COLLECTION.keys())}")
    name = STATE_TO_COLLECTION[key]
    return chroma_client.get_or_create_collection(name=name)


def retrieve(query: str, collection, top_k: int):
    """Initial vector retrieval from Chroma."""
    q_vec = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

    items = []
    for i in range(len(docs)):
        items.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if dists and i < len(dists) else None,
            }
        )
    return items

# =========================================================
# LLM-based Reranker
# =========================================================

RERANK_SYSTEM_PROMPT = """
You are a high-precision retrieval ranker.

Your task is to select ONLY the text chunks that directly answer the user’s question.
Your decision must be based purely on the semantic content of each chunk—without assuming any
specific domain, topic, or document structure.

STRICT RULES:
1. Include a chunk ONLY if it directly contains information needed to answer the question.
2. Exclude ANY chunk that:
   - Mentions similar keywords but does NOT answer the question
   - Provides background, anecdotes, examples, or supplementary details not requested
   - Describes related but different concepts
   - Contains high-level commentary instead of factual content
3. Prefer chunks that contain:
   - Direct definitions
   - Explanations
   - Factual statements
   - Explicit answers
   - Procedures, steps, or criteria
4. Penalize chunks that:
   - Are vague or generic
   - Contain tangentially related information
   - Require inference beyond what is stated
5. If fewer than 3 chunks are relevant, return ONLY the relevant ones.

ALIGNMENT GUIDANCE:
• If the question asks “why” → select chunks explaining reasons, motivations, causes.
• If the question asks “how” → select procedural or operational chunks.
• If the question asks “what is / define” → select definitional chunks.
• If the question asks for data → select quantitative chunks.
• If the question asks for statements made by someone → select quoted or attributed chunks.
• If the question asks for steps → select enumerated or ordered chunks.
• If the question asks about specific entities → select chunks explicitly referencing those entities.

DO NOT:
- Assume meaning beyond what is stated.
- Combine answers from unrelated topics.
- Include chunks simply because they “sound related.”
- Retrieve based on surface keyword matches alone.

OUTPUT:
You will be given a list of candidate chunks, each labeled like:
[ID: <chunk_id>] <chunk text>

Return ONLY valid JSON of the form:
{
  "ranked_ids": ["<chunk_id_1>", "<chunk_id_2>", "..."]
}

- ranked_ids must be ordered from most relevant to least relevant.
- Include at most the chunks that are actually useful; do NOT add ids you haven't seen.
- If no chunk is relevant, return: { "ranked_ids": [] }.
"""


def llm_rerank(query: str, items: List[Dict[str, Any]], rerank_k: int) -> List[Dict[str, Any]]:
    """Use OpenAI LLM to rerank retrieved chunks."""
    if not items:
        return []

    # Limit how many chunks we send to the model
    MAX_CANDIDATES = min(len(items), 15)
    candidates = items[:MAX_CANDIDATES]

    # Build the chunk list text
    chunk_lines = []
    for idx, it in enumerate(candidates, start=1):
        cid = it["id"]
        text = it["document"].replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + "..."
        chunk_lines.append(f"{idx}. [ID: {cid}] {text}")

    chunks_block = "\n\n".join(chunk_lines)

    user_content = (
        f"QUESTION:\n{query}\n\n"
        f"CANDIDATE CHUNKS:\n{chunks_block}\n\n"
        "Now select and rank only the chunks that directly answer the question, "
        "and return JSON as specified."
    )

    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        ranked_ids = parsed.get("ranked_ids", [])
        if not isinstance(ranked_ids, list):
            raise ValueError("ranked_ids is not a list")
    except Exception as e:
        print(f"[RERANK] LLM rerank failed, falling back to original order: {e}")
        # Just return original top rerank_k
        return items[: min(rerank_k, len(items))]

    # Map ids → item
    item_by_id = {it["id"]: it for it in candidates}
    ordered: List[Dict[str, Any]] = []

    for cid in ranked_ids:
        if cid in item_by_id:
            ordered.append(item_by_id[cid])

    if not ordered:
        # Fallback if all ids invalid
        ordered = candidates

    return ordered[: min(rerank_k, len(ordered))]

# =========================================================
# Context Building
# =========================================================


def build_answer_and_diverse_context(
    reranked: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    max_answer_docs: int = 5,
    max_diverse_docs: int = 5,
    max_chars: int = 12000,
):
    """
    Build:
      - answer_context: main docs for answering (only top reranked)
      - diverse_context_only: extra docs used only for follow-up generation
      - context_docs: unique docs used as 'sources'
    """
    top_main = reranked[:max_answer_docs]
    main_ids = {d["id"] for d in top_main}

    diverse_docs = [d for d in retrieved if d["id"] not in main_ids][:max_diverse_docs]

    seen_texts = set()
    answer_chunks = []
    context_docs: List[Dict[str, Any]] = []

    for d in top_main:
        if d["document"] not in seen_texts:
            seen_texts.add(d["document"])
            context_docs.append(d)
            src = d.get("metadata", {}).get("source", "unknown")
            answer_chunks.append(f"[Source: {src}]\n{d['document']}")

    answer_context = "\n\n".join(answer_chunks)
    if len(answer_context) > max_chars:
        answer_context = answer_context[:max_chars]

    diverse_chunks = []
    seen_texts_div = set()
    for d in diverse_docs:
        if d["document"] not in seen_texts_div:
            seen_texts_div.add(d["document"])
            src = d.get("metadata", {}).get("source", "unknown")
            diverse_chunks.append(f"[Source: {src}]\n{d['document']}")

    diverse_context_only = "\n\n".join(diverse_chunks)
    if len(diverse_context_only) > max_chars:
        diverse_context_only = diverse_context_only[:max_chars]

    return answer_context, diverse_context_only, context_docs

# =========================================================
# History Utilities
# =========================================================


def trim_history(history: List[ChatTurn]) -> List[ChatTurn]:
    return history[-10:] if history else []


def summarize_history(history: List[ChatTurn]) -> str:
    if not history:
        return ""
    return " ".join(h.content.strip() for h in history[-5:] if h.content)

# =========================================================
# System Prompts for Answer + Follow-up
# =========================================================

ANSWER_SYSTEM_PROMPT = """
You are a document-grounded assistant specializing in U.S. child-care, adoption, and administrative regulations.

Instructions:
- Answer ONLY using the provided context excerpts.
- Every factual statement should have a citation like [Source: filename].
- If the context does not contain enough information, say so clearly.
- Do NOT invent policies, laws, or numbers.
- Keep the answer concise, around 400 characters, but complete enough to be useful.
- Use short paragraphs or bullets when helpful.

Return ONLY the answer text. Do not include JSON or any extra keys.
"""

FOLLOWUP_SYSTEM_PROMPT = """
You are helping generate a follow-up question after an answer has already been given.

You are given:
- The user's original question
- The assistant's answer
- Additional diverse context passages from the documents
- A brief summary of recent conversation

Your task:
- Propose exactly ONE natural, conversational follow-up question.
- It must:
  - Be relevant to both the user's question and the assistant's answer.
  - Use the diverse context if it hints at important related topics not fully covered yet.
  - Encourage deeper exploration of the same regulation, process, or related practical detail.
  - Avoid trivial questions like "Do you want more help?" or repeating the same question.

Return ONLY valid JSON of the form:
{
  "next_question": "<follow-up question here>"
}
"""

# =========================================================
# LLM Calls (Answer + Follow-up)
# =========================================================


def generate_answer(model: str, question: str, context_text: str, history: List[ChatTurn]) -> str:
    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]

    for h in history:
        messages.append({"role": h.role, "content": h.content})

    summary = summarize_history(history)

    user_content = (
        f"User question:\n{question}\n\n"
        f"Context excerpts:\n{context_text}\n\n"
        f"Recent conversation summary:\n{summary}\n\n"
        "Now produce the answer text only."
    )

    messages.append({"role": "user", "content": user_content})

    resp = oai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def generate_followup(
    model: str,
    question: str,
    answer: str,
    diverse_context: str,
    history: List[ChatTurn],
) -> str:
    messages = [{"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT}]

    for h in history:
        messages.append({"role": h.role, "content": h.content})

    summary = summarize_history(history)

    user_content = (
        f"Original user question:\n{question}\n\n"
        f"Assistant's answer:\n{answer}\n\n"
        f"Additional diverse context from related documents:\n{diverse_context}\n\n"
        f"Recent conversation summary:\n{summary}\n\n"
        "Now produce exactly one useful follow-up question in JSON as specified."
    )

    messages.append({"role": "user", "content": user_content})

    try:
        resp = oai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        next_q = parsed.get("next_question", "").strip()
        if not next_q:
            raise ValueError("Empty next_question")
        return next_q
    except Exception as e:
        print(f"[FOLLOWUP] Failed to parse JSON follow-up, using fallback: {e}")
        return "Would you like to explore eligibility details, application steps, or timelines related to this topic?"

# =========================================================
# Routes
# =========================================================


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.time()

    collection = pick_collection(req.state)
    retrieved = retrieve(req.question, collection, top_k=req.top_k)

    reranked = llm_rerank(req.question, retrieved, rerank_k=req.rerank_k)

    answer_context, diverse_context, context_docs = build_answer_and_diverse_context(
        reranked, retrieved
    )

    trimmed_history = trim_history(req.history or [])

    answer_text = generate_answer(
        model=OPENAI_MODEL,
        question=req.question,
        context_text=answer_context,
        history=trimmed_history,
    )

    next_question = generate_followup(
        model=OPENAI_MODEL,
        question=req.question,
        answer=answer_text,
        diverse_context=diverse_context,
        history=trimmed_history,
    )

    elapsed_ms = round((time.time() - start) * 1000)
    print(
        f"[chat] state={req.state} retrieved={len(retrieved)} "
        f"reranked={len(reranked)} time_ms={elapsed_ms}"
    )

    # Debug: show top few reranked docs
    for i, r in enumerate(reranked[:3]):
        src = r["metadata"].get("source", "unknown")
        dist = r.get("distance", 0)
        print(f"[DEBUG] Top{i+1} src={src} dist={dist}")

    return ChatResponse(
        answer=answer_text,
        next_question=next_question,
        sources=context_docs,
    )
