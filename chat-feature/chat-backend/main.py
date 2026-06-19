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
import re  # for follow-up extraction helpers
# from frontend.db.database import SessionLocal
# from frontend.db.models import State
import os
from sqlalchemy import create_engine, text

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

def normalize_state_name_for_collection(name: str) -> str:
    """
    Convert DB state/category names into a Chroma collection name.
    Example:
      'Student Records & Enrollment Data'
      -> 'student_records_enrollment_data'
    """
    name = name.strip().lower()
    name = re.sub(r"&", " and ", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name
def load_state_collection_mapping_from_db() -> Dict[str, str]:
    """
    Fallback: read states from Postgres directly.
    Avoid importing database/models packages from another service.
    """

    database_url = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:admin123@postgres:5432/childcare",
    )
    try:
        engine = create_engine(database_url, future=True)
        mapping: Dict[str, str] = {}

        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM states"))
            for row in result:
                raw_name = row[0]
                state_name = (raw_name or "").strip().lower()
                if state_name:
                    mapping[state_name] = normalize_state_name_for_collection(raw_name)

        print(f"Loaded states mapping from DB: {mapping}")
        return mapping

    except Exception as e:
        print(f"Failed to load states from DB: {e}")
        return {}
# def load_state_collection_mapping_from_db() -> Dict[str, str]:
#     """
#     Fallback: read states from DB seeded by db.py.
#     Assumes collection name can be derived from state/category name.
#     """
#     try:

#         db = SessionLocal()
#         try:
#             states = db.query(State).all()
#             mapping: Dict[str, str] = {}

#             for st in states:
#                 state_name = (st.name or "").strip().lower()
#                 if state_name:
#                     mapping[state_name] = normalize_state_name_for_collection(st.name)

#             print(f"Loaded states mapping from DB: {mapping}")
#             return mapping
#         finally:
#             db.close()

#     except Exception as e:
#         print(f"Failed to load states from DB: {e}")
#         return {}

def load_state_collection_mapping():
    """
    Load state/category -> Chroma collection mapping.
    Priority:
      1. config.json
      2. DB seeded states
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)

            mapping: Dict[str, str] = {}
            for st in cfg.get("states", []):
                name = st.get("name", "").strip().lower()
                cname = st.get("collection_name")
                if name and cname:
                    mapping[name] = normalize_state_name_for_collection(name)

            if mapping:
                print(f"Loaded states mapping from config.json: {mapping}")
                return mapping
        except Exception as e:
            print(f"Failed reading config.json: {e}")

    print("config.json missing/empty; falling back to DB states")
    db_mapping = load_state_collection_mapping_from_db()
    if db_mapping:
        return db_mapping

    print("No states found in DB either; using empty mapping")
    return {
            "california": "california_state",
            "new york": "newyork_state",
        }

# def load_state_collection_mapping() -> Dict[str, str]:
#     """Load state → Chroma collection mapping from config.json (lowercased keys)."""
#     if not CONFIG_FILE.exists():
#         print("config.json not found; using fallback CA/NY mapping")
#         return {
#             "california": "california_state",
#             "new york": "newyork_state",
#         }

#     with open(CONFIG_FILE, "r") as f:
#         cfg = json.load(f)

#     mapping: Dict[str, str] = {}
#     for st in cfg.get("states", []):
#         name = st.get("name", "").strip().lower()
#         cname = st.get("collection_name")
#         if name and cname:
#             mapping[name] = cname

#     print(f"Loaded states mapping: {mapping}")
#     return mapping


STATE_TO_COLLECTION = load_state_collection_mapping()


# =========================================================
# Initialize Clients (Chroma + Embeddings + OpenAI LLM)
# =========================================================


def connect_chroma(retries: int = 12, delay: float = 2.0):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            print(f"Connected to ChromaDB on attempt {attempt}")
            return client
        except Exception as e:
            last_err = e
            print(f"Waiting for ChromaDB (attempt {attempt}/{retries})... {e}")
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
    # Defaults chosen so you naturally get 5 (answer) + 15 (follow-up pool)
    top_k: Optional[int] = 40       # vector DB n_results
    rerank_k: Optional[int] = 20    # how many to send into LLM reranker
    lookup_online: Optional[bool] = False


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
    if not STATE_TO_COLLECTION:
        raise ValueError("No states/categories are configured in backend.")

    if key not in STATE_TO_COLLECTION:
        raise ValueError(
            f"Unsupported state '{state}'. Available: {list(STATE_TO_COLLECTION.keys())}"
        )
    name = STATE_TO_COLLECTION[key]
    return chroma_client.get_or_create_collection(name=name)


def retrieve(query: str, collection, top_k: int):
    """Initial vector retrieval from Chroma."""
    print(f"[RETRIEVE] query='{query}' top_k={top_k}")
    q_vec = embedder.encode(query).tolist()
    print("Q vector = ",q_vec)
    res = collection.query(query_embeddings=[q_vec], n_results=top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

    items = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        src = (meta or {}).get("source", "unknown")
        print(f"[RETRIEVE] #{i+1} id={ids[i]} source={src} dist={dists[i]}")
        items.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": meta,
                "distance": dists[i] if dists and i < len(dists) else None,
            }
        )
    print(f"[RETRIEVE] total_items={len(items)}")
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
        print("[RERANK] no items to rerank")
        return []

    MAX_CANDIDATES = min(len(items), rerank_k)
    candidates = items[:MAX_CANDIDATES]
    print(f"[RERANK] got {len(items)} items, using {MAX_CANDIDATES} as candidates")

    # Build the chunk list text
    chunk_lines = []
    for idx, it in enumerate(candidates, start=1):
        cid = it["id"]
        text = it["document"].replace("\n", " ")
        # if len(text) > 700:
        #     text = text[:700] + "..."
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
        print(f"[RERANK] raw LLM output: {raw[:400]}...")
        parsed = json.loads(raw)
        ranked_ids = parsed.get("ranked_ids", [])
        if not isinstance(ranked_ids, list):
            raise ValueError("ranked_ids is not a list")
    except Exception as e:
        print(f"[RERANK] LLM rerank failed, falling back to original order: {e}")
        return items[: min(rerank_k, len(items))]

    # Map ids → item
    item_by_id = {it["id"]: it for it in candidates}
    ordered: List[Dict[str, Any]] = []

    for cid in ranked_ids:
        if cid in item_by_id:
            ordered.append(item_by_id[cid])
        else:
            print(f"[RERANK] WARNING: ranked_id {cid} not in candidates")

    if not ordered:
        print("[RERANK] no valid ids in ranked_ids, falling back to candidates")
        ordered = candidates

    print(f"[RERANK] final ordered count={len(ordered)}")
    for i, it in enumerate(ordered[:5]):
        src = (it.get("metadata") or {}).get("source", "unknown")
        print(f"[RERANK] Top{i+1} id={it['id']} source={src} dist={it.get('distance')}")

    return ordered[: min(rerank_k, len(ordered))]


# =========================================================
# Context Building (answer-only)
# =========================================================


def build_answer_context(
    docs: List[Dict[str, Any]],
    max_chars: int = 12000,
):
    """
    Build:
      - answer_context: main docs for answering
      - context_docs: unique docs used as 'sources'
    """
    seen_texts = set()
    answer_chunks = []
    context_docs: List[Dict[str, Any]] = []

    for d in docs:
        if d["document"] not in seen_texts:
            seen_texts.add(d["document"])
            context_docs.append(d)
            src = d.get("metadata", {}).get("source", "unknown")
            answer_chunks.append(f"[Source: {src}]\n{d['document']}")

    answer_context = "\n\n".join(answer_chunks)
    if len(answer_context) > max_chars:
        answer_context = answer_context[:max_chars]

    print(f"[CONTEXT] built answer_context chars={len(answer_context)} "
          f"unique_docs={len(context_docs)}")
    return answer_context, context_docs


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
- Use short paragraphs or bullets when helpful.
- When the document contains a general policy and a special-case policy, and the question refers to the special case, answer ONLY using the special-case rule.

STRICT GROUNDEDNESS RULES:
- You MUST answer ONLY using the provided context excerpts.
- You MUST cite every factual statement using the exact filename(s) found in the context.
- If a question REQUIRES information from more than one document, you MUST combine the relevant excerpts and produce a unified answer.
- If a required document is NOT present in the context chunks, you MUST respond:
  “The provided context does not include enough information to answer this completely.”

NO HALLUCINATION RULES:
- Do NOT add information not explicitly stated in the retrieved context.
- Do NOT infer requirements, policies, or procedures not directly included in the context.
- If the context is missing something, state that it is missing.

SPECIAL-CASE PRIORITY RULE:
- When the documents contain both a general rule and a special-case rule, and the question refers to the special case, answer ONLY using the special-case rule.

TABLE RULE:
- If the information in the context is clearly structured (such as repeated fields, numeric lists, ranges, income tables, thresholds, or row-like patterns), you MUST present it using a clean HTML <table>.
- Do NOT use Markdown tables.
- Do NOT wrap the HTML in code blocks.
- Use <table>, <tr>, <th>, and <td> tags.
- Always include headers if identifiable from context.
- If the context contains only one row or cannot logically form a table, use normal text.

CITATION RULE:
- Every sentence containing factual information must include a citation in this form:
  [Source: filename]

MULTI-DOCUMENT REQUIREMENT:
- If the question mentions two concepts which are known to belong to different documents, the answer MUST cite BOTH documents OR state clearly that one is missing.

FORMATTING:
- Answers must be factual and complete.
- Bullets or short paragraphs are allowed.
- HTML tables must render cleanly.

Return ONLY the answer text (which may contain HTML). Do not include JSON or any extra keys.
"""

FOLLOWUP_SYSTEM_PROMPT = """
You generate a follow-up question ONLY if it can be fully answered using the SAME retrieved context.

You will be given:
- The original user question
- The assistant's answer
- A small set of candidate snippets taken from OTHER relevant chunks
  that the user has NOT yet seen.

Your job is to create a SEMANTIC BRIDGE:
- The follow-up must introduce NEW information from the candidate snippets.
- It must be clearly related to the topic of the assistant's answer.
- It should feel like a natural "next step" question a helpful case worker would ask.

STRICT RESTRICTIONS:
1. DO NOT ask about anything already covered in the assistant's answer.
2. DO NOT ask about unrelated content even if it appears in the snippets.
3. DO NOT infer or invent; only use what appears in the snippets.
4. If no meaningful, grounded follow-up exists, return an empty string.

TONE REQUIREMENT:
- The follow-up question MUST begin with one of:
  - "Do you want"
  - "Do you need"
  - "Would you like"

FORMAT:
Return ONLY valid JSON in this exact shape:
{
  "next_question": "<question or empty string>"
}
"""


# =========================================================
# LLM Calls (Answer + Follow-up)
# =========================================================

ONLINE_SYSTEM_PROMPT = """
You are a helpful assistant.

- Answer the user's question directly.
- If you are unsure, say so.
- Keep it concise but complete.
"""

def generate_online_answer(model: str, question: str, history: List[ChatTurn]) -> str:
    messages = [{"role": "system", "content": ONLINE_SYSTEM_PROMPT}]

    for h in history:
        messages.append({"role": h.role, "content": h.content})

    messages.append({"role": "user", "content": question})

    resp = oai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

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
        temperature=0.1,
    )
    answer = resp.choices[0].message.content.strip()
    print(f"[ANSWER] answer_text (first 400 chars): {answer[:400]}")
    return answer


def should_generate_followup(answer_text: str) -> bool:
    """
    Suppress follow-up questions when the answer indicates:
    - insufficient information,
    - missing context,
    - uncertainty,
    - inability to answer,
    - or references to external sources.
    This is domain-agnostic and works for ANY topic.
    """
    txt = answer_text.lower()

    negative_patterns = [
        # Signals missing/insufficient context
        "context does not provide",
        "context does not contain",
        "context does not include",
        "no information available",
        "not enough information",
        "insufficient information",
        "cannot determine from the context",
        "nothing in the context",
        "not mentioned in the context",

        # Signals uncertainty or inability to answer
        "cannot answer",
        "cannot provide an answer",
        "unable to answer",
        "i am not sure",
        "uncertain",
        "unknown",

        # Signals redirection to external sources (domain-agnostic)
        "refer to the relevant",
        "refer to the appropriate",
        "contact the appropriate",
        "consult the appropriate",
        "check with your",
        "contact your",
        "refer to official",
        "refer to external documentation",
        "outside the scope",
        "requires external",
        "requires additional sources",

        # Strong indicator answer is incomplete or speculative
        "i would be speculating",
        "cannot confirm",
        "cannot verify",
    ]

    suppressed = any(p in txt for p in negative_patterns)
    print(f"[FOLLOWUP] should_generate_followup={not suppressed}")
    if suppressed:
        print("[FOLLOWUP] suppressed due to negative pattern match")
    return not suppressed


def generate_followup(
    model: str,
    question: str,
    answer: str,
    candidate_snippets: str,
    history: List[ChatTurn],
) -> str:
    messages = [{"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT}]

    for h in history:
        messages.append({"role": h.role, "content": h.content})

    summary = summarize_history(history)

    user_content = (
        f"Original user question:\n{question}\n\n"
        f"Assistant's answer:\n{answer}\n\n"
        f"Candidate follow-up snippets (from other chunks):\n{candidate_snippets}\n\n"
        f"Recent conversation summary:\n{summary}\n\n"
        "Now decide whether there is a grounded next question. "
        "If yes, return it in JSON as specified. If not, return an empty string in JSON."
    )

    messages.append({"role": "user", "content": user_content})

    try:
        resp = oai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[FOLLOWUP] raw LLM output: {raw[:400]}")
        parsed = json.loads(raw)
        next_q = parsed.get("next_question", "")
        if not isinstance(next_q, str):
            raise ValueError("next_question is not a string")
        next_q = next_q.strip()
        print(f"[FOLLOWUP] next_question='{next_q}'")
        return next_q
    except Exception as e:
        print(f"[FOLLOWUP] Failed to parse JSON follow-up, using empty follow-up: {e}")
        return ""


# =========================================================
# Follow-up Extraction Helpers (Pool-Based Logic)
# =========================================================


def extract_key_points(text: str) -> List[str]:
    """
    Extract bullet points, numbered items, or substantial lines from a chunk.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bullets: List[str] = []
    for l in lines:
        # bullet or dash or en dash + space
        if re.match(r"^[•\-–]\s+", l):
            bullets.append(l)
        # numbered list like "1. " or "2) "
        elif re.match(r"^\d+[.)]\s+", l):
            bullets.append(l)
        else:
            # fallback: longer lines treated as key points
            if len(l.split()) > 6:
                bullets.append(l)
    return bullets


# =========================================================
# Sources UI Helper
# =========================================================


def build_sources_ui(context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a UI-friendly sources structure:

    [
      {
        "source": "<filename>",
        "chunks": [
          {
            "id": "<chunk id>",
            "chunk_index": <int or None>,
            "snippet": "<first line / first few words>"
          },
          ...
        ]
      },
      ...
    ]
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for d in context_docs:
        meta = d.get("metadata", {}) or {}
        filename = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index")
        chunk_id = d.get("id")

        text = d.get("document", "") or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            snippet = lines[0]
        else:
            snippet = text[:160]

        if len(snippet) > 160:
            snippet = snippet[:157] + "..."

        if filename not in grouped:
            grouped[filename] = {
                "source": filename,
                "chunks": [],
            }

        # Avoid duplicate chunk ids
        existing_ids = {c.get("id") for c in grouped[filename]["chunks"]}
        if chunk_id not in existing_ids:
            grouped[filename]["chunks"].append(
                {
                    "id": chunk_id,
                    "chunk_index": chunk_index,
                    "snippet": snippet,
                }
            )

    # Optionally sort chunks by chunk_index when available
    for g in grouped.values():
        g["chunks"].sort(key=lambda c: (c["chunk_index"] is None, c["chunk_index"]))

    return list(grouped.values())


# =========================================================
# Routes
# =========================================================


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/states")
def get_states():
    return {
        "states": [
            {"label": state_name, "value": state_name}
            for state_name in STATE_TO_COLLECTION.keys()
        ]
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.time()

    print("\n==================== /chat REQUEST ====================")
    print(f"[REQ] question='{req.question}'")
    print(f"[REQ] state='{req.state}' top_k={req.top_k} rerank_k={req.rerank_k}")

    trimmed_history = trim_history(req.history or [])
    if req.lookup_online:
        print("[MODE] lookup_online=True → skipping Chroma/RAG pipeline")
        answer_text = generate_online_answer(
            model=OPENAI_MODEL,
            question=req.question,
            history=trimmed_history,
        )
        return ChatResponse(
            answer=answer_text,
            next_question="",
            sources=[],
        )

    collection = pick_collection(req.state)
    retrieved = retrieve(req.question, collection, top_k=req.top_k)

    # Rerank up to rerank_k
    reranked = llm_rerank(req.question, retrieved, rerank_k=req.rerank_k)
    print(f"[PIPELINE] after rerank: count={len(reranked)}")

    # =========================================================
    # SPLIT RERANKED INTO: top_5 (answer) + remaining (follow-up pool)
    # =========================================================
    ANSWER_TOP_N = 5   # always use top 5 for answer
    answer_docs = reranked[:ANSWER_TOP_N]
    followup_pool_docs = retrieved[ANSWER_TOP_N : ANSWER_TOP_N + 15]

    print(f"[SPLIT] answer_docs={len(answer_docs)} followup_pool_docs={len(followup_pool_docs)}")

    for i, d in enumerate(answer_docs):
        src = (d.get("metadata") or {}).get("source", "unknown")
        print(f"[SPLIT] ANSWER_DOC #{i+1} id={d['id']} source={src}")

    for i, d in enumerate(followup_pool_docs):
        src = (d.get("metadata") or {}).get("source", "unknown")
        print(f"[SPLIT] FOLLOWUP_POOL_DOC #{i+1} id={d['id']} source={src}")

    # Build answer context only from top 5
    answer_context, context_docs = build_answer_context(answer_docs)

    # trimmed_history = trim_history(req.history or [])

    # 1) Generate grounded answer
    answer_text = generate_answer(
        model=OPENAI_MODEL,
        question=req.question,
        context_text=answer_context,
        history=trimmed_history,
    )

    # =========================================================
    # FOLLOW-UP EXTRACTION (from followup_pool_docs only)
    # =========================================================
    followup_candidates: List[str] = []
    if should_generate_followup(answer_text) and followup_pool_docs:
        pool_items: List[str] = []
        for d in followup_pool_docs:
            src = (d.get("metadata") or {}).get("source", "unknown")
            items = extract_key_points(d["document"])
            print(f"[FOLLOWUP] from doc id={d['id']} source={src} extracted_key_points={len(items)}")
            pool_items.extend(items)

        # Deduplicate
        before_dedup = len(pool_items)
        pool_items = list(dict.fromkeys(pool_items))
        print(f"[FOLLOWUP] pool_items before_dedup={before_dedup} after_dedup={len(pool_items)}")

        # Remove anything overlapping with the answer
        ans_words = set(answer_text.lower().split())
        filtered_items: List[str] = []
        for item in pool_items:
            item_words = set(item.lower().split())
            overlap = len(ans_words & item_words)
            if overlap == 0:
                filtered_items.append(item)

        print(f"[FOLLOWUP] filtered_items count (no overlap with answer)={len(filtered_items)}")

        # Select the best semantic bridge (match to the original question)
        best_item = ""
        best_score = 0
        q_words = set(req.question.lower().split())

        for item in filtered_items:
            item_words = set(item.lower().split())
            score = len(q_words & item_words)
            if score > best_score:
                best_score = score
                best_item = item

        print(f"[FOLLOWUP] best_score={best_score}")

        if best_item:
            # Normal case: lexical match exists
            print(f"[FOLLOWUP] best_item='{best_item}'")
            followup_candidates = [best_item]
        else:
            # FALLBACK: still allow LLM to generate a grounded follow-up
            # by giving it the first 3 filtered items
            print("[FOLLOWUP] no suitable lexical match → using fallback items")
            followup_candidates = filtered_items[:3]

    else:
        print("[FOLLOWUP] skipped: either suppression or empty followup pool")

    followup_context_text = ""
    next_question = ""

    if followup_candidates:
        followup_context_text = "\n".join(followup_candidates[:6])
        print("\n[FOLLOWUP CONTEXT]\n", followup_context_text)

        next_question = generate_followup(
            model=OPENAI_MODEL,
            question=req.question,
            answer=answer_text,
            candidate_snippets=followup_context_text,
            history=trimmed_history,
        )
    else:
        print("[FOLLOWUP] no followup_candidates → next_question will be empty")

    elapsed_ms = round((time.time() - start) * 1000)
    print(
        f"[chat] state={req.state} retrieved={len(retrieved)} "
        f"reranked={len(reranked)} time_ms={elapsed_ms}"
    )

    # Debug: show top few reranked docs
    for i, r in enumerate(reranked[:3]):
        src = (r.get("metadata") or {}).get("source", "unknown")
        dist = r.get("distance", 0)
        print(f"[DEBUG] Top{i+1} src={src} dist={dist}")

    # Build UI-friendly sources (from the chunks actually used for the answer)
    structured_sources = build_sources_ui(context_docs)

    print("==================== /chat END ====================\n")

    return ChatResponse(
        answer=answer_text,
        next_question=next_question,
        sources=structured_sources,
    )
