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
# Context Building (answer-only, no "diverse" context)
# =========================================================


def build_answer_context(
    reranked: List[Dict[str, Any]],
    max_answer_docs: int = 5,
    max_chars: int = 12000,
):
    """
    Build:
      - answer_context: main docs for answering (top reranked)
      - context_docs: unique docs used as 'sources'
    """
    top_main = reranked[:max_answer_docs]

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

TABLE RULE:
- If the information in the context is clearly structured (such as repeated fields, numeric lists, ranges, income tables, thresholds, or row-like patterns), you MUST present it using a clean HTML <table>.
- Do NOT use Markdown tables.
- Do NOT wrap the HTML in code blocks.
- Use <table>, <tr>, <th>, and <td> tags.
- Always include headers if identifiable from context.
- If the context contains only one row or cannot logically form a table, use normal text.

FORMATTING:
- Answers must be concise but complete.
- Bullets or short paragraphs are allowed.
- HTML tables must render cleanly.

Return ONLY the answer text (which may contain HTML). Do not include JSON or any extra keys.
"""

FOLLOWUP_SYSTEM_PROMPT = """
You generate a follow-up question ONLY if it can be fully answered using the SAME context excerpts.

You must also use adjacency logic:
- Look at the specific chunk(s) that contain the answer.
- Identify the NEXT detail, requirement, step, rule, exception, or field
  that appears in the same paragraph or bullet list, or in the immediately
  adjacent chunks in the document (by chunk_index).
- Formulate a follow-up question that asks about that next detail.

TONE REQUIREMENT:
- The follow-up question MUST start with one of:
  - "Do you want"
  - "Do you need"
  - "Would you like"

STRICT RULES:
1. Only propose a follow-up if it is fully grounded in the provided context.
2. The follow-up must refer to information that appears CLOSE to the part used
   in the answer (same bullet group, same section, or next listed requirement).
3. Do NOT ask anything the context does not contain.
4. Do NOT generalize, guess, infer, or broaden.
5. If no meaningful adjacent detail exists, return an empty string.

FORMAT:
Return ONLY JSON:
{
  "next_question": "<question or empty string>"
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

    return not any(p in txt for p in negative_patterns)


def generate_followup(
    model: str,
    question: str,
    answer: str,
    context_text: str,
    history: List[ChatTurn],
) -> str:
    messages = [{"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT}]

    for h in history:
        messages.append({"role": h.role, "content": h.content})

    summary = summarize_history(history)

    user_content = (
        f"Original user question:\n{question}\n\n"
        f"Assistant's answer:\n{answer}\n\n"
        f"Context excerpts used for the answer:\n{context_text}\n\n"
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
        parsed = json.loads(raw)
        next_q = parsed.get("next_question", "")
        if not isinstance(next_q, str):
            raise ValueError("next_question is not a string")
        next_q = next_q.strip()
        return next_q
    except Exception as e:
        print(f"[FOLLOWUP] Failed to parse JSON follow-up, using empty follow-up: {e}")
        return ""

# =========================================================
# Follow-up Extraction Helpers (Adjacent-Context Logic)
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


def match_answered_items(answer: str, items: List[str]) -> List[str]:
    """
    Determine which extracted items were actually used in the answer.
    Uses simple token overlap to remain deterministic & safe.
    """
    used: List[str] = []
    ans = answer.lower()
    for item in items:
        words = item.lower().split()
        overlap = sum(1 for w in words if w in ans)
        if overlap >= max(3, int(len(words) * 0.3)):
            used.append(item)
    return used


def select_adjacent_items(items: List[str], used_items: List[str]) -> List[str]:
    """
    Select neighbors of used items within the SAME chunk (by bullet order).
    This is intra-chunk adjacency, independent of chunk_index.
    """
    if not used_items:
        return []

    neighbors: List[str] = []

    used_set = set(used_items)
    used_indexes = [idx for idx, val in enumerate(items) if val in used_set]

    for idx in used_indexes:
        for n in (idx - 1, idx + 1):
            if 0 <= n < len(items):
                if items[n] not in used_items:
                    neighbors.append(items[n])

    # de-duplicate while preserving order
    return list(dict.fromkeys(neighbors))


def select_adjacent_chunks(
    reranked: List[Dict[str, Any]],
    used_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Select document-adjacent chunks based on metadata['chunk_index'].
    For each used chunk, we look at chunk_index - 1 and + 1
    among the reranked results.
    """
    if not used_chunks:
        return []

    used_indexes: List[int] = []
    for c in used_chunks:
        meta = c.get("metadata", {})
        if "chunk_index" in meta and isinstance(meta["chunk_index"], int):
            used_indexes.append(meta["chunk_index"])

    if not used_indexes:
        return []

    # Build lookup: chunk_index -> chunk
    index_lookup: Dict[int, Dict[str, Any]] = {}
    for c in reranked:
        meta = c.get("metadata", {})
        idx = meta.get("chunk_index")
        if isinstance(idx, int):
            index_lookup[idx] = c

    neighbors: List[Dict[str, Any]] = []
    for idx in used_indexes:
        for neighbor_idx in (idx - 1, idx + 1):
            if neighbor_idx in index_lookup:
                neighbors.append(index_lookup[neighbor_idx])

    # Deduplicate by id while preserving order
    seen_ids = set()
    result: List[Dict[str, Any]] = []
    for c in neighbors:
        cid = c.get("id")
        if cid and cid not in seen_ids:
            result.append(c)
            seen_ids.add(cid)

    return result

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

    This lets the frontend group by file and show collapsible chunks/snippets
    instead of listing the same filename repeatedly.
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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.time()

    collection = pick_collection(req.state)
    retrieved = retrieve(req.question, collection, top_k=req.top_k)

    reranked = llm_rerank(req.question, retrieved, rerank_k=req.rerank_k)

    # Build answer context from reranked docs
    answer_context, context_docs = build_answer_context(reranked)

    trimmed_history = trim_history(req.history or [])

    # 1) Generate grounded answer
    answer_text = generate_answer(
        model=OPENAI_MODEL,
        question=req.question,
        context_text=answer_context,
        history=trimmed_history,
    )

    # 2) Optionally generate grounded follow-up (chunk_index + intra-chunk adjacency)
    followup_candidates: List[str] = []

    # Keep info per chunk to support fallback later
    per_chunk_info: List[Dict[str, Any]] = []
    used_chunks: List[Dict[str, Any]] = []

    # First: same-chunk adjacency based on bullet-level usage
    for d in context_docs:
        chunk_text = d["document"]
        items = extract_key_points(chunk_text)
        used_items = match_answered_items(answer_text, items)

        per_chunk_info.append(
            {
                "chunk": d,
                "items": items,
                "used_items": used_items,
            }
        )

        neighbors_same = select_adjacent_items(items, used_items)
        followup_candidates.extend(neighbors_same)

        if used_items:
            used_chunks.append(d)

    # Second: adjacent chunks based on chunk_index
    adjacent_chunks = select_adjacent_chunks(reranked, used_chunks)
    for ch in adjacent_chunks:
        items = extract_key_points(ch["document"])
        per_chunk_info.append(
            {
                "chunk": ch,
                "items": items,
                "used_items": [],
            }
        )
        followup_candidates.extend(items)

    # Deduplicate early (preserve order)
    followup_candidates = list(dict.fromkeys(followup_candidates))

    # =====================================================
    # Fallback follow-up logic:
    # If neighbors are empty, fall back to unused items from
    # chunks that contributed to the answer, then any items.
    # =====================================================
    if should_generate_followup(answer_text):
        if not followup_candidates:
            # 1) Try unused items from chunks where some items were used
            fallback_items: List[str] = []
            for info in per_chunk_info:
                items = info["items"]
                used_items = set(info["used_items"] or [])
                if items and used_items and len(used_items) < len(items):
                    unused = [i for i in items if i not in used_items]
                    fallback_items.extend(unused)

            # 2) If still empty, consider any items from context chunks
            if not fallback_items:
                for info in per_chunk_info:
                    items = info["items"]
                    if items:
                        fallback_items.extend(items)

            # Deduplicate
            fallback_items = list(dict.fromkeys(fallback_items))
            followup_candidates = fallback_items

    # Build the follow-up context text from candidates
    followup_context_text = ""
    next_question = ""

    if should_generate_followup(answer_text) and followup_candidates:
        followup_context_text = "\n".join(followup_candidates[:6])
        print("\n[FOLLOWUP CONTEXT]\n", followup_context_text)

        next_question = generate_followup(
            model=OPENAI_MODEL,
            question=req.question,
            answer=answer_text,
            context_text=followup_context_text,
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

    # Build UI-friendly sources
    structured_sources = build_sources_ui(context_docs)

    

    return ChatResponse(
        answer=answer_text,
        next_question=next_question,
        sources=structured_sources,
    )