from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import shutil
import json
from pathlib import Path
import time
import traceback
import re

from openai import OpenAI
from dotenv import load_dotenv

# =========================================================
# Base Paths & Environment
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"Loaded .env from {ENV_PATH}")
else:
    print(f".env not found at {ENV_PATH}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
oai = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_LLM_MODEL = "gpt-4o-mini"

# =========================================================
# FastAPI App
# =========================================================

app = FastAPI()

# =========================================================
# Config Loader
# =========================================================

CONFIG_FILE = Path("/app/config.json")
state_config: Dict[str, str] = {}
state_collections: Dict[str, Any] = {}


def load_config():
    """
    Load state -> collection_name mapping from config.json.
    Falls back to California/New York if file missing or invalid.
    """
    global state_config, state_collections
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                state_config = {
                    st["name"]: st["collection_name"]
                    for st in cfg.get("states", [])
                    if "name" in st and "collection_name" in st
                }
            print(f"[CONFIG] Loaded states: {list(state_config.keys())}", flush=True)
        else:
            state_config = {
                "California": "california_state",
                "New York": "newyork_state",
            }
            print("[CONFIG] Using default states (config.json not found)", flush=True)
    except Exception as e:
        print(f"[CONFIG ERROR] {e}", flush=True)
        state_config = {
            "California": "california_state",
            "New York": "newyork_state",
        }


load_config()

# =========================================================
# Storage Paths
# =========================================================

DATA_DIR = Path("/app/Data")
DATA_DIR.mkdir(exist_ok=True)

# =========================================================
# ChromaDB Connection
# =========================================================

try:
    for attempt in range(10):
        chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
        if chroma_client.heartbeat():
            print("✅ Connected to ChromaDB!")
            break
        print("Waiting for ChromaDB...")
        time.sleep(3)
except Exception as e:
    raise RuntimeError("❌ Could not connect to ChromaDB.") from e

# Create state collections from config
for state_name, collection_name in state_config.items():
    try:
        state_collections[state_name] = chroma_client.get_or_create_collection(
            name=collection_name
        )
        print(f"[COLLECTION] Ready: {collection_name}")
    except Exception as e:
        print(f"[COLLECTION ERROR] {e}", flush=True)

# Optional QA collection (unchanged)
qa_collection_california = chroma_client.get_or_create_collection(name="qa_pairs")

# =========================================================
# Embedding model & splitter
# =========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# =========================================================
# Request Models
# =========================================================


class QuestionRequest(BaseModel):
    question: str
    state: str = "California"


class SavePairRequest(BaseModel):
    question: str
    sql: str


class VectorizeRequest(BaseModel):
    state: str
    file_paths: List[str]


# =========================================================
# Category Labels (Used for Optional Chunk Classification)
# =========================================================

CATEGORY_LABELS = [
    "definition",
    "requirement",
    "procedure",
    "statistic",
    "benefit",
    "challenge",
    "feature",
    "comparison",
]

# =========================================================
# Heuristic Category Scorer
# =========================================================


def heuristic_category_scores(text: str) -> Dict[str, int]:
    """
    Very lightweight pattern-based classifier to tag chunks with coarse labels.
    """
    t = text.lower()
    scores = {cat: 0 for cat in CATEGORY_LABELS}

    definition_kw = ["is defined as", "refers to", "means", "defined as"]
    statistic_kw = [
        "percent",
        "percentage",
        "ratio",
        "survey",
        "data",
        "average",
        "median",
        "mean",
    ]
    requirement_kw = ["must", "shall", "required", "mandatory", "eligibility"]
    procedure_kw = ["steps", "step", "process", "procedure", "how to"]
    benefit_kw = ["benefit", "advantage", "improves", "helps", "reduces", "supports"]
    challenge_kw = ["challenge", "difficulty", "problem", "issue", "barrier", "concern"]
    feature_kw = ["feature", "includes", "consists of", "characteristic"]
    comparison_kw = ["versus", "vs", "compared to", "difference", "in contrast"]

    for kw in definition_kw:
        if kw in t:
            scores["definition"] += 2

    for kw in statistic_kw:
        if kw in t:
            scores["statistic"] += 1

    for kw in requirement_kw:
        if kw in t:
            scores["requirement"] += 1

    for kw in procedure_kw:
        if kw in t:
            scores["procedure"] += 1

    for kw in benefit_kw:
        if kw in t:
            scores["benefit"] += 1

    for kw in challenge_kw:
        if kw in t:
            scores["challenge"] += 1

    for kw in feature_kw:
        if kw in t:
            scores["feature"] += 1

    for kw in comparison_kw:
        if kw in t:
            scores["comparison"] += 1

    return scores


def normalize_scores(scores: Dict[str, int]) -> Dict[str, float]:
    max_score = max(scores.values(), default=0)
    if max_score == 0:
        return {}
    return {k: round(v / max_score, 3) for k, v in scores.items() if v > 0}


# =========================================================
# LLM Fallback Classifier (Optional)
# =========================================================


def llm_category_fallback(text: str) -> Dict[str, float]:
    """
    If heuristic classification is too weak, optionally ask the LLM to tag the text.
    Returns {category: 1.0, ...} style scores for recognized labels.
    """
    if not OPENAI_API_KEY:
        return {}

    try:
        labels_str = ", ".join(CATEGORY_LABELS)
        prompt = (
            "Classify the following text into one or more categories.\n"
            f"Valid categories: {labels_str}.\n"
            "Return ONLY a comma-separated list of category names.\n\n"
            f"Text:\n{text[:4000]}\n"
        )

        resp = oai.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = resp.choices[0].message.content.strip().lower()
        if not raw:
            return {}

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        valid = {p for p in parts if p in CATEGORY_LABELS}

        return {p: 1.0 for p in valid}

    except Exception as e:
        print(f"[OPENAI FALLBACK ERROR] {e}", flush=True)
        return {}


# =========================================================
# Combined Category Helper
# =========================================================


def get_chunk_categories(text: str) -> Dict[str, float]:
    """
    First use heuristic scores; if all zero, try LLM.
    If still nothing, fall back to {"other": 1.0}.
    """
    scores = heuristic_category_scores(text)
    normalized = normalize_scores(scores)

    if normalized:
        return normalized

    llm_result = llm_category_fallback(text)
    if llm_result:
        return llm_result

    return {"other": 1.0}


# =========================================================
# VECTORIZE — FILE-AGNOSTIC INGESTION PIPELINE
# =========================================================

@app.post("/vectorize")
async def vectorize_documents(request: VectorizeRequest):
    """
    Vectorizes uploaded documents for a given state.

    - Uses dynamic state -> collection mapping from config.json.
    - Supports common doc types:
        * .pdf       -> PyPDFLoader
        * .doc/.docx -> UnstructuredWordDocumentLoader
        * .csv       -> CSVLoader
        * everything else -> TextLoader (file-agnostic fallback)
    - Splits into chunks and stores embeddings + metadata in Chroma.

    Metadata schema is aligned with chat-feature backend:
        {
            "source": <filename>,
            "state": <state name>,
            "chunk_index": <int>,
            "file_type": <extension>,
            "categories": <JSON string of category scores>
        }
    """

    try:
        state = request.state
        fpaths = request.file_paths

        # Validate state
        if state not in state_collections:
            return {
                "error": f"Invalid state '{state}'. "
                         f"Available: {list(state_collections.keys())}"
            }

        collection = state_collections[state]

        total_chunks = 0
        processed_files: List[str] = []

        for fp in fpaths:
            p = Path(fp)

            if not p.exists():
                print(f"[VECTORIZE] File missing: {fp}")
                continue

            ext = p.suffix.lower()
            print(f"[VECTORIZE] Loading file: {p.name} (ext={ext})")

            # ----- Choose loader based on extension -----
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(p))
                elif ext in [".doc", ".docx"]:
                    loader = UnstructuredWordDocumentLoader(str(p))
                elif ext == ".csv":
                    loader = CSVLoader(str(p))
                else:
                    # Generic text-based fallback for any other extension (.txt, .md, .rtf, .html, .json, etc.)
                    loader = TextLoader(str(p), encoding="utf-8")

                docs = loader.load()
            except Exception as e:
                print(f"[VECTORIZE] Failed to load {p.name} with loader: {e}")
                continue

            if not docs:
                print(f"[VECTORIZE] No text extracted from {p.name}")
                continue

            # Split into chunks
            chunks = text_splitter.split_documents(docs)
            print(f"[VECTORIZE] {len(chunks)} chunks created from {p.name}")

            for i, chunk in enumerate(chunks):
                # LangChain Document has .page_content
                text = getattr(chunk, "page_content", "").strip()
                if not text:
                    continue

                # Category scoring (optional metadata)
                categories = get_chunk_categories(text)
                categories_json = json.dumps(categories)

                # Embedding
                embedding = model.encode(text).tolist()

                # Unique chunk id (aligned with your existing pattern)
                chunk_id = f"{p.stem}_chunk_{i}_{total_chunks}"

                # Store into Chroma
                collection.add(
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[
                        {
                            "source": p.name,
                            "state": state,
                            "chunk_index": i,
                            "file_type": ext,
                            "categories": categories_json,
                        }
                    ],
                    ids=[chunk_id],
                )

                total_chunks += 1

            processed_files.append(p.name)

        return {
            "message": "Vectorization complete",
            "processed_files": processed_files,
            "total_chunks": total_chunks,
            "collection_count": collection.count(),
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =========================================================
# INSPECTION ENDPOINTS
# =========================================================

@app.get("/states")
async def get_states():
    """
    Returns all states defined in config.json (dynamic).
    """
    try:
        return {"states": list(state_config.keys()), "count": len(state_config)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/collections")
async def get_collections():
    """
    Lists all Chroma collections and shows small sample of their content.
    """
    try:
        cols = chroma_client.list_collections()
        result = []

        for col in cols:
            c = chroma_client.get_collection(col.name)
            count = c.count()

            peek = None
            if count > 0:
                pdata = c.peek(limit=3)
                peek = {
                    "documents": pdata.get("documents", []),
                    "metadatas": pdata.get("metadatas", []),
                    "ids": pdata.get("ids", []),
                }

            result.append(
                {
                    "name": col.name,
                    "count": count,
                    "sample": peek,
                }
            )

        return {"collections": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =========================================================
# UPLOAD ENDPOINT
# =========================================================

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...), state: str = Form(...)
):
    """
    Uploads files into /app/Data/<state> for later vectorization.
    """
    try:
        state_folder = state.lower().replace(" ", "_")
        upload_dir = DATA_DIR / state_folder
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[str] = []

        for f in files:
            dest = upload_dir / f.filename
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(str(dest))

        return {
            "message": "Uploaded successfully",
            "file_paths": saved_paths,
            "state": state,
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =========================================================
# LOGIN ENDPOINT (DEMO)
# =========================================================

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/login")
async def login(request: LoginRequest):
    """
    Demo login endpoint — replace with real authentication later.
    """
    try:
        if request.username == "admin" and request.password == "admin123":
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Login successful",
                    "user": {"username": request.username, "role": "admin"},
                },
            )
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Invalid username or password"},
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Login failed: {str(e)}"},
        )


# =========================================================
# PLACEHOLDER /ask ENDPOINT
# =========================================================

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Placeholder API for simple QA (not used by new chat-feature backend).
    """
    return {"message": "placeholder"}
