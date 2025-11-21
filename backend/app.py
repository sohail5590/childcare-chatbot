from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import shutil
import json
from pathlib import Path
import time
import traceback
import re
import requests

from openai import OpenAI

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # adjust based on your directory
env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")
else:
    print(f".env not found at {env_path}")

# =========================================================
# Initialize FastAPI app
# =========================================================
app = FastAPI()

# Load configuration file
CONFIG_FILE = Path("/app/config.json")
state_config = {}
state_collections = {}

def load_config():
    global state_config, state_collections
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
                state_config = {state['name']: state['collection_name'] for state in config_data['states']}
                print(f"[CONFIG] Loaded states: {list(state_config.keys())}", flush=True)
        else:
            state_config = {
                "California": "california_state",
                "New York": "newyork_state"
            }
            print("[CONFIG] Using default states (config.json not found)", flush=True)
    except Exception as e:
        print(f"[CONFIG ERROR] {e}", flush=True)
        state_config = {
            "California": "california_state",
            "New York": "newyork_state"
        }

load_config()

# =========================================================
# OpenAI client
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
oai = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_LLM_MODEL = "gpt-4o-mini"

# =========================================================
# Base directory for uploads
# =========================================================
DATA_DIR = Path("/app/Data")
DATA_DIR.mkdir(exist_ok=True)

# =========================================================
# Connect to ChromaDB
# =========================================================
try:
    for attempt in range(10):
        chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
        if chroma_client.heartbeat():
            print("✅ Connected to ChromaDB!")
            break
        time.sleep(5)
except Exception as e:
    raise RuntimeError("❌ Could not connect to ChromaDB.", str(e))

# Create state collections
for state_name, collection_name in state_config.items():
    try:
        state_collections[state_name] = chroma_client.get_or_create_collection(name=collection_name)
        print(f"[COLLECTION] Ready: {collection_name}")
    except Exception as e:
        print(f"[COLLECTION ERROR] {e}", flush=True)

# QA collection
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
# Category labels
# =========================================================
CATEGORY_LABELS = [
    "definition",
    "requirement",
    "procedure",
    "statistic",
    "benefit",
    "challenge",
    "feature",
    "comparison"
]

# =========================================================
# Heuristic scoring functions
# =========================================================
def heuristic_category_scores(text: str) -> Dict[str, int]:
    t = text.lower()
    scores = {cat: 0 for cat in CATEGORY_LABELS}

    # Definitions
    definition_keywords = [
        "is defined as", "refers to", "means", "defined as",
        "in this section,", "for the purposes of"
    ]

    # Statistic
    statistic_patterns = [r"\b\d{1,3}%\b"]
    statistic_keywords = [
        "percent", "percentage", "ratio", "figure", "table", "chart",
        "survey", "data", "results", "average", "median", "mean", "distribution"
    ]

    # Requirement
    requirement_keywords = [
        "must", "shall", "required", "mandatory", "shall not",
        "must not", "compliance", "regulation", "rule", "criteria", "eligibility"
    ]

    # Procedure
    procedure_keywords = [
        "steps", "step", "process", "procedure", "workflow",
        "sequence", "how to", "instructions", "method", "stage", "phase"
    ]

    # Benefit
    benefit_keywords = [
        "benefit", "advantage", "improves", "increases", "reduces",
        "helps", "enhances", "positive outcome", "gain"
    ]

    # Challenge
    challenge_keywords = [
        "challenge", "difficulty", "problem", "issue", "barrier",
        "concern", "hard to", "struggle", "limitation"
    ]

    # Feature
    feature_keywords = [
        "feature", "characteristic", "attribute", "property",
        "type", "category", "includes", "consists of"
    ]

    # Comparison
    comparison_keywords = [
        "compared to", "versus", "vs", "difference",
        "similarity", "in contrast", "better than", "worse than"
    ]

    for kw in definition_keywords:
        if kw in t:
            scores["definition"] += 2

    for pattern in statistic_patterns:
        if re.search(pattern, t):
            scores["statistic"] += 3
    for kw in statistic_keywords:
        if kw in t:
            scores["statistic"] += 1

    for kw in requirement_keywords:
        if kw in t:
            scores["requirement"] += 1

    for kw in procedure_keywords:
        if kw in t:
            scores["procedure"] += 1

    for kw in benefit_keywords:
        if kw in t:
            scores["benefit"] += 1

    for kw in challenge_keywords:
        if kw in t:
            scores["challenge"] += 1

    for kw in feature_keywords:
        if kw in t:
            scores["feature"] += 1

    for kw in comparison_keywords:
        if kw in t:
            scores["comparison"] += 1

    return scores

def normalize_scores(scores: Dict[str, int]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score == 0:
        return {}
    return {k: round(v / max_score, 3) for k, v in scores.items() if v > 0}

# =========================================================
# LLM fallback (OpenAI)
# =========================================================
def llm_category_fallback(text: str) -> Dict[str, float]:
    try:
        labels_str = ", ".join(CATEGORY_LABELS)
        prompt = (
            "Classify the following text into one or more categories.\n"
            f"Valid categories: {labels_str}.\n"
            "Return a comma-separated list.\n\n"
            f"Text:\n{text[:4000]}\n\n"
            "Categories:"
        )

        resp = oai.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = resp.choices[0].message["content"].lower().strip()

        if not raw:
            return {}

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        result = {p: 1.0 for p in parts if p in CATEGORY_LABELS}

        return result

    except Exception as e:
        print(f"[OPENAI FALLBACK ERROR] {e}", flush=True)
        return {}

# =========================================================
# Combined classifier
# =========================================================
def get_chunk_categories(text: str) -> Dict[str, float]:
    scores = heuristic_category_scores(text)
    normalized = normalize_scores(scores)

    if normalized:
        return normalized

    llm_result = llm_category_fallback(text)
    if llm_result:
        return llm_result

    return {"other": 1.0}

# Endpoint for Login
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
async def login(request: LoginRequest):
    """
    Authenticate user with username and password.
    Returns success status and user information.
    """
    try:
        username = request.username
        password = request.password
        
        print(f"[LOGIN] Login attempt for user: {username}")
        
        # TODO: Replace with actual authentication logic (database, OAuth, etc.)
        # For now, using simple demo credentials
        if username == "admin" and password == "admin123":
            print(f"[LOGIN] Successful login for user: {username}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Login successful",
                    "user": {
                        "username": username,
                        "role": "admin"
                    }
                }
            )
        else:
            print(f"[LOGIN] Failed login attempt for user: {username}")
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "message": "Invalid username or password"
                }
            )
    
    except Exception as e:
        print(f"[LOGIN ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Login failed: {str(e)}"
            }
        )



# =========================================================
# INSPECTION ENDPOINTS
# =========================================================
@app.get("/states")
async def get_states():
    try:
        return {"states": list(state_config.keys()), "count": len(state_config)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/collections")
async def get_collections():
    try:
        cols = chroma_client.list_collections()
        result = []
        for col in cols:
            c = chroma_client.get_collection(col.name)
            cnt = c.count()
            peek = None
            if cnt > 0:
                pdata = c.peek(limit=3)
                peek = {
                    "documents": pdata.get("documents", []),
                    "metadatas": pdata.get("metadatas", []),
                    "ids": pdata.get("ids", []),
                }
            result.append({"name": col.name, "count": cnt, "sample": peek})
        return {"collections": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# =========================================================
# UPLOAD
# =========================================================
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), state: str = Form(...)):
    try:
        state_dir = state.lower().replace(" ", "_")
        upload_dir = DATA_DIR / state_dir
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        for f in files:
            dest = upload_dir / f.filename
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved.append(str(dest))

        return {"message": "Uploaded", "file_paths": saved, "state": state}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# =========================================================
# VECTORIZE (FINAL)
# =========================================================
@app.post("/vectorize")
async def vectorize_documents(request: VectorizeRequest):
    try:
        state = request.state
        fpaths = request.file_paths

        if state not in state_collections:
            return {"error": f"Invalid state: {state}"}

        collection = state_collections[state]

        total_chunks = 0
        done_files = []

        for fp in fpaths:
            p = Path(fp)
            if not p.exists():
                print(f"[VECTORIZE] File missing: {fp}")
                continue

            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(str(p))
            elif ext == ".csv":
                loader = CSVLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p))
            else:
                print(f"[VECTORIZE] Unsupported: {ext}")
                continue

            docs = loader.load()
            chunks = text_splitter.split_documents(docs)

            for i, chunk in enumerate(chunks):
                text = chunk.page_content

                categories = get_chunk_categories(text)
                categories_json = json.dumps(categories)  # FIX for Chroma

                emb = model.encode(text).tolist()

                doc_id = f"{p.stem}_chunk_{i}_{total_chunks}"

                collection.add(
                    embeddings=[emb],
                    documents=[text],
                    metadatas=[{
                        "source": p.name,
                        "state": state,
                        "chunk_index": i,
                        "file_type": ext,
                        "categories": categories_json
                    }],
                    ids=[doc_id]
                )

                total_chunks += 1

            done_files.append(p.name)

        return {
            "message": "Vectorization complete",
            "processed_files": done_files,
            "total_chunks": total_chunks,
            "collection_count": collection.count()
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# =========================================================
# Placeholder /ask
# =========================================================
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    return {"message": "placeholder"}