from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import shutil
from pathlib import Path
import time
import traceback

# =========================================================
# Initialize FastAPI app
# =========================================================
app = FastAPI()

DATA_DIR = Path("/app/Data")
DATA_DIR.mkdir(exist_ok=True)

# =========================================================
# Connect to ChromaDB
# =========================================================
for attempt in range(10):
    try:
        chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
        chroma_client.heartbeat()
        print("✅ Connected to ChromaDB!")
        break
    except Exception as e:
        print(f"⏳ Waiting for ChromaDB (attempt {attempt+1}/10): {e}")
        time.sleep(3)
else:
    raise RuntimeError("❌ Could not connect to ChromaDB after multiple retries.")

# =========================================================
# Collections
# =========================================================
collection_california = chroma_client.get_or_create_collection(name="california_state")
collection_newyork = chroma_client.get_or_create_collection(name="newyork_state")
qa_collection_california = chroma_client.get_or_create_collection(name="qa_pairs")

# =========================================================
# Embedding model & text splitter
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
# INSPECTION ENDPOINTS
# =========================================================
@app.get("/collections")
async def get_collections():
    try:
        collections = chroma_client.list_collections()
        result = []
        for col in collections:
            collection = chroma_client.get_collection(col.name)
            count = collection.count()
            sample_data = None
            if count > 0:
                peek_data = collection.peek(limit=3)
                sample_data = {
                    "documents": peek_data.get("documents", []),
                    "metadatas": peek_data.get("metadatas", []),
                    "ids": peek_data.get("ids", []),
                }
            result.append({
                "name": col.name,
                "document_count": count,
                "sample_data": sample_data
            })
        return JSONResponse(status_code=200, content={
            "total_collections": len(result),
            "collections": result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =========================================================
# UPLOAD DOCUMENTS
# =========================================================
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), state: str = Form(...)):
    try:
        print(f"[UPLOAD] Received {len(files)} files for state: {state}")
        state_dir = state.lower().replace(" ", "_")
        upload_dir = DATA_DIR / state_dir
        upload_dir.mkdir(exist_ok=True, parents=True)
        saved_files = []

        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"[UPLOAD] Saved file: {file_path}")
            saved_files.append(str(file_path))

        return JSONResponse(status_code=200, content={
            "message": f"Successfully uploaded {len(saved_files)} file(s)",
            "file_paths": saved_files,
            "state": state
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =========================================================
# VECTORIZE DOCUMENTS
# =========================================================
@app.post("/vectorize")
async def vectorize_documents(request: VectorizeRequest):
    try:
        state = request.state
        file_paths = request.file_paths
        print(f"[VECTORIZE] Processing {len(file_paths)} files for state: {state}")

        # Select collection
        if state.lower() == "california":
            collection = collection_california
        elif state.lower() == "new york":
            collection = collection_newyork
        else:
            return JSONResponse(status_code=400, content={"error": f"Invalid state: {state}"})

        total_chunks = 0
        processed_files = []

        for file_path in file_paths:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"[VECTORIZE] File not found: {file_path}")
                continue

            ext = file_path_obj.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path_obj))
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(str(file_path_obj))
            elif ext == ".csv":
                loader = CSVLoader(str(file_path_obj))
            elif ext == ".txt":
                loader = TextLoader(str(file_path_obj))
            else:
                print(f"[VECTORIZE] Unsupported file type: {ext}")
                continue

            print(f"[VECTORIZE] Loading {file_path_obj.name} ...")
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            print(f"[VECTORIZE] Split into {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk.page_content).tolist()
                doc_id = f"{file_path_obj.stem}_chunk_{i}_{total_chunks}"
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk.page_content],
                    metadatas=[{
                        "source": file_path_obj.name,
                        "state": state,
                        "chunk_index": i,
                        "file_type": ext
                    }],
                    ids=[doc_id]
                )
                total_chunks += 1

            processed_files.append(file_path_obj.name)
            print(f"[VECTORIZE] ✅ Completed: {file_path_obj.name}")

        print(f"[VECTORIZE] Total chunks stored: {total_chunks}")
        print(f"[VECTORIZE] Collection count (after insert): {collection.count()}")

        # 🟩 Peek into stored chunks
        if collection.count() > 0:
            peek_data = collection.peek(limit=3)
            print("\n[DEBUG] Sample stored documents:")
            for i, doc in enumerate(peek_data.get("documents", [])):
                print(f"--- Chunk {i+1} ---")
                print(doc[:300].replace("\n", " "))
                print("Metadata:", peek_data["metadatas"][i])
                print()

        return JSONResponse(status_code=200, content={
            "message": f"Successfully vectorized {len(processed_files)} file(s)",
            "total_chunks": total_chunks,
            "processed_files": processed_files,
            "state": state,
            "collection_count": collection.count()
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Vectorization failed: {str(e)}"})


# =========================================================
# Placeholder for /ask
# =========================================================
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    return {"message": "Question endpoint placeholder"}