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

import requests
# import psycopg2
import os
import shutil
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Base directory for storing uploaded files
DATA_DIR = Path("/app/Data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB client - Connect to ChromaDB container
chroma_client = chromadb.HttpClient(host='chromodb', port=8000)

# Create or get collections for both states
collection_california = chroma_client.get_or_create_collection(name="california_state")
collection_newyork = chroma_client.get_or_create_collection(name="newyork_state")
qa_collection_california = chroma_client.get_or_create_collection(name="qa_pairs")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Define request model
class QuestionRequest(BaseModel):
    question: str
    state: str = "California"

class SavePairRequest(BaseModel):
    question: str
    sql: str

class VectorizeRequest(BaseModel):
    state: str
    file_paths: List[str]

# Ollama endpoint
OLLAMA_URL = "http://ollama:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# Endpoint for Login (Placeholder)
@app.post("/login")
async def login():
    pass


# ==================== INSPECTION ENDPOINTS ====================

@app.get("/collections")
async def get_collections():
    """
    Get list of all ChromaDB collections with their details.
    Returns collection names, document counts, and sample data.
    """
    try:
        collections = chroma_client.list_collections()
        
        result = []
        for col in collections:
            collection = chroma_client.get_collection(col.name)
            count = collection.count()
            
            # Get sample documents if collection has data
            sample_data = None
            if count > 0:
                peek_data = collection.peek(limit=3)
                sample_data = {
                    "documents": peek_data.get("documents", []),
                    "metadatas": peek_data.get("metadatas", []),
                    "ids": peek_data.get("ids", [])
                }
            
            result.append({
                "name": col.name,
                "document_count": count,
                "sample_data": sample_data
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "total_collections": len(result),
                "collections": result
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get collections: {str(e)}"}
        )


@app.get("/collection/{collection_name}")
async def get_collection_details(collection_name: str, limit: int = 10):
    """
    Get detailed information about a specific collection.
    Query params:
    - limit: number of sample documents to return (default: 10)
    """
    try:
        print(f"[COLLECTION DETAILS] Attempting to get collection: {collection_name}")
        
        # Try to get the collection
        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception as get_error:
            print(f"[COLLECTION DETAILS] Error getting collection: {type(get_error).__name__}: {str(get_error)}")
            # List all available collections for debugging
            all_collections = chroma_client.list_collections()
            available = [c.name for c in all_collections]
            print(f"[COLLECTION DETAILS] Available collections: {available}")
            
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Collection '{collection_name}' not found",
                    "available_collections": available
                }
            )
        
        count = collection.count()
        print(f"[COLLECTION DETAILS] Collection has {count} documents")
        
        # Get sample documents
        peek_limit = min(limit, count) if count > 0 else 0
        print(f"[COLLECTION DETAILS] Peeking with limit: {peek_limit}")
        
        peek_data = collection.peek(limit=peek_limit)
        
        # Get collection metadata if available
        metadata = collection.metadata if hasattr(collection, 'metadata') else None
        
        # Convert embeddings to list if they exist (to make them JSON serializable)
        embeddings = peek_data.get("embeddings", [])
        embeddings_sample = []
        
        # Check if embeddings exist (use len() instead of boolean check to avoid numpy array ambiguity)
        if embeddings is not None and len(embeddings) > 0:
            # Convert numpy arrays to lists
            embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            embeddings_sample = embeddings_list[0][:10] if len(embeddings_list) > 0 else []
        
        print(f"[COLLECTION DETAILS] Successfully retrieved collection details")
        
        return JSONResponse(
            status_code=200,
            content={
                "name": collection_name,
                "document_count": count,
                "metadata": metadata,
                "sample_documents": {
                    "documents": peek_data.get("documents", []),
                    "metadatas": peek_data.get("metadatas", []),
                    "ids": peek_data.get("ids", []),
                    "embeddings_count": len(embeddings),
                    "embeddings_sample": embeddings_sample
                }
            }
        )
    
    except Exception as e:
        print(f"[COLLECTION DETAILS ERROR] Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get collection details: {str(e)}"}
        )


@app.get("/collection/{collection_name}/search")
async def search_collection(collection_name: str, query: str, n_results: int = 5):
    """
    Search a collection using a text query.
    Query params:
    - query: search text
    - n_results: number of results to return (default: 5)
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        
        # Generate query embedding
        query_embedding = model.encode(query).tolist()
        
        # Search collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "collection": collection_name,
                "results": {
                    "documents": results.get("documents", [[]])[0],
                    "metadatas": results.get("metadatas", [[]])[0],
                    "distances": results.get("distances", [[]])[0],
                    "ids": results.get("ids", [[]])[0]
                }
            }
        )
    
    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={"error": f"Collection '{collection_name}' not found"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Search failed: {str(e)}"}
        )


@app.delete("/collection/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a collection (use with caution!)
    """
    try:
        chroma_client.delete_collection(collection_name)
        return JSONResponse(
            status_code=200,
            content={"message": f"Collection '{collection_name}' deleted successfully"}
        )
    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={"error": f"Collection '{collection_name}' not found"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete collection: {str(e)}"}
        )


# ==================== END INSPECTION ENDPOINTS ====================

# Endpoint for document upload and saving to Data directory
@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    state: str = Form(...)
):
    """
    Upload documents and save them to the Data directory organized by state.
    Returns the file paths for subsequent vectorization.
    """
    try:
        print(f"[UPLOAD] Received {len(files)} files for state: {state}")
        print(f"[UPLOAD] Base DATA_DIR: {DATA_DIR}")
        
        # Normalize state name for directory
        state_dir = state.lower().replace(" ", "_")
        upload_dir = DATA_DIR / state_dir
        upload_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"[UPLOAD] Upload directory: {upload_dir}")
        print(f"[UPLOAD] Directory exists: {upload_dir.exists()}")
        
        saved_files = []
        
        for file in files:
            # Create safe filename
            file_path = upload_dir / file.filename
            
            print(f"[UPLOAD] Saving file: {file.filename} to {file_path}")
            
            # Save file to disk
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"[UPLOAD] File saved. Size: {file_path.stat().st_size} bytes")
            saved_files.append(str(file_path))
        
        print(f"[UPLOAD] Successfully saved {len(saved_files)} files")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded {len(saved_files)} file(s)",
                "file_paths": saved_files,
                "state": state
            }
        )
    
    except Exception as e:
        print(f"[UPLOAD ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )


# Endpoint to vectorize documents and store in ChromaDB
@app.post("/vectorize")
async def vectorize_documents(request: VectorizeRequest):
    """
    Read documents using LangChain, vectorize content, and store in ChromaDB.
    Handles PDF, DOCX, CSV, and TXT files.
    """
    try:
        state = request.state
        file_paths = request.file_paths
        
        print(f"[VECTORIZE] Processing {len(file_paths)} files for state: {state}")
        
        # Select appropriate collection based on state
        if state.lower() == "california":
            collection = collection_california
        elif state.lower() == "new york":
            collection = collection_newyork
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid state: {state}"}
            )
        
        print(f"[VECTORIZE] Using collection: {collection.name}")
        
        total_chunks = 0
        processed_files = []
        
        for file_path in file_paths:
            file_path_obj = Path(file_path)
            
            print(f"[VECTORIZE] Processing file: {file_path}")
            print(f"[VECTORIZE] File exists: {file_path_obj.exists()}")
            
            if not file_path_obj.exists():
                print(f"[VECTORIZE] File not found, skipping: {file_path}")
                continue
            
            # Load document based on file type
            try:
                file_extension = file_path_obj.suffix.lower()
                
                print(f"[VECTORIZE] File extension: {file_extension}")
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(str(file_path_obj))
                elif file_extension in ['.docx', '.doc']:
                    loader = UnstructuredWordDocumentLoader(str(file_path_obj))
                elif file_extension == '.csv':
                    loader = CSVLoader(str(file_path_obj))
                elif file_extension == '.txt':
                    loader = TextLoader(str(file_path_obj))
                else:
                    print(f"[VECTORIZE] Unsupported file type: {file_extension}")
                    continue
                
                # Load and split documents
                print(f"[VECTORIZE] Loading document...")
                documents = loader.load()
                print(f"[VECTORIZE] Loaded {len(documents)} document(s)")
                
                chunks = text_splitter.split_documents(documents)
                print(f"[VECTORIZE] Split into {len(chunks)} chunks")
                
                # Vectorize and store in ChromaDB
                for i, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = model.encode(chunk.page_content).tolist()
                    
                    # Create unique ID
                    doc_id = f"{file_path_obj.stem}_chunk_{i}_{total_chunks}"
                    
                    # Add to collection
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk.page_content],
                        metadatas=[{
                            "source": file_path_obj.name,
                            "state": state,
                            "chunk_index": i,
                            "file_type": file_extension
                        }],
                        ids=[doc_id]
                    )
                    total_chunks += 1
                
                processed_files.append(file_path_obj.name)
                print(f"[VECTORIZE] Successfully processed: {file_path_obj.name}")
            
            except Exception as e:
                print(f"[VECTORIZE ERROR] Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[VECTORIZE] Total chunks stored: {total_chunks}")
        print(f"[VECTORIZE] Collection count: {collection.count()}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully vectorized {len(processed_files)} file(s)",
                "total_chunks": total_chunks,
                "processed_files": processed_files,
                "state": state
            }
        )
    
    except Exception as e:
        print(f"[VECTORIZE ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Vectorization failed: {str(e)}"}
        )

# Endpoint to handle user questions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    pass