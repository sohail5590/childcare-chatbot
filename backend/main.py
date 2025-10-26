# backend/main.py

# --- Standard Library and Framework Imports ---
import os
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests # We'll use requests to call the Ollama API

# --- RAG and Machine Learning Imports ---
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables from the .env file
load_dotenv()

# --- Application Initialization ---
app = FastAPI()

# --- Ollama and Prompt Configuration ---
OLLAMA_URL = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:11434/api/generate"
OLLAMA_MODEL = "codegemma:7b" # Or another model you have pulled, like 'mistral'
# This multiline string is the master prompt for the LLM.
PROMPT_TEMPLATE = """
You are a MySQL expert. Given a database schema, your job is to write a SQL query from the required columns and user question. Provide aggregate queries if necessary
### DATABASE SCHEMA:
The database contains a single table named `movies`.
The columns in the `movies` table are:
- `Movie_Name` (VARCHAR): The title of the movie.
- `Year_Of_Release` (INT): The year the movie was released.
- `Watch_Time` (VARCHAR): The duration of the movie (e.g., '152 min').
- `Movie_Rating` (DECIMAL): The public rating of the movie (e.g., 8.7).
- `Metascore_of_Movie` (INT): The critic score for the movie (e.g., 74).
- `Votes` (INT): The number of votes the movie received.
- `Gross` (VARCHAR): The gross box office revenue (e.g., '$291.98M').
- `Description` (TEXT): A summary of the movie's plot.
### INSTRUCTIONS:
- Write only the SQL query to answer the question.
- Do NOT include any explanations, comments, or markdown formatting.
- Ensure the query is valid for MySQL.
- Return Movie_Name in case of non-aggregated queries
### USER QUESTION:
{question}
###required columns
{requiredclmns}
### SQL QUERY:
"""


# --- Global Client Initialization ---
# These clients are initialized once when the application starts up for efficiency.
try:
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=8000)
    
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Getting 'movies' collection...")
    movie_collection = chroma_client.get_collection(name="movies")
    
    print("Successfully connected to ChromaDB and loaded model.")
except Exception as e:
    print(f"FATAL: Failed to initialize ChromaDB or SentenceTransformer model: {e}")
    movie_collection = None

# --- Pydantic Model ---
# Defines the expected structure for the incoming request body
class QueryRequest(BaseModel):
    query: str

# --- Database Connection Function ---
def get_db_connection():
    """Establishes and returns a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

# --- API Endpoints ---
@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"Status": "Backend is running!"}


@app.post("/generate-query")
def generate_query(request: QueryRequest):
    """
    Takes a natural language query, finds relevant movies in ChromaDB, 
    and then fetches their details from MySQL.
    """
    if movie_collection is None:
        raise HTTPException(status_code=503, detail="Vector database is not available.")

    # Step 1: Vectorize the user's query
    query_embedding = embedding_model.encode(request.query).tolist()

    # Step 2: Query ChromaDB to find the top 5 most similar movie descriptions
    results = movie_collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    print("\n--- DEBUG: ChromaDB Metadata Received ---")
    if results['metadatas'] and results['metadatas'][0]:
        # Print the first metadata dictionary from the results
        print(results['metadatas'][0]) 
    
    # Step 3: Extract the 'Movie_Name' from the metadata of the results
    column_names = [meta['column'] for meta in results['metadatas'][0]]
    
    if not column_names:
        return None
        

    # Step 4: Construct and execute a simple, safe SQL query against MySQL
    placeholders = ', '.join(['%s'] * len(column_names))
    
     # --- Step a: Generate the SQL query using Ollama ---
    generated_sql = generate_sql_from_ollama(request.query, column_names)
    
    if not generated_sql:
        raise HTTPException(status_code=500, detail="Failed to generate SQL query from the LLM.")

    ##sql_query = f"SELECT * FROM movies WHERE Movie_Name IN ({placeholders});"

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Structured database connection unavailable.")

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(generated_sql)
        final_result = cursor.fetchall()
        
        return {
            "user_query": request.query,
            "sql_query": cursor.statement,
            "result": final_result
        }
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database query failed: {err}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            
def generate_sql_from_ollama(user_question: str, requiredcolumns: list) -> str:
    """Sends the prompt to Ollama and gets the generated SQL query."""
    try:
        full_prompt = PROMPT_TEMPLATE.format(question=user_question, requiredclmns= requiredcolumns)
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False # We want the full response at once
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        # Clean up the response: strip whitespace and remove markdown code blocks
        generated_sql = response.json().get("response", "").strip()
        if generated_sql.startswith("```sql"):
            generated_sql = generated_sql[6:]
        if generated_sql.endswith("```"):
            generated_sql = generated_sql[:-3]
        return generated_sql.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None
