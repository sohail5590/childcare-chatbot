# backend/ingest_data.py

# --- Import Necessary Libraries ---
# pandas is used for reading and manipulating the CSV data.
import pandas as pd
# chromadb is the client library for interacting with the Chroma vector database.
import chromadb
# SentenceTransformer is used to convert text descriptions into numerical vector embeddings.
from sentence_transformers import SentenceTransformer
# os is used to access environment variables for configuration.
import os
# SQLAlchemy's create_engine is used to create a robust connection to the MySQL database.
from sqlalchemy import create_engine

import re


# --- CONFIGURATION ---
# These constants make the script easy to configure and modify.

# ChromaDB connection details, read from environment variables with defaults.
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
# The name of the collection where vectors will be stored.
COLLECTION_NAME = "movies"
# The name of the pre-trained model to use for creating embeddings.
MODEL_NAME = 'all-MiniLM-L6-v2'
# The path to the source CSV file, located within the same directory.
CSV_FILE_PATH = 'movie_data.csv'

INIT_SQL = 'init.sql'


# --- Function to setup MySQL Database ---
def setup_mysql_database(df, table_name='movies'):
    """
    Creates a connection to the MySQL database and populates a table
    with the data from the provided pandas DataFrame.
    """
    print("--- Setting up MySQL Database ---")
    try:
        # Read database credentials from environment variables set in docker-compose.yml.
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        host = os.getenv("DB_HOST")
        db_name = os.getenv("MYSQL_DATABASE")

        # Create a SQLAlchemy engine to manage the database connection.
        engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{db_name}")

        # Sanitize column names to be valid for SQL (e.g., 'Movie Name' -> 'Movie_Name').
        sanitized_columns = {col: col.strip().replace(' ', '_') for col in df.columns}
        df_renamed = df.rename(columns=sanitized_columns)

        # Write the entire DataFrame to the MySQL table.
        # `if_exists='replace'` will drop the table if it already exists and create a new one.
        # `chunksize` helps manage memory for very large files.
        df_renamed.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=1000)
        
        print(f"Successfully created and populated MySQL table '{table_name}'.")

    except Exception as e:
        print(f"Error setting up MySQL database: {e}")
        exit()

def get_ddl_extracts():
    """
    reads the sql, extract table names and comments and return dataframe
    """
    # Read the DDL content from the file
    with open(INIT_SQL,'r',encoding='utf-8') as file:
        ddl = file.read()
    
    # The re.IGNORECASE flag makes the search for "COMMENT" case-insensitive
    regex = r"^\s*`?(\w+)`?\s+.*?COMMENT\s+'((?:''|[^'])*)'"
    
    # Find all CREATE TABLE blocks with comments
    table_blocks = re.findall(regex, ddl, re.IGNORECASE | re.MULTILINE)
    # Print the results
    #for column, comment in table_blocks:
        #print(f"Column: {column}, Comment: {comment}")


    #print(table_blocks)
    # Prepare data for DataFrame
    data = []
    for block in table_blocks:
     table_name = block[0]
     comment = block[1] if block[1] else block[2]
     data.append({'column': table_name, 'comment': comment})

    # Create DataFrame
    df = pd.DataFrame(data, columns=['column', 'comment'])

    print(df)
    return df

# --- Main Ingestion Logic ---
print("--- Starting Data Ingestion ---")
#get_ddl_extracts()

# 1. Read the source CSV file into a pandas DataFrame.
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Successfully loaded {len(df)} rows from {CSV_FILE_PATH}.")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# 2. Setup MySQL with all columns from the CSV.
setup_mysql_database(df.copy(), table_name='movies')

# 3. Setup ChromaDB for vector search.
print("--- Setting up ChromaDB ---")
try:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
    model = SentenceTransformer(MODEL_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("Successfully connected to ChromaDB and loaded SentenceTransformer model.")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    exit()
    
#new data frame
ndf = get_ddl_extracts()
# 4. Prepare data for ChromaDB.
#df.columns = df.columns.str.strip()
# `comments`: The text content we want to vectorize (the 'Description').
comments = ndf["comment"].tolist()

print(f"comments: {comments}")
# `metadatas`: All other columns are stored as factual metadata.
metadatas = ndf.drop(columns=["comment"]).to_dict('records')
# `ids`: A list of unique identifiers for each entry.
ids = [f"movie_{i}" for i in range(len(ndf))]

# 5. Generate embeddings and add all data to the ChromaDB collection.
print("Generating embeddings for ChromaDB... This may take a moment.")
try:
    # Convert text documents into numerical vector embeddings.
    embeddings = model.encode(comments, show_progress_bar=True).tolist()
    
    # Add the data to the collection in batches for efficiency and reliability.
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=comments[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"Added batch {i//batch_size + 1} of {(len(ids)-1)//batch_size + 1} to ChromaDB")
except Exception as e:
    print(f"An error occurred during ChromaDB ingestion: {e}")

print("--- Data Ingestion for both MySQL and ChromaDB is Complete ---")