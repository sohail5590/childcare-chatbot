# frontend/app.py

import streamlit as st
import requests
import os
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Query Engine",
    page_icon="🎬",
    layout="wide"
)

# --- App Title and Description ---
st.title("🎬 The Ultimate Movie & TV Show Query Engine")
st.caption("Ask a question in natural language, and I'll convert it to a SQL query to search the movie database.")

# --- Environment Variables ---
# Get the backend URL from an environment variable, with a fallback for local development
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/generate-query")

# --- Session State Initialization ---
# This is crucial for maintaining the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you find a movie today?"}
    ]

# --- Helper Function to Call Backend ---
def get_backend_response(user_query: str):
    """Sends the user's query to the backend and returns the response."""
    try:
        response = requests.post(BACKEND_URL, json={"query": user_query}, timeout=30)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not connect to the backend: {e}"}

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Example Prompts ---
st.sidebar.title("Example Prompts")
example_prompts = [
    "Show me the highest-rated dramas from 1994",
    "Find action movies starring a famous actor from the 90s",
    "What are some movies about computer hackers?",
]
for prompt in example_prompts:
    if st.sidebar.button(prompt):
        st.session_state.selected_prompt = prompt
    else:
        st.session_state.selected_prompt = None
        
# --- User Input ---
if prompt := st.chat_input("Ask about a movie...") or st.session_state.selected_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Display assistant response in a "thinking" state
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = get_backend_response(prompt)
            
            # --- Handle Backend Response ---
            if "error" in response_data:
                error_message = response_data["error"]
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            else:
                sql_query = response_data.get("sql_query", "No SQL query generated.")
                result = response_data.get("result", [])
                
                # Format the full response content
                full_response = ""
                
                # Show the results table if data is returned
                if result:
                    df = pd.DataFrame(result)
                    st.dataframe(df, use_container_width=True)
                    full_response += "Here's what I found:\n"
                    full_response += df.to_string(index=False) + "\n\n"
                else:
                    st.warning("I couldn't find any results for your query.")
                    full_response += "I couldn't find any results for your query.\n\n"

                # Show the SQL query in an expander
                with st.expander("View Generated SQL Query"):
                    st.code(sql_query, language="sql")
                full_response += f"Generated SQL:\n{sql_query}"

                # Add the complete response to session state (for history)
                st.session_state.messages.append({"role": "assistant", "content": full_response})