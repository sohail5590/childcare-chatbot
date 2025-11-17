import streamlit as st 
import requests
import pandas as pd
import json
from pathlib import Path

BACKEND_URL = "http://backend:9000"
CONFIG_FILE = Path("/app/config.json")

# Function to load available states from config
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_states():
    """Load available states from backend API or config file"""
    try:
        # First try to get from backend API
        response = requests.get(f"{BACKEND_URL}/states", timeout=2)
        if response.status_code == 200:
            data = response.json()
            states = data.get("states", [])
            if states:
                return states
    except:
        pass
    
    # Fallback: Try to read from local config file
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
                return [state['name'] for state in config_data['states']]
    except:
        pass
    
    # Final fallback to default states
    return ["California", "New York"]

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = ""
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "California"
if 'show_login_page' not in st.session_state:
    st.session_state.show_login_page = False

st.set_page_config(
    page_title="Unified Child Care Portal",
    layout="wide"
)

# ==================== LOGIN/ADMIN PAGE ====================
if st.session_state.show_login_page:
    
    # Show logout button at the top right if authenticated
    if st.session_state.authenticated:
        col_title, col_logout = st.columns([4, 1])
        with col_title:
            st.title("🔐 Admin Panel")
        with col_logout:
            if st.button("← Back to Main", type="secondary"):
                st.session_state.show_login_page = False
                st.rerun()
            if st.button("Logout", type="primary"):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.session_state.show_login_page = False
                st.rerun()
    else:
        st.title("🔐 Admin Login")
    
    st.markdown("---")
    
    # ============ LOGIN FORM (if not authenticated) ============
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.subheader("Please login to access admin features")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    login_button = st.form_submit_button("Login", type="primary", use_container_width=True)
                with col_btn2:
                    cancel_button = st.form_submit_button("Cancel", use_container_width=True)
                
                if cancel_button:
                    st.session_state.show_login_page = False
                    st.rerun()
                
                if login_button:
                    if username and password:
                        with st.spinner("Authenticating..."):
                            try:
                                # Call backend login endpoint
                                response = requests.post(
                                    f"{BACKEND_URL}/login",
                                    json={
                                        "username": username,
                                        "password": password
                                    }
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    if result.get("success"):
                                        st.session_state.authenticated = True
                                        st.session_state.username = username
                                        st.success("✅ Login successful!")
                                        st.rerun()
                                    else:
                                        st.error("❌ " + result.get("message", "Login failed"))
                                else:
                                    result = response.json()
                                    st.error("❌ " + result.get("message", "Invalid username or password"))
                            
                            except requests.exceptions.ConnectionError:
                                st.error("❌ Cannot connect to backend service")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter both username and password")
    
    # ============ DOCUMENT UPLOAD SECTION (after successful login) ============
    else:
        st.success(f"Welcome, **{st.session_state.username}**! 👋")
        st.markdown("---")
        
        st.subheader("📤 Upload Documents to Knowledge Base")
        st.caption("Upload documents to vectorize and store in ChromaDB")
        
        # Load available states
        available_states = load_states()
        
        # State selection for document upload (Dropdown)
        upload_state = st.selectbox(
            "Select State for Documents:",
            options=available_states,
            key="upload_state_selector",
            help="Choose which state's knowledge base to upload to"
        )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'docx', 'doc', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, Word, CSV, or text files"
        )
        
        if uploaded_files:
            st.info(f"📁 Selected {len(uploaded_files)} file(s): {', '.join([f.name for f in uploaded_files])}")
            
            if st.button("🚀 Upload & Vectorize to ChromaDB", type="primary", use_container_width=True):
                with st.spinner("Uploading and vectorizing documents..."):
                    try:
                        files = []
                        for uploaded_file in uploaded_files:
                            files.append(
                                ('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                            )
                        
                        # Step 1: Upload files to backend
                        upload_response = requests.post(
                            f"{BACKEND_URL}/upload",
                            files=files,
                            data={'state': upload_state}
                        )
                        
                        if upload_response.status_code == 200:
                            upload_result = upload_response.json()
                            st.success(f"✅ Step 1: Uploaded {len(uploaded_files)} file(s) to {upload_state}")
                            
                            # Step 2: Vectorize the uploaded documents
                            with st.spinner("Step 2: Vectorizing documents..."):
                                vectorize_response = requests.post(
                                    f"{BACKEND_URL}/vectorize",
                                    json={
                                        'state': upload_state,
                                        'file_paths': upload_result.get('file_paths', [])
                                    }
                                )
                                
                                if vectorize_response.status_code == 200:
                                    vectorize_result = vectorize_response.json()
                                    st.success(f"✅ Step 2: Successfully vectorized and stored in ChromaDB!")
                                    
                                    # Show summary
                                    with st.expander("📊 Upload Summary", expanded=True):
                                        st.write(f"**State:** {upload_state}")
                                        st.write(f"**Files Processed:** {len(vectorize_result.get('processed_files', []))}")
                                        st.write(f"**Total Chunks Created:** {vectorize_result.get('total_chunks', 0)}")
                                        st.write(f"**Processed Files:** {', '.join(vectorize_result.get('processed_files', []))}")
                                else:
                                    st.warning("⚠️ Files uploaded but vectorization failed")
                                    st.error(vectorize_response.json().get('error', 'Unknown error'))
                        else:
                            st.error("❌ Error uploading documents")
                            st.error(upload_response.json().get('error', 'Unknown error'))
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend service")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("🔍 Quick Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 View Collections", use_container_width=True):
                with st.spinner("Fetching collections..."):
                    try:
                        response = requests.get(f"{BACKEND_URL}/collections")
                        if response.status_code == 200:
                            data = response.json()
                            st.json(data)
                        else:
                            st.error("Failed to fetch collections")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("🏠 Back to Main App", use_container_width=True):
                st.session_state.show_login_page = False
                st.rerun()
    
    st.stop()  # Stop rendering the rest of the page

# ==================== MAIN APPLICATION ====================

# Sidebar - Login Button or User Info
if not st.session_state.authenticated:
    if st.sidebar.button("🔐 Admin Login", use_container_width=True, type="primary"):
        st.session_state.show_login_page = True
        st.rerun()
else:
    st.sidebar.success(f"Welcome, {st.session_state.username}! 👋")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.show_login_page = False
        st.rerun()

st.sidebar.markdown("---")

# Frequently Asked Questions Section
st.sidebar.title("❓ Frequently Asked Questions")
st.sidebar.caption("Copy and paste these questions if needed")

example_prompts = [
    "Show me the highest-rated child care programs",
    "Find child care programs for toddlers",
    "What are some after-school programs available?",
    "What are the eligibility requirements?",
    "How do I apply for child care assistance?",
]

for i, prompt in enumerate(example_prompts, 1):
    st.sidebar.markdown(f"**{i}.** {prompt}")

st.sidebar.markdown("---")

# Main Content Area
st.title("🏫 Unified Child Care Query Engine")

# Load available states
available_states = load_states()

# State Selection with Dropdown (Always visible - no login required)
st.subheader("Select State")
default_index = 0
if st.session_state.selected_state in available_states:
    default_index = available_states.index(st.session_state.selected_state)

selected_state = st.selectbox(
    "Choose your state:",
    options=available_states,
    index=default_index,
    key="state_selector",
    help="Select the state for your child care query"
)
st.session_state.selected_state = selected_state

st.markdown("---")

# Question Input Section (Always visible - no login required)
st.subheader("Ask Your Question")
user_question = st.text_area(
    "Enter your question about child care services:", 
    value=st.session_state.selected_prompt,
    height=100,
    placeholder="Type your question here...",
    key="question_input"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.session_state.selected_prompt = ""
    st.rerun()

if ask_button:
    # Get the question from the text area widget
    question_to_ask = user_question.strip()
    
    if question_to_ask == "":
        st.warning("⚠️ Please enter a question.")
    else:
        # Clear the selected prompt after using it
        if st.session_state.selected_prompt:
            st.session_state.selected_prompt = ""
            
        with st.spinner("🔄 Getting answer..."):
            try:
                # Send question with selected state to backend
                response = requests.post(
                    f"{BACKEND_URL}/ask", 
                    json={
                        "question": question_to_ask,
                        "state": selected_state
                    }
                )
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                    st.markdown("---")
                    st.markdown("### 💡 Answer:")
                    st.info(f"**State:** {selected_state}")
                    st.write(answer)
                else:
                    st.error("❌ Error from backend service.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend service. Please ensure the backend is running.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Information footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 <strong>Tip:</strong> Login is only required for administrators to upload documents to the knowledge base.</p>
    <p>Anyone can ask questions without logging in!</p>
</div>
""", unsafe_allow_html=True)