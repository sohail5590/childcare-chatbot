import streamlit as st 
import requests
import pandas as pd

BACKEND_URL = "http://backend:9000"

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = ""
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "California"

st.set_page_config(
    page_title="Unified Child Care Portal",
    layout="wide"
)

# Login Section in Sidebar (for document upload only)
st.sidebar.title("🔐 Admin Login")
st.sidebar.caption("(Required only for document upload)")

if not st.session_state.authenticated:
    with st.sidebar.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            # TODO: Replace with actual authentication logic
            # For now, using simple demo credentials
            if username and password:
                if username == "admin" and password == "admin123":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")
            else:
                st.sidebar.warning("Please enter both username and password")
else:
    st.sidebar.success(f"Welcome, {st.session_state.username}! 👋")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

st.sidebar.markdown("---")

# Document Upload Section (Only visible when logged in)
if st.session_state.authenticated:
    st.sidebar.title("📤 Upload Documents")
    st.sidebar.caption("Upload documents to the knowledge base")
    
    # State selection for document upload
    upload_state = st.sidebar.radio(
        "Select State for Documents:",
        options=["California", "New York"],
        key="upload_state_selector",
        help="Choose which state's knowledge base to upload to"
    )
    
    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'doc', 'csv', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, Word, CSV, or text files"
    )
    
    if uploaded_files:
        print('Here Upload Files')
        if st.sidebar.button("Upload to ChromaDB", type="primary"):
            with st.spinner("Uploading and vectorizing documents..."):
                try:
                    files = []
                    for uploaded_file in uploaded_files:
                        files.append(
                            ('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                        )
                    print(files)
                    # Step 1: Upload files to backend
                    upload_response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        data={'state': upload_state}
                    )
                    
                    if upload_response.status_code == 200:
                        upload_result = upload_response.json()
                        print('Upload Result =', upload_result)
                        st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s) to {upload_state}")
                        
                        # Step 2: Vectorize the uploaded documents
                        print('Now Vectorizing')
                        vectorize_response = requests.post(
                            f"{BACKEND_URL}/vectorize",
                            json={
                                'state': upload_state,
                                'file_paths': upload_result.get('file_paths', [])
                            }
                        )
                        
                        if vectorize_response.status_code == 200:
                            st.sidebar.success(f"Successfully vectorized and stored in ChromaDB!")
                        else:
                            st.sidebar.warning("Files uploaded but vectorization failed")
                    else:
                        st.sidebar.error("Error uploading documents")
                except requests.exceptions.ConnectionError:
                    st.sidebar.error("Cannot connect to backend service")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")

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

# State Selection with Radio Buttons (Always visible - no login required)
st.subheader("Select State")
selected_state = st.radio(
    "Choose your state:",
    options=["California", "New York"],
    index=0 if st.session_state.selected_state == "California" else 1,
    horizontal=True,
    key="state_selector"
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
                    st.error("Error from backend service.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend service. Please ensure the backend is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Information footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 <strong>Tip:</strong> Login is only required for administrators to upload documents to the knowledge base.</p>
    <p>Anyone can ask questions without logging in!</p>
</div>
""", unsafe_allow_html=True)