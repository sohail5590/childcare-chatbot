import os
import json
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:9000")
DB_API_URL = os.getenv("DB_API_URL", "http://db-api:9200")
CONFIG_FILE = Path("/app/config.json")


# ---------------------------------------------------------
# HELPERS TO TALK TO DB-API
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def load_states_data():
    """Return (state_names, {name: id}) from DB-API /states."""
    try:
        resp = requests.get(f"{DB_API_URL}/states", timeout=5)
        if resp.status_code == 200:
            states = resp.json() or []
            state_names = [s["name"] for s in states]
            name_to_id = {s["name"]: s["id"] for s in states}
            return state_names, name_to_id
    except Exception:
        pass

    # fallback: config
    try:
        if CONFIG_FILE.exists():
            cfg = json.loads(CONFIG_FILE.read_text())
            states = [s["name"] for s in cfg.get("states", [])]
            return states, {s: i + 1 for i, s in enumerate(states)}
    except Exception:
        pass

    # last fallback: default
    states = ["California", "Kentucky", "Ohio", "New York"]
    return states, {s: i + 1 for i, s in enumerate(states)}


def get_states_with_documents():
    try:
        resp = requests.get(f"{DB_API_URL}/states/with-documents", timeout=5)
        if resp.status_code == 200:
            return resp.json() or []
    except:
        pass
    return []


def get_counties_for_state(state_id: int):
    try:
        resp = requests.get(f"{DB_API_URL}/counties", params={"state_id": state_id}, timeout=5)
        if resp.status_code == 200:
            return resp.json() or []
    except:
        pass
    return []


def create_document_record(state_id, county_id, file_name, file_path, uploaded_by):
    payload = {
        "state_id": state_id,
        "county_id": county_id,
        "file_name": file_name,
        "file_path": file_path,
        "uploaded_by": uploaded_by,
        "vector_status": "complete",
        "vector_collection": None,
    }
    try:
        requests.post(f"{DB_API_URL}/documents", json=payload, timeout=10)
    except:
        pass


def get_documents_for_state(state_id: int):
    try:
        resp = requests.get(f"{DB_API_URL}/documents", params={"state_id": state_id}, timeout=5)
        if resp.status_code == 200:
            return resp.json() or []
    except:
        pass
    return []


def delete_document_record(doc_id: int):
    try:
        requests.delete(f"{DB_API_URL}/documents/{doc_id}", timeout=5)
    except:
        pass


def update_document_record(doc_id: int, payload: dict):
    try:
        requests.put(f"{DB_API_URL}/documents/{doc_id}", json=payload, timeout=5)
    except:
        pass


def delete_from_vector_store(file_path: str):
    """Call backend to delete vectors for a file."""
    try:
        requests.post(f"{BACKEND_URL}/delete_document", json={"file_path": file_path}, timeout=15)
    except:
        pass


def revectorize_document(state: str, uploaded_file, old_file_path: str | None):
    """Upload + vectorize replacement file; backend handles this."""
    try:
        new_files = [("files", (uploaded_file.name, uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream"))]

        # upload
        upload_resp = requests.post(
            f"{BACKEND_URL}/upload",
            files=new_files,
            data={"state": state},
            timeout=120,
        )
        if upload_resp.status_code != 200:
            return None, None

        file_paths = upload_resp.json().get("file_paths", [])
        if not file_paths:
            return None, None
        new_file_path = file_paths[0]

        # vectorize
        vect_resp = requests.post(
            f"{BACKEND_URL}/vectorize",
            json={"state": state, "file_paths": [new_file_path]},
            timeout=300,
        )
        if vect_resp.status_code != 200:
            return None, None

        # cleanup old vector file
        if old_file_path:
            delete_from_vector_store(old_file_path)

        return uploaded_file.name, new_file_path

    except Exception:
        return None, None


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
defaults = {
    "authenticated": False,
    "username": "",
    "selected_prompt": "",
    "selected_state": "California",
    "show_login_page": True,
    "state_id_map": {},
    "confirm_delete_doc_id": None,
    "editing_doc": None,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.set_page_config(page_title="Unified Childcare Portal", layout="wide")


# =========================================================
#  LOGIN PAGE (UNCHANGED — DO NOT MODIFY)
# =========================================================
def show_login_page():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stHeader"] {display: none;}
        body { background-color: #f1f5f9; }
        .login-card {
            background: white;
            padding: 2rem 2.2rem;
            border-radius: 15px;
            max-width: 420px;
            margin: 2.5rem auto 1.5rem auto;
            box-shadow: 0 4px 25px rgba(0,0,0,0.12);
            text-align: center;
        }
        .gov-logo { font-size: 36px; margin-bottom: 5px; }
        .gov-title { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
        .gov-subtitle { font-size: 0.9rem; color: #475569; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="login-card">
            <div class="gov-logo">🏛️</div>
            <div class="gov-title">Unified Childcare<br>Chat Assistant</div>
            <div class="gov-subtitle">Please sign in to continue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 2, 1])
    with center:
        username = st.text_input("Username", placeholder="Enter username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")

        if st.button("Login", type="primary", use_container_width=True):
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/login",
                        json={"username": username, "password": password},
                        timeout=10,
                    )
                    if resp.status_code == 200 and resp.json().get("success"):
                        st.session_state.authenticated = True
                        st.session_state.show_login_page = False
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

    st.stop()


# =========================================================
#  ADMIN PANEL
# =========================================================
def show_admin_panel():

    st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>",
                unsafe_allow_html=True)

    # Load states from DB
    state_names, id_map = load_states_data()
    st.session_state.state_id_map = id_map

    # Header
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.title("🔐 Admin Panel")
        st.caption(f"Signed in as **{st.session_state.username}**")
    with header_right:
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.show_login_page = True
            st.rerun()

    st.markdown("---")

    tab_upload, tab_view = st.tabs(["📤 Upload a Document", "📁 View Currently Uploaded Documents"])

    # ------------------------------------------------------
    # TAB 1 — UPLOAD DOCUMENT
    # ------------------------------------------------------
    with tab_upload:
        st.subheader("Upload Documents to Knowledge Base")

        state = st.selectbox("Select State *", [""] + state_names)
        state_id = id_map.get(state) if state else None

        county_id = None
        county_name = None

        counties = get_counties_for_state(state_id) if state_id else []
        if counties:
            county_list = ["(All counties) / None"] + [c["name"] for c in counties]
            selected_county_name = st.selectbox("Optional: Select County", county_list)
            if selected_county_name != "(All counties) / None":
                county_name = selected_county_name
                for c in counties:
                    if c["name"] == selected_county_name:
                        county_id = c["id"]
                        break

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "docx", "doc", "csv", "txt"],
            accept_multiple_files=True,
        )

        if st.button("🚀 Upload & Vectorize", type="primary"):
            if not state_id:
                st.error("Please choose a state.")
                return
            if not uploaded_files:
                st.error("Please upload at least one file.")
                return

            with st.spinner("Processing..."):
                try:
                    files = [
                        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                        for f in uploaded_files
                    ]

                    # upload
                    resp_upload = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        data={"state": state},
                        timeout=300,
                    )
                    if resp_upload.status_code != 200:
                        st.error("Upload failed.")
                        return

                    up_data = resp_upload.json()
                    file_paths = up_data.get("file_paths", [])

                    # vectorize
                    resp_vect = requests.post(
                        f"{BACKEND_URL}/vectorize",
                        json={"state": state, "file_paths": file_paths},
                        timeout=300,
                    )
                    if resp_vect.status_code != 200:
                        st.error("Vectorization failed.")
                        return

                    vect_result = resp_vect.json()
                    st.success("Uploaded & vectorized successfully!")

                    # Save to DB
                    for f, fp in zip(uploaded_files, file_paths):
                        create_document_record(
                            state_id=state_id,
                            county_id=county_id,
                            file_name=f.name,
                            file_path=fp,
                            uploaded_by=st.session_state.username,
                        )

                    with st.expander("📊 Upload Summary", expanded=True):
                        st.write({
                            "state": state,
                            "county": county_name,
                            "file_paths": file_paths,
                            "vector_result": vect_result,
                        })

                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # ------------------------------------------------------
    # TAB 2 — VIEW DOCUMENTS
    # ------------------------------------------------------
    with tab_view:
        st.subheader("Currently Uploaded Documents")

        states_with_docs = get_states_with_documents()
        if not states_with_docs:
            st.info("No documents uploaded yet.")
            return

        state_options = {s["name"]: s["id"] for s in states_with_docs}
        chosen_state = st.selectbox("Choose a state:", list(state_options.keys()))
        chosen_state_id = state_options[chosen_state]

        docs = get_documents_for_state(chosen_state_id)
        if not docs:
            st.info("No documents for this state.")
            return

        # Header
        st.markdown("#### Documents")
        header_cols = st.columns([3, 2, 2, 2, 2, 1.2, 1.2])
        header_cols[0].markdown("**Name**")
        header_cols[1].markdown("**Uploaded By**")
        header_cols[2].markdown("**State**")
        header_cols[3].markdown("**County**")
        header_cols[4].markdown("**Uploaded At**")
        header_cols[5].markdown("**Edit**")
        header_cols[6].markdown("**Delete**")

        # rows
        for d in docs:
            doc_id = d["id"]
            row = st.columns([3, 2, 2, 2, 2, 1.2, 1.2])
            row[0].write(d["file_name"])
            row[1].write(d["uploaded_by"] or "-")
            row[2].write(d["state_name"])
            row[3].write(d["county_name"] or "")
            row[4].write(d["uploaded_at"])

            if row[5].button("✏️", key=f"edit_{doc_id}"):
                st.session_state.editing_doc = d
                st.session_state.confirm_delete_doc_id = None
                st.rerun()

            if row[6].button("🗑️", key=f"delete_{doc_id}"):
                st.session_state.confirm_delete_doc_id = doc_id
                st.session_state.editing_doc = None
                st.rerun()

        # Delete confirmation
        if st.session_state.confirm_delete_doc_id:
            delete_id = st.session_state.confirm_delete_doc_id
            st.warning("Delete this document? This will also remove vectors.")
            c1, c2 = st.columns(2)
            if c1.button("✔ Yes"):
                doc_to_delete = next((x for x in docs if x["id"] == delete_id), None)
                if doc_to_delete:
                    delete_document_record(delete_id)
                    fp = doc_to_delete["file_path"]
                    if fp:
                        delete_from_vector_store(fp)
                st.success("Document deleted.")
                st.session_state.confirm_delete_doc_id = None
                st.rerun()

            if c2.button("✖ No"):
                st.session_state.confirm_delete_doc_id = None
                st.rerun()

        # Edit modal
        if st.session_state.editing_doc:
            d = st.session_state.editing_doc
            st.markdown("---")
            st.markdown("### ✏️ Edit Document")

            doc_id = d["id"]
            current_path = d["file_path"]
            current_county = d.get("county_name")

            counties_list = get_counties_for_state(d["state_id"])
            county_names = ["(leave unchanged)"] + [c["name"] for c in counties_list]
            county_id_map = {c["name"]: c["id"] for c in counties_list}

            new_county_name = st.selectbox(
                "Update County",
                options=county_names,
                index=county_names.index(current_county)
                if current_county in county_names
                else 0,
            )

            new_file = st.file_uploader(
                "Replace file (optional)",
                type=["pdf", "docx", "doc", "csv", "txt"],
            )

            col_s, col_c = st.columns(2)
            if col_s.button("Save Changes"):
                update_payload = {}

                # County update
                if new_county_name != "(leave unchanged)" and new_county_name != current_county:
                    update_payload["county_id"] = county_id_map[new_county_name]
                else:
                    update_payload["county_id"] = d.get("county_id")

                # File update
                if new_file is not None:
                    new_name, new_path = revectorize_document(
                        state=d["state_name"],
                        uploaded_file=new_file,
                        old_file_path=current_path,
                    )
                    if new_name and new_path:
                        update_payload["file_name"] = new_name
                        update_payload["file_path"] = new_path
                    else:
                        st.error("Failed to replace document.")
                        return

                update_document_record(doc_id, update_payload)
                st.success("Updated successfully.")
                st.session_state.editing_doc = None
                st.rerun()

            if col_c.button("Cancel"):
                st.session_state.editing_doc = None
                st.rerun()


# =========================================================
# ROUTER
# =========================================================
if not st.session_state.authenticated:
    show_login_page()
else:
    show_admin_panel()
