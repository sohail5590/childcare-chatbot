from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
import crud
import schemas
import models

# Auto-create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Childcare DB API")

# Allow Streamlit + any other service to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# ========= STATES =========
@app.get("/states", response_model=List[schemas.State])
def list_states(db: Session = Depends(get_db)):
    return crud.get_states(db)


@app.post("/states", response_model=schemas.State)
def add_state(state: schemas.StateCreate, db: Session = Depends(get_db)):
    return crud.create_state(db, state)


@app.get("/states/with-documents", response_model=List[schemas.State])
def states_with_documents(db: Session = Depends(get_db)):
    return crud.get_states_with_documents(db)


# ========= COUNTIES =========
@app.get("/counties", response_model=List[schemas.County])
def list_counties(state_id: int = Query(...), db: Session = Depends(get_db)):
    return crud.get_counties_by_state(db, state_id)


@app.post("/counties", response_model=schemas.County)
def add_county(county: schemas.CountyCreate, db: Session = Depends(get_db)):
    return crud.create_county(db, county)


# ========= DOCUMENTS =========
@app.post("/documents", response_model=schemas.Document)
def add_document(doc: schemas.DocumentCreate, db: Session = Depends(get_db)):
    return crud.create_document(db, doc)


@app.get("/documents")
def list_documents(state_id: int = Query(...), db: Session = Depends(get_db)):
    """Return documents for a state with state_name/county_name included."""
    return crud.get_documents_by_state(db, state_id)


@app.put("/documents/{doc_id}", response_model=schemas.Document)
def update_document(
    doc_id: int,
    patch: Dict[str, Any],
    db: Session = Depends(get_db),
):
    updated = crud.update_document(db, doc_id, patch)
    if not updated:
        raise HTTPException(status_code=404, detail="Document not found")
    return updated


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    ok = crud.delete_document(db, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True}
