from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

import models
import schemas


# ===================
# STATES
# ===================
def get_states(db: Session) -> List[models.State]:
    return db.query(models.State).order_by(models.State.name).all()


def create_state(db: Session, state: schemas.StateCreate) -> models.State:
    db_state = models.State(name=state.name)
    db.add(db_state)
    db.commit()
    db.refresh(db_state)
    return db_state


def get_states_with_documents(db: Session) -> List[models.State]:
    """States that have at least one document."""
    return (
        db.query(models.State)
        .join(models.Document)
        .group_by(models.State.id)
        .order_by(models.State.name)
        .all()
    )


# ===================
# COUNTIES
# ===================
def get_counties_by_state(db: Session, state_id: int) -> List[models.County]:
    return (
        db.query(models.County)
        .filter(models.County.state_id == state_id)
        .order_by(models.County.name)
        .all()
    )


def create_county(db: Session, county: schemas.CountyCreate) -> models.County:
    db_county = models.County(name=county.name, state_id=county.state_id)
    db.add(db_county)
    db.commit()
    db.refresh(db_county)
    return db_county


# ===================
# DOCUMENTS
# ===================
def create_document(db: Session, doc: schemas.DocumentCreate) -> models.Document:
    db_doc = models.Document(**doc.dict())
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


def get_documents_by_state(db: Session, state_id: int) -> List[Dict[str, Any]]:
    """
    Returns list of dicts with state/county names because the Streamlit
    frontend expects: state_name, county_name, uploaded_at, etc.
    """
    q = (
        db.query(
            models.Document,
            models.State.name.label("state_name"),
            models.County.name.label("county_name"),
        )
        .join(models.State, models.Document.state_id == models.State.id)
        .outerjoin(models.County, models.Document.county_id == models.County.id)
        .filter(models.Document.state_id == state_id)
        .order_by(models.Document.uploaded_at.desc())
    )

    results: List[Dict[str, Any]] = []
    for doc, state_name, county_name in q.all():
        results.append(
            {
                "id": doc.id,
                "file_name": doc.file_name,
                "file_path": doc.file_path,
                "uploaded_by": doc.uploaded_by,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "state_id": doc.state_id,
                "county_id": doc.county_id,
                "state_name": state_name,
                "county_name": county_name,
                "vector_status": doc.vector_status,
                "vector_collection": doc.vector_collection,
            }
        )
    return results


def update_document(db: Session, doc_id: int, updates: Dict[str, Any]) -> Optional[models.Document]:
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        return None

    for key, value in updates.items():
        if hasattr(doc, key):
            setattr(doc, key, value)

    db.commit()
    db.refresh(doc)
    return doc


def delete_document(db: Session, doc_id: int) -> bool:
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True
