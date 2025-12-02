from typing import Optional

from pydantic import BaseModel


# ===== STATES =====
class StateBase(BaseModel):
    name: str


class StateCreate(StateBase):
    pass


class State(StateBase):
    id: int

    class Config:
        orm_mode = True


# ===== COUNTIES =====
class CountyBase(BaseModel):
    name: str
    state_id: int


class CountyCreate(CountyBase):
    pass


class County(CountyBase):
    id: int

    class Config:
        orm_mode = True


# ===== DOCUMENTS =====
class DocumentBase(BaseModel):
    file_name: str
    file_path: str
    state_id: int
    county_id: Optional[int] = None
    uploaded_by: Optional[str] = None
    vector_status: Optional[str] = None
    vector_collection: Optional[str] = None


class DocumentCreate(DocumentBase):
    pass


class Document(DocumentBase):
    id: int

    class Config:
        orm_mode = True


class DocumentWithNames(BaseModel):
    id: int
    file_name: str
    file_path: str
    uploaded_by: Optional[str]
    uploaded_at: str
    state_id: int
    county_id: Optional[int]
    state_name: str
    county_name: Optional[str]

    class Config:
        orm_mode = False
