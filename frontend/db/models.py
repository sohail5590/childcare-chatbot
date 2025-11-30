from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Text,
)
from sqlalchemy.orm import relationship

from database import Base


class State(Base):
    __tablename__ = "states"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)

    counties = relationship("County", back_populates="state", cascade="all, delete")
    documents = relationship("Document", back_populates="state", cascade="all, delete")


class County(Base):
    __tablename__ = "counties"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)

    state_id = Column(Integer, ForeignKey("states.id"), nullable=False)
    state = relationship("State", back_populates="counties")

    documents = relationship("Document", back_populates="county", cascade="all, delete")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)

    # File metadata
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    uploaded_by = Column(String(255), nullable=True)

    # Vectorization metadata
    vector_status = Column(String(50), nullable=True)
    vector_collection = Column(String(255), nullable=True)

    # Foreign keys
    state_id = Column(Integer, ForeignKey("states.id"), nullable=False)
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=True)

    state = relationship("State", back_populates="documents")
    county = relationship("County", back_populates="documents")
