"""Request/response models for the RAG endpoints (machine-consumed → validated)."""

from uuid import UUID

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_type: str
    source_uri: str
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: UUID
    chunk_count: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class Source(BaseModel):
    chunk_id: UUID
    document_id: UUID


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
