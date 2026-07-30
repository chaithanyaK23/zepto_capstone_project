"""Pydantic models for the Zepto support assistant API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Incoming POST /ask payload."""

    query: str = Field(..., min_length=1, description="Customer question to answer.")


class AskResponse(BaseModel):
    """Validated JSON response returned by every graph path."""

    answer: str = Field(..., description="Final assistant answer.")
    sources: list[str] = Field(default_factory=list, description="Retrieved chunk or document IDs.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0 to 1.")
