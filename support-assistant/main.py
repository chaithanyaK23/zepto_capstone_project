"""FastAPI wrapper for the Zepto support assistant LangGraph."""

from __future__ import annotations

from fastapi import FastAPI

from ingest import ensure_collection
from rag_graph import answer_query
from schemas import AskRequest, AskResponse


app = FastAPI(title="Zepto Support Assistant", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    """Build or open the local Chroma collection when the API starts."""
    ensure_collection()


@app.get("/")
def health_check() -> dict[str, str]:
    """Small health route for local smoke testing."""
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Accept a user query and return the validated graph response."""
    return answer_query(request.query)
