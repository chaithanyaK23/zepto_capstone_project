"""LangGraph workflow for routing, retrieval, and structured mock answers."""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

import requests
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from ingest import query_collection
from prompt_template import build_prompt
from schemas import AskResponse


Intent = Literal["policy_question", "general_question"]

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

GENERAL_MOCK_ANSWER = "I can only answer questions about Zepto policies right now."


class SupportState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""

    query: str
    intent: Intent
    retrieved_chunks: list[dict[str, Any]]
    response: dict[str, Any]


def is_mock_mode() -> bool:
    """Default to deterministic mock mode unless MOCK_LLM=0 is explicitly set."""
    return os.getenv("MOCK_LLM", "1") != "0"


def keyword_intent(query: str) -> Intent:
    """Classify policy questions with the required keyword heuristic."""
    lowered_query = query.lower()
    if any(keyword in lowered_query for keyword in POLICY_KEYWORDS):
        return "policy_question"
    return "general_question"


def classify_with_optional_llm(query: str) -> Intent:
    """Optional real-LLM classification path; falls back safely if no key is set."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return keyword_intent(query)

    prompt = (
        "Classify the query as exactly one label: policy_question or general_question.\n"
        f"Query: {query}"
    )
    try:
        raw_text = call_groq_text(prompt, api_key)
    except Exception:
        return keyword_intent(query)

    return "policy_question" if "policy_question" in raw_text.lower() else "general_question"


def call_groq_text(prompt: str, api_key: str) -> str:
    """Minimal optional Groq call used only when MOCK_LLM=0 and a key exists."""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def validate_llm_json_with_retries(query: str, context: str, sources: list[str]) -> AskResponse:
    """Try to validate optional LLM JSON up to 3 total attempts."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return AskResponse(
            answer="MOCK_LLM=0 was requested, but no GROQ_API_KEY is configured.",
            sources=sources,
            confidence=0.0,
        )

    prompt = build_prompt(query=query, context=context)
    last_error = ""
    for attempt in range(3):
        corrective_note = "" if attempt == 0 else f"\nFix this JSON validation error: {last_error}"
        raw_text = call_groq_text(prompt + corrective_note, api_key)
        try:
            parsed = json.loads(raw_text)
            return AskResponse.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)

    return AskResponse(
        answer=f"Unable to produce a valid structured response after retries: {last_error}",
        sources=sources,
        confidence=0.0,
    )


def classify_intent(state: SupportState) -> SupportState:
    """LangGraph node 1: route the query to retrieval or direct answer."""
    query = state["query"]
    state["intent"] = keyword_intent(query) if is_mock_mode() else classify_with_optional_llm(query)
    return state


def retrieve_and_answer(state: SupportState) -> SupportState:
    """LangGraph node 2: retrieve top-3 policy chunks and generate an answer."""
    query = state["query"]
    chunks = query_collection(query, top_k=3)
    state["retrieved_chunks"] = chunks
    sources = [chunk["chunk_id"] for chunk in chunks]

    if is_mock_mode():
        # Each chunk is now a complete sentence, so the top result is a
        # clean, self-contained answer — no joining of fragments needed.
        top_sentence = chunks[0]["text"] if chunks else ""
        response = AskResponse(
            answer=f"Based on the retrieved context: {top_sentence}",
            sources=sources,
            confidence=1.0,
        )
    else:
        context = "\n\n".join(f"{chunk['chunk_id']}: {chunk['text']}" for chunk in chunks)
        response = validate_llm_json_with_retries(query=query, context=context, sources=sources)

    state["response"] = response.model_dump()
    return state


def direct_answer(state: SupportState) -> SupportState:
    """LangGraph node 3: answer general questions without retrieval."""
    if is_mock_mode():
        response = AskResponse(answer=GENERAL_MOCK_ANSWER, sources=[], confidence=1.0)
    else:
        response = validate_llm_json_with_retries(
            query=state["query"],
            context="No retrieval context was used because this is a general question.",
            sources=[],
        )

    state["response"] = response.model_dump()
    return state


def route_after_classification(state: SupportState) -> str:
    """Conditional edge selector used by LangGraph."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph() -> Any:
    """Compile the StateGraph with the 3 required nodes and conditional edge."""
    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


compiled_graph = build_graph()


def answer_query(query: str) -> AskResponse:
    """Run the graph and return a validated response model."""
    result = compiled_graph.invoke({"query": query})
    return AskResponse.model_validate(result["response"])
