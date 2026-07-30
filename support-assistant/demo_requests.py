"""Run two default mock-mode example calls and print raw JSON responses."""

from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

from main import app


def main() -> None:
    """Demonstrate one policy query and one general query in mock mode."""
    os.environ.setdefault("MOCK_LLM", "1")
    client = TestClient(app)

    examples = [
        {"query": "What is Zepto's delivery fee for small orders?"},
        {"query": "Who won the football match yesterday?"},
    ]

    for payload in examples:
        response = client.post("/ask", json=payload)
        print(f"Request: {json.dumps(payload)}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()


if __name__ == "__main__":
    main()
