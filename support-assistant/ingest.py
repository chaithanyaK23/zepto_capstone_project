"""Document ingestion, embedding, ChromaDB storage, and retrieval helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
DOCS_DIR = MODULE_DIR / "docs"
CHROMA_DIR = MODULE_DIR / "chroma_store"
COLLECTION_NAME = "zepto_policy_corpus"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


import re

def split_by_sentences(text: str) -> list[str]:
    raw = re.split(r'\.(?=\s|$)', text)
    sentences = [s.strip().strip('"').strip() for s in raw]
    # Append a period back to the sentences since re.split removed them
    return [s + "." for s in sentences if len(s) > 10]

def load_documents() -> list[dict[str, str]]:
    """Load the committed Zepto policy documents and split into overlapping 2-sentence chunks."""
    documents: list[dict[str, str]] = []
    source_count = 0

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        source_count += 1
        text = path.read_text(encoding="utf-8").strip()
        sentences = split_by_sentences(text)
        
        for i in range(len(sentences)):
            chunk_text = " ".join(sentences[i:i+2])
            documents.append(
                {
                    "chunk_id": f"{path.stem}_chunk_{i}",
                    "source": path.name,
                    "text": chunk_text,
                }
            )

    if source_count != 8:
        raise RuntimeError(f"Expected 8 policy documents, found {source_count}.")

    return documents


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load the local sentence-transformers model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create normalized embeddings so Chroma cosine search is stable."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def get_chroma_collection(reset: bool = False) -> Any:
    """Open the persistent Chroma collection used by the assistant."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # Chroma raises if the collection does not exist; that is safe to ignore.
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(reset: bool = True) -> int:
    """Embed all 8 documents and store them in ChromaDB."""
    documents = load_documents()
    collection = get_chroma_collection(reset=reset)

    # Chroma stores the original text, metadata, IDs, and vector embeddings together.
    collection.add(
        ids=[document["chunk_id"] for document in documents],
        documents=[document["text"] for document in documents],
        metadatas=[
            {"source": document["source"], "chunk_id": document["chunk_id"]}
            for document in documents
        ],
        embeddings=embed_texts([document["text"] for document in documents]),
    )

    return collection.count()


def ensure_collection() -> Any:
    """Create the vector index if it has not been built yet."""
    collection = get_chroma_collection(reset=False)
    if collection.count() < 8:
        ingest_documents(reset=True)
        collection = get_chroma_collection(reset=False)
    return collection


def query_collection(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve the top policy chunks for a user query."""
    collection = ensure_collection()
    query_embedding = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        retrieved.append(
            {
                "chunk_id": chunk_id,
                "source": metadata.get("source", chunk_id),
                "text": document,
                "distance": distance,
            }
        )

    return retrieved


def main() -> None:
    """CLI entry point used to build the Chroma index before running the API."""
    count = ingest_documents(reset=True)
    print(f"Embedded and stored {count} Zepto policy chunks in ChromaDB.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Persistence path: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
