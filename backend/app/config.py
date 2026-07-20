"""
Central configuration: chat/vision LLMs, embeddings, and the Qdrant vector store.
Ported from Agentic_RAG_ready.ipynb (cell 6) so the API serves the exact same
retrieval + generation stack that was validated in the notebook.
"""
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

# Embedded Qdrant by default (no server needed, data persisted to disk).
# Set QDRANT_URL to point at a real Qdrant server/cluster instead (recommended
# for HF Spaces, since the container disk is not guaranteed to persist across
# restarts) — e.g. a free Qdrant Cloud cluster.
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip() or None
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_db")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "agentic_rag_openai")

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)


def get_vision_llm() -> ChatOpenAI:
    return ChatOpenAI(model=VISION_MODEL, temperature=0.0)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _make_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=QDRANT_PATH)


_qdrant_client = _make_client()

_existing = [c.name for c in _qdrant_client.get_collections().collections]
if QDRANT_COLLECTION not in _existing:
    _qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

vectorstore = QdrantVectorStore(
    client=_qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embedding=get_embeddings(),
)
