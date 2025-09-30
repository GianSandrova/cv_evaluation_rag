from typing import List, Dict, Any, Optional
import os, uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from src.models.embedder import get_model

_client: Optional[QdrantClient] = None
def get_client() -> QdrantClient:
    global _client
    if _client is None:
        db_path = os.path.abspath("data/qdrant")
        os.makedirs(db_path, exist_ok=True)
        # Local mode: penyimpanan di folder, tanpa server/Docker
        _client = QdrantClient(path=db_path)
    return _client

def ensure_collection(name: str):
    client = get_client()
    dim = get_model().get_sentence_embedding_dimension()
    try:
        client.get_collection(name)
    except Exception:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
    embeddings: Optional[List[List[float]]] = None,
):
    ensure_collection(collection_name)
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]
    points = [
        PointStruct(id=ids[i], vector=embeddings[i], payload={
            **metadatas[i], "document": documents[i]
        })
        for i in range(len(documents))
    ]
    client = get_client()
    client.upsert(collection_name=collection_name, points=points)
    return True

def query_topk(
    collection_name: str,
    query_vector: List[float],
    where: Optional[Dict[str, Any]] = None,
    n_results: int = 5,
):
    ensure_collection(collection_name)
    client = get_client()
    flt = None
    if where:
        # sederhana: semua kondisi exact match sebagai must
        flt = Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in where.items()
        ])
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=n_results,
        query_filter=flt,
        with_payload=True,
    )
    # samakan bentuk return agar mirip Chroma
    documents = [[(h.payload.get("document") or "") for h in hits]]
    metadatas = [[{k: v for k, v in h.payload.items() if k != "document"} for h in hits]]
    distances = [[1.0 - (h.score or 0.0) for h in hits]]  # score=cosine sim → jarak ~ 1-sim
    ids = [[h.id for h in hits]]
    return {"documents": documents, "metadatas": metadatas, "distances": distances, "ids": ids}

def close_client():
    """Close Qdrant local client gracefully (avoid shutdown noise on Windows)."""
    global _client
    if _client is not None:
        try:
            _client.close()  # releases portalocker
        except Exception:
            pass
        _client = None

# Tutup lebih awal saat proses berakhir (sebelum __del__ dipanggil)
import atexit
atexit.register(close_client)