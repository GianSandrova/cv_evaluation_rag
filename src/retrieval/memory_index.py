# src/retrieval/memory_index.py
from __future__ import annotations
import uuid
from typing import List, Dict, Any, Optional
import numpy as np

from src.io.loaders import load_text_from_file
from src.processing.normalizer import normalize_text
from src.processing.chunker import chunk_by_words
from src.models.embedder import embed_texts
from src.config import CHUNK_WORDS, CHUNK_OVERLAP_WORDS

class MemoryIndex:
    """Simple in-memory vector index (cosine) for ephemeral use."""
    def __init__(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        self.documents = documents
        self.metadatas = metadatas
        if documents:
            self.embeddings = np.array(embed_texts(documents), dtype=np.float32)
        else:
            self.embeddings = np.empty((0, 0), dtype=np.float32)

    def search(self, query_text: str, k: int = 5):
        if len(self.documents) == 0:
            return {"documents":[[]], "metadatas":[[]], "distances":[[]], "ids":[[]]}
        q = np.array(embed_texts([query_text])[0], dtype=np.float32)  # normalized
        sims = self.embeddings @ q                                   # cosine similarity
        topk = min(k, sims.shape[0])
        idx = np.argsort(-sims)[:topk]
        docs = [self.documents[i] for i in idx]
        mds  = [self.metadatas[i] for i in idx]
        dists = [1.0 - float(sims[i]) for i in idx]                  # distance ~ 1 - sim
        ids  = [mds[i]["id"] for i in range(len(mds))]
        return {"documents":[docs], "metadatas":[mds], "distances":[dists], "ids":[ids]}

def build_index_from_files(
    paths: List[str],
    job_id: str,
    candidate_id: str,
    source_type: str,                         # "cv" | "project"
    lang_hint: Optional[str] = None,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> MemoryIndex:
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for p in paths:
        raw = load_text_from_file(p)
        norm = normalize_text(raw, mask_pii_flag=True)
        chunks = chunk_by_words(norm, chunk_words, overlap_words)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            metas.append({
                "id": str(uuid.uuid4()),
                "job_id": job_id,
                "candidate_id": candidate_id,
                "source_type": source_type,
                "filename": p.split("/")[-1].split("\\")[-1],
                "chunk_idx": i,
                **({"lang": lang_hint} if lang_hint else {})
            })
    return MemoryIndex(docs, metas)
