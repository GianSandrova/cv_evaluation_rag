# src/models/embedder.py
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, HF_CACHE_DIR

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=HF_CACHE_DIR,
            # local_files_only=True,  # aktifkan kalau cache sudah lengkap
        )
    return _model

def embed_texts(texts: List[str]):
    m = get_model()
    embs = m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() if hasattr(e, "tolist") else e for e in embs]
