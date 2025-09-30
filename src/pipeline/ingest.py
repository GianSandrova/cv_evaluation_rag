import hashlib
import os
import time
import uuid
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from src.processing.chunker import chunk_by_words, split_by_headings


from src.config import (
    COLL_JOBS_CORPUS,
    CHUNK_WORDS,
    CHUNK_OVERLAP_WORDS,
)
from src.io.loaders import load_text_from_file
from src.processing.normalizer import normalize_text
from src.models.embedder import embed_texts
from src.storage.qdrant_store import add_documents

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_metadatas(
    job_id: str,
    source_type: str,
    filename: str,
    section: Optional[str],
    chunk_count: int,
    extra: Optional[Dict] = None,
):
    base = extra.copy() if extra else {}
    mds = []
    ts = time.time()
    for i in range(chunk_count):
        md = {
            "job_id": job_id,
            "source_type": source_type,   # "jd" | "rubric"
            "filename": os.path.basename(filename),
            "chunk_idx": i,
            "ts": ts,
        }
        if section:
            md["section"] = section
        md.update(base)
        mds.append(md)
    return mds

def ingest_single_file(
    file_path: str,
    job_id: str,
    source_type: str,           # "jd" | "rubric"
    section: Optional[str] = None,  # e.g., "overview", "about_the_job", or "rubric_cv"
    collection_name: str = COLL_JOBS_CORPUS,
    mask_pii: bool = True,
) -> Tuple[int, List[str]]:
    """
    Returns: (n_chunks, ids)
    """
    raw = load_text_from_file(file_path)
    norm = normalize_text(raw, mask_pii_flag=mask_pii)

    # chunking
    chunks = chunk_by_words(norm, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)
    if not chunks:
        return 0, []

    # embeddings
    embs = embed_texts(chunks)

    # ids
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # metadatas
    sha = file_sha256(file_path)
    metas = build_metadatas(
        job_id=job_id,
        source_type=source_type,
        filename=file_path,
        section=section,
        chunk_count=len(chunks),
        extra={"sha256": sha, "lang": "en"},  # JD kamu english; ubah kalau perlu
    )

    # write to chroma
    add_documents(collection_name, chunks, metas, ids=ids, embeddings=embs)
    return len(chunks), ids


def ingest_jd_auto_sections(
    file_path: str,
    job_id: str,
    collection_name: str = COLL_JOBS_CORPUS,
    mask_pii: bool = True,
):
    raw = load_text_from_file(file_path)
    norm = normalize_text(raw, mask_pii_flag=mask_pii)

    sections = split_by_headings(norm)  # -> [(section_key, text), ...]
    total, all_ids = 0, []
    sha = file_sha256(file_path)

    for section_key, sec_text in sections:
        chunks = chunk_by_words(sec_text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)
        if not chunks:
            continue
        embs = embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metas = build_metadatas(
            job_id=job_id,
            source_type="jd",
            filename=file_path,
            section=section_key,
            chunk_count=len(chunks),
            extra={"sha256": sha, "lang": "en"},
        )
        add_documents(collection_name, chunks, metas, ids=ids, embeddings=embs)
        total += len(chunks)
        all_ids.extend(ids)

    return total, all_ids

def ingest_batch(
    paths: List[str],
    job_id: str,
    source_type: str,
    section: Optional[str] = None,
    collection_name: str = COLL_JOBS_CORPUS,
    mask_pii: bool = True,
):
    total_chunks = 0
    all_ids: List[str] = []
    for p in tqdm(paths, desc=f"Ingest {source_type}/{section or ''}"):
        n, ids = ingest_single_file(
            file_path=p,
            job_id=job_id,
            source_type=source_type,
            section=section,
            collection_name=collection_name,
            mask_pii=mask_pii,
        )
        total_chunks += n
        all_ids.extend(ids)
    return {"chunks": total_chunks, "ids": all_ids, "collection": collection_name}
