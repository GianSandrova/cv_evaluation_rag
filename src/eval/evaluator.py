# src/eval/evaluator.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from src.config import COLL_JOBS_CORPUS
from src.llm.groq_client import call_groq
from src.models.embedder import embed_texts
from src.storage.qdrant_store import query_topk
from src.retrieval.memory_index import MemoryIndex, build_index_from_files


# =======================
# Pydantic result schema
# =======================

class EvidenceItem(BaseModel):
    chunk_id: Optional[str] = None
    filename: Optional[str] = None
    snippet: str


class DimScore(BaseModel):
    name: str
    weight: float  # 0..1
    score: float   # 1..5
    rationale: str
    evidence: List[EvidenceItem] = Field(default_factory=list)


class LLMResult(BaseModel):
    # Expected structure from the LLM (we'll coerce/validate)
    cv: Dict[str, Any] = Field(default_factory=dict)        # expects: match_rate (0..1), feedback, dimensions[]
    project: Dict[str, Any] = Field(default_factory=dict)   # expects: feedback, dimensions[]
    overall_summary: str
    risks: List[str] = Field(default_factory=list)


# =======================
# Retrieval helpers
# =======================

def _hits_to_evidence(hits: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert search hits (both Qdrant + MemoryIndex) to evidence items."""
    ev: List[Dict[str, Any]] = []
    if not hits or not hits.get("documents"):
        return ev
    docs = hits["documents"][0]
    mds = hits["metadatas"][0]
    ids = hits.get("ids", [[]])[0] if hits.get("ids") else [None] * len(docs)
    for i, d in enumerate(docs):
        md = mds[i] if i < len(mds) else {}
        ev.append(
            {
                "chunk_id": str(ids[i]) if ids and i < len(ids) else None,
                "filename": md.get("filename"),
                "snippet": (d or "")[:400],
            }
        )
    return ev


def _retrieve(
    job_id: str,
    candidate_id: Optional[str],
    *,
    k_final: int = 8,
    cv_index: Optional[MemoryIndex] = None,
    project_index: Optional[MemoryIndex] = None,
) -> Dict[str, Any]:
    """Retrieve JD, rubric (from Qdrant), and CV/Project evidence (from Qdrant OR in-memory)."""

    def _q(where: Dict[str, Any], query: str) -> Dict[str, Any]:
        qvec = embed_texts([query])[0]
        return query_topk(
            collection_name=COLL_JOBS_CORPUS,
            query_vector=qvec,
            where=where,
            n_results=k_final,
        )

    # Persisted context
    jd_hits = _q(
        {"job_id": job_id, "source_type": "jd"},
        "backend responsibilities llm rag chaining async reliability safeguards",
    )
    rub_cv_hits = _q(
        {"job_id": job_id, "source_type": "rubric", "section": "rubric_cv"},
        "cv match technical skills experience achievements culture collaboration",
    )
    rub_prj_hits = _q(
        {"job_id": job_id, "source_type": "rubric", "section": "rubric_project"},
        "project correctness code quality resilience error handling documentation creativity",
    )

    # Candidate evidence (either ephemeral memory index or persisted)
    if cv_index is not None:
        cv_hits = cv_index.search("skills experience backend databases apis cloud ai llm", k=k_final)
    else:
        assert candidate_id, "candidate_id is required when cv_index is None"
        cv_hits = _q(
            {"job_id": job_id, "source_type": "cv", "candidate_id": candidate_id},
            "skills experience backend databases apis cloud ai llm",
        )

    if project_index is not None:
        prj_hits = project_index.search(
            "prompt design chaining rag retrieval error handling retries randomness readme tests",
            k=k_final,
        )
    else:
        assert candidate_id, "candidate_id is required when project_index is None"
        prj_hits = _q(
            {"job_id": job_id, "source_type": "project", "candidate_id": candidate_id},
            "prompt design chaining rag retrieval error handling retries randomness readme tests",
        )

    job_text = "\n\n".join(jd_hits["documents"][0])[:6000] if jd_hits["documents"][0] else ""
    rubric_cv_text = "\n\n".join(rub_cv_hits["documents"][0])[:4000] if rub_cv_hits["documents"][0] else ""
    rubric_prj_text = "\n\n".join(rub_prj_hits["documents"][0])[:4000] if rub_prj_hits["documents"][0] else ""

    return {
        "job_text": job_text,
        "rubric_cv": rubric_cv_text,
        "rubric_project": rubric_prj_text,
        "cv_evidence": _hits_to_evidence(cv_hits),
        "project_evidence": _hits_to_evidence(prj_hits),
    }


# =======================
# Prompting
# =======================

def _build_messages(ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    SYSTEM = (
        "You are a strict recruitment screening assistant.\n"
        "Use ONLY the provided evidence (Job Description, rubric, candidate CV & project snippets).\n"
        "Score strictly according to the rubric, penalize missing or unclear evidence.\n"
        "Return ONLY valid JSON. Do not include explanations outside JSON."
    )

    # Schema hint to guide the LLM to produce consistent JSON
    schema_hint = {
        "cv": {
            "match_rate": "0..1 (weighted from 1..5 rubric dimensions below)",
            "feedback": "2-4 sentences",
            "dimensions": [
                {"name": "Technical Skills Match", "weight": 0.40, "score": "1..5", "rationale": "...", "evidence": []},
                {"name": "Experience Level", "weight": 0.25, "score": "1..5", "rationale": "...", "evidence": []},
                {"name": "Relevant Achievements", "weight": 0.20, "score": "1..5", "rationale": "...", "evidence": []},
                {"name": "Cultural / Collaboration Fit", "weight": 0.15, "score": "1..5", "rationale": "...", "evidence": []},
            ],
        },
        "project": {
            "feedback": "2-4 sentences",
            "dimensions": [
                {"name": "Correctness (Prompt & Chaining)", "weight": 0.30, "score": "1..5", "rationale": "...", "evidence": []},
                {"name": "Code Quality & Structure", "weight": 0.25, "score": "1..5", "rationale": "...", "evidence": []},
                {"name": "Resilience & Error Handling", "weight": 0.20, "score": "1..5", "rationale": "...", "evidence": []},
                {"name": "Documentation & Explanation", "weight": 0.15, "score": "1..5", "rationale": "...", "evidence": []},
                {"name": "Creativity / Bonus", "weight": 0.10, "score": "1..5", "rationale": "...", "evidence": []},
            ],
        },
        "overall_summary": "3-5 sentences",
        "risks": [],
    }

    USER = {
        "instructions": (
            "Score the candidate strictly against the rubric.\n"
            "For each dimension, provide a score (1..5), rationale, and 1-3 evidence snippets.\n"
            "Use only the supplied evidence; if missing, score low and mention it.\n"
            "Output JSON only."
        ),
        "job_description": ctx.get("job_text", ""),
        "rubric_cv": ctx.get("rubric_cv", ""),
        "rubric_project": ctx.get("rubric_project", ""),
        "cv_evidence": ctx.get("cv_evidence", []),
        "project_evidence": ctx.get("project_evidence", []),
        "output_schema": schema_hint,
    }

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(USER, ensure_ascii=False)},
    ]


# =======================
# Aggregation
# =======================

def _weighted_avg(items: List[Dict[str, Any]]) -> float:
    num = sum((float(i.get("score", 0) or 0) * float(i.get("weight", 0) or 0)) for i in items)
    den = sum((float(i.get("weight", 0) or 0)) for i in items) or 1.0
    return num / den


def _eval_with_ctx(ctx: Dict[str, Any]) -> Dict[str, Any]:
    messages = _build_messages(ctx)

    # Call Groq in JSON mode, fallback once without strict JSON mode if validation fails
    raw = call_groq(messages, json_mode=True)
    try:
        obj = LLMResult.model_validate_json(raw)
    except ValidationError:
        raw2 = call_groq(messages, json_mode=False)
        obj = LLMResult.model_validate_json(raw2)

    cv_dims = obj.cv.get("dimensions", []) if isinstance(obj.cv, dict) else []
    prj_dims = obj.project.get("dimensions", []) if isinstance(obj.project, dict) else []

    cv_avg_5pt = _weighted_avg(cv_dims) if cv_dims else 0.0
    cv_match_rate = round(cv_avg_5pt / 5.0, 4)
    project_score = round(_weighted_avg(prj_dims), 2) if prj_dims else 0.0

    # Simple decision heuristic (tweak as needed)
    if cv_match_rate >= 0.75 and project_score >= 4.0:
        decision = "advance"
    elif cv_match_rate >= 0.55 and project_score >= 3.2:
        decision = "review"
    else:
        decision = "reject"

    return {
        "cv_match_rate": cv_match_rate,
        "cv_feedback": obj.cv.get("feedback", "") if isinstance(obj.cv, dict) else "",
        "project_score": project_score,
        "project_feedback": obj.project.get("feedback", "") if isinstance(obj.project, dict) else "",
        "overall_summary": obj.overall_summary,
        "details": {
            "cv_dimensions": cv_dims,
            "project_dimensions": prj_dims,
            "risks": obj.risks,
        },
        "decision": decision,
    }


# =======================
# Public API (two modes)
# =======================

def evaluate_candidate(job_id: str, candidate_id: str) -> Dict[str, Any]:
    """
    PERSISTENT mode:
    - JD & rubric from Qdrant
    - CV & Project are also expected in Qdrant under the given candidate_id
    """
    ctx = _retrieve(job_id, candidate_id, k_final=8, cv_index=None, project_index=None)
    return _eval_with_ctx(ctx)


def evaluate_candidate_from_files(
    job_id: str,
    cv_paths: List[str],
    project_paths: List[str],
    *,
    candidate_id: str = "upload",  # label only; not persisted
) -> Dict[str, Any]:
    """
    EPHEMERAL mode (recommended for privacy during upload):
    - JD & rubric from Qdrant
    - CV & Project are read from files, embedded & queried in memory (NOT saved)
    """
    cv_idx = build_index_from_files(cv_paths, job_id, candidate_id, source_type="cv")
    prj_idx = build_index_from_files(project_paths, job_id, candidate_id, source_type="project")
    ctx = _retrieve(job_id, candidate_id=None, k_final=8, cv_index=cv_idx, project_index=prj_idx)
    return _eval_with_ctx(ctx)
