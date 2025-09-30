# src/api/app.py
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from redis import Redis
from rq import Queue
from rq.job import Job

from src.config import REDIS_URL, QUEUE_NAME, JOB_ID_DEFAULT, UPLOAD_DIR
from src.utils.uploads import save_uploads, new_batch_id, list_batch_paths
from src.queue.jobs import run_eval_upload_job

app = FastAPI(title="AI Screening API", version="0.4.0")

# ---------- helpers ----------
PUBLIC_RESULT_KEYS = [
    "cv_match_rate", "cv_feedback",
    "project_score", "project_feedback",
    "overall_summary",
]

def public_result_view(res: Dict[str, Any]) -> Dict[str, Any]:
    return {k: res.get(k) for k in PUBLIC_RESULT_KEYS if k in res}

def get_redis() -> Redis:
    return Redis.from_url(REDIS_URL)

def get_queue(redis_conn: Redis | None = None) -> Queue:
    if redis_conn is None:
        redis_conn = get_redis()
    return Queue(QUEUE_NAME, connection=redis_conn)

# ---------- health ----------
@app.get("/health")
def health():
    try:
        ok = get_redis().ping()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis not reachable: {e}")
    return {"ok": True, "redis": ok}

# ---------- 1) POST /upload ----------
@app.post("/upload")
async def upload_files(
    cv_files: List[UploadFile] = File(default=[]),
    project_files: List[UploadFile] = File(default=[]),
):
    if not cv_files and not project_files:
        raise HTTPException(status_code=400, detail="Upload at least one CV or Project file")
    batch_id = new_batch_id()
    saved_cv = save_uploads(cv_files, batch_id, "cv") if cv_files else []
    saved_pr = save_uploads(project_files, batch_id, "project") if project_files else []
    return JSONResponse({
        "batch_id": batch_id,
        "files": {"cv": [str(p) for p in saved_cv], "project": [str(p) for p in saved_pr]}
    })

# ---------- 2) POST /evaluate (async via RQ) ----------
class EvaluateRequest(BaseModel):
    job_id: str = JOB_ID_DEFAULT
    batch_id: str

@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    import os
    cv_paths, pr_paths = list_batch_paths(req.batch_id)
    # debug ringan
    print(f"[api] evaluate batch={req.batch_id}")
    print(f"[api] UPLOAD_DIR={UPLOAD_DIR}")
    print(f"[api] cv_paths={cv_paths} exists={[os.path.exists(p) for p in cv_paths]}")
    print(f"[api] pr_paths={pr_paths}  exists={[os.path.exists(p) for p in pr_paths]}")

    if not cv_paths and not pr_paths:
        raise HTTPException(status_code=404, detail="No uploaded files found for batch_id")

    q = get_queue()
    job = q.enqueue(
        run_eval_upload_job,
        req.job_id, cv_paths, pr_paths, req.batch_id,
        job_timeout=1800,   # 30 menit aman utk cold start
    )
    print(f"[api] enqueue -> id={job.get_id()}")
    return JSONResponse({"id": job.get_id(), "status": "queued"})

# ---------- 3) GET /result/{id} ----------
@app.get("/result/{task_id}")
def get_result(task_id: str):
    redis_conn = get_redis()
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Task not found")

    status_map = {
        "queued": "queued",
        "started": "processing",
        "deferred": "queued",
        "finished": "completed",
        "failed": "failed",
        None: "unknown",
    }
    status = status_map.get(job.get_status(), "unknown")
    payload: Dict[str, Any] = {"id": task_id, "status": status}

    if status == "completed":
        res = job.result
        if isinstance(res, dict) and "result" in res and isinstance(res["result"], dict):
            payload["result"] = public_result_view(res["result"])
        else:
            payload["result"] = res
    elif status == "failed":
        payload["error"] = str(job.exc_info or "")[:2000]

    return JSONResponse(payload)
