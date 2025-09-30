from typing import List, Dict, Any
import os, shutil, sys, time

from src.eval.evaluator import evaluate_candidate_from_files
from src.storage.qdrant_store import close_client
from src.config import UPLOAD_DIR

def run_eval_upload_job(job_id: str, cv_paths: List[str], project_paths: List[str], batch_id: str) -> Dict[str, Any]:
    print(f"[job] start job_id={job_id} batch={batch_id}")
    print(f"[job] cv_paths={cv_paths}")
    print(f"[job] project_paths={project_paths}")
    sys.stdout.flush()

    try:
        t0 = time.time()
        print("[job] evaluating...")
        res = evaluate_candidate_from_files(
            job_id=job_id,
            cv_paths=cv_paths or [],
            project_paths=project_paths or [],
            candidate_id="upload",
        )
        dt = time.time() - t0
        print(f"[job] done in {dt:.1f}s")
        return {"status": "completed", "result": res}
    finally:
        try:
            base = os.path.abspath(os.path.join(UPLOAD_DIR, batch_id))
            print(f"[job] cleanup {base}")
            if base.startswith(os.path.abspath(UPLOAD_DIR)) and os.path.isdir(base):
                shutil.rmtree(base, ignore_errors=True)
        except Exception as e:
            print("[job] cleanup error:", e)
        close_client()
        sys.stdout.flush()
