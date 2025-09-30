# src/queue/worker.py
import os
from redis import Redis
from rq import Queue, Worker
try:
    # SimpleWorker tersedia di RQ >= 1.5
    from rq import SimpleWorker
except Exception:
    SimpleWorker = None

from src.config import REDIS_URL, QUEUE_NAME, HF_CACHE_DIR

def main():
    print(f"[worker] REDIS_URL={REDIS_URL}  QUEUE_NAME={QUEUE_NAME}")
    redis_conn = Redis.from_url(REDIS_URL)
    print(f"[worker] ping={redis_conn.ping()}")
    print(f"[worker] HF_CACHE_DIR={HF_CACHE_DIR}")

    # Warm-up model supaya tidak cold-start di tengah job
    try:
        print("[worker] warming up embedding model...")
        from src.models.embedder import get_model
        _ = get_model()
        print("[worker] warmup done")
    except Exception as e:
        print("[worker] warmup failed:", e)

    q = Queue(QUEUE_NAME, connection=redis_conn)
    print(f"[worker] listening on queues={[q.name]}")

    is_windows = os.name == "nt"
    WorkerClass = SimpleWorker if (is_windows and SimpleWorker is not None) else Worker

    w = WorkerClass(
        [q],
        connection=redis_conn,
        default_worker_ttl=3600,     # heartbeat TTL
        job_monitoring_interval=60,  # update status
    )
    w.work(with_scheduler=False)

if __name__ == "__main__":
    main()
