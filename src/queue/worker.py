# src/queue/worker.py
import os
from pathlib import Path
from redis import Redis
from rq import Queue, Worker
try:
    from rq import SimpleWorker  # prefer on Windows (no fork)
except Exception:
    SimpleWorker = None

# Muat .env paling awal supaya flag logging/LLM kebaca
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.config import REDIS_URL, QUEUE_NAME, HF_CACHE_DIR
from src.utils.logs import setup_logging

# Kurangi warning tokenizer
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _setup_hf_cache(logger):
    """Ensure HuggingFace cache directories are set & exist."""
    if HF_CACHE_DIR:
        cache_path = Path(HF_CACHE_DIR)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_path))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_path))
        logger.info("[worker] HF cache at: %s", cache_path.resolve())
    else:
        logger.info("[worker] HF_CACHE_DIR not set (using default HF cache).")


def main():
    logger = setup_logging("worker")
    logger.info("[worker] LOG_LEVEL=%s EVAL_LOG=%s LOG_LLM_RAW=%s",
                os.getenv("LOG_LEVEL", "INFO"),
                os.getenv("EVAL_LOG", "0"),
                os.getenv("LOG_LLM_RAW", "0"))

    _setup_hf_cache(logger)

    # Connect Redis
    logger.info("[worker] REDIS_URL=%s  QUEUE_NAME=%s", REDIS_URL, QUEUE_NAME)
    redis_conn = Redis.from_url(REDIS_URL)
    logger.info("[worker] redis ping=%s", redis_conn.ping())

    # Warm-up embedding model
    try:
        logger.info("[worker] warming up embedding model...")
        from src.models.embedder import get_model
        _ = get_model()
        logger.info("[worker] warmup done")
    except Exception as e:
        logger.exception("[worker] warmup failed: %s", e)

    q = Queue(QUEUE_NAME, connection=redis_conn)
    logger.info("*** Listening on %s ...", q.name)

    is_windows = os.name == "nt"
    WorkerClass = SimpleWorker if (is_windows and SimpleWorker is not None) else Worker

    w = WorkerClass(
        [q],
        connection=redis_conn,
        default_worker_ttl=3600,
        job_monitoring_interval=60,
    )
    try:
        w.work(with_scheduler=False)
    except Exception as e:
        logger.exception("[worker] fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
