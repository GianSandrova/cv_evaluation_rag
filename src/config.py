# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]        
HF_CACHE_DIR = str((PROJECT_ROOT / "data" / "hf_cache").resolve())

os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

JOB_ID_DEFAULT = os.getenv("JOB_ID")
COLL_JOBS_CORPUS = "jobs_corpus"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_WORDS = 320
CHUNK_OVERLAP_WORDS = 60

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "eval")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
