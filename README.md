# AI Screening Backend — FastAPI

Backend sederhana untuk **screening kandidat**: upload CV/Project → dibandingkan dengan JD & Rubric → hasil evaluasi terstruktur (JSON) via job asinkron.

---

## 1) Prasyarat

* Python 3.10–3.12
* Redis berjalan lokal (WSL/Docker/Windows Service). Cek: `redis-cli ping` → `PONG`.
* Groq API key.

> **Catatan:** Qdrant berjalan **embedded** (akses folder lokal). Embedded **tidak mendukung multi-proses** pada storage yang sama. Desain di sini memastikan **API tidak menyentuh Qdrant**, hanya **worker** & **script ingest** (jalankan **tidak bersamaan**).

---

## 2) Setup Cepat

1. **Install dependency**

   ```bash
   pip install -r requirements.txt
   ```

2. **Buat `.env` di root**

   ```env
   # Groq
   GROQ_API_KEY=YOUR_GROQ_KEY
   GROQ_MODEL=llama-3.1-8b-instant

   # Redis
   REDIS_URL=redis://localhost:6379/0
   QUEUE_NAME=eval

   # Upload
   UPLOAD_DIR=./data/uploads

   # Default Job
   JOB_ID=backend-01

   # Debug (opsional)
   # KEEP_UPLOADS=1  # jangan hapus file upload setelah job (untuk debugging)
   ```

3. **Siapkan cache model (opsional tapi disarankan)** — agar tidak download saat pertama jalan

   ```bash
   python -c "from src.models.embedder import get_model; get_model(); print('model cached')"
   ```

> Project otomatis memakai cache HF di `data/hf_cache` (lihat `src/config.py`). Setelah cache siap dan stabil, Anda dapat mengaktifkan mode offline dengan mengatur `HF_LOCAL_ONLY=1` di `src/config.py` (opsional).

---

## 3) Ingest Data (JD & Rubric)

Letakkan file sumber di:

```
data/raw/jd.pdf
data/raw/rubrik.pdf
```

Jalankan **saat worker TIDAK aktif**:

```bash
# Job Description (section bebas; minimal satu)
python -m scripts.ingest_jd_rubric --source-type jd --section overview --paths data/raw/jd.pdf

# Rubric CV
python -m scripts.ingest_jd_rubric --source-type rubric --section rubric_cv --paths data/raw/rubrik.pdf

# Rubric Project
python -m scripts.ingest_jd_rubric --source-type rubric --section rubric_project --paths data/raw/rubrik.pdf
```

---

## 4) Menjalankan

**Worker** (Windows otomatis pakai SimpleWorker):

```bash
python -m src.queue.worker
```

Log yang diharapkan:

```
[worker] warming up embedding model...
[worker] warmup done
*** Listening on eval...
```

**API**:

```bash
uvicorn src.api.app:app --reload
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 5) Alur API

### (1) Upload

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "cv_files=@data/raw/CV_Gian.pdf" \
  -F "project_files=@data/raw/CV_Gian.pdf"
```

Response → simpan `batch_id`.

### (2) Evaluate (enqueue)

```bash
curl -X POST "http://127.0.0.1:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d "{\"job_id\":\"backend-01\", \"batch_id\":\"<batch_id>\"}"
```

Response → `{ "id": "<job-id>", "status": "queued" }`.

### (3) Result (polling)

```bash
curl "http://127.0.0.1:8000/result/<job-id>"
```

Contoh hasil selesai:

```json
{
  "id": "<job-id>",
  "status": "completed",
  "result": {
    "cv_match_rate": 0.62,
    "cv_feedback": "...",
    "project_score": 3.4,
    "project_feedback": "...",
    "overall_summary": "..."
  }
}
```

Status lain: `queued`, `processing`, `failed` (lihat field `error`).

---

## 6) Struktur Proyek (ringkas)

```
src/
  api/app.py            # FastAPI endpoints (/upload, /evaluate, /result)
  queue/worker.py       # RQ worker (SimpleWorker di Windows)
  queue/jobs.py         # Job evaluator
  eval/evaluator.py     # Orkestrasi retrieval + LLM scoring
  storage/qdrant_store.py # Qdrant embedded client
  models/embedder.py    # Embedding model (Qwen)
  llm/groq_client.py    # Groq chat completions
  io/loaders.py         # Loader PDF/DOCX/TXT
  processing/*          # Normalizer + chunker
  retrieval/memory_index.py # Ephemeral index untuk upload kandidat
  config.py             # Konfigurasi & HF cache
scripts/
  ingest_jd_rubric.py   # Ingest JD & Rubric ke Qdrant
```

---

## 7) Troubleshooting Singkat

* **`queued` lama** → worker belum jalan / tidak dengar queue `eval` / `REDIS_URL` berbeda. Cek log worker.
* **`AbandonedJobError`** → worker mati atau heartbeat habis. Pastikan SimpleWorker aktif & `job_timeout` cukup (sudah 1800s).
* **`FileNotFoundError` saat evaluate** → `batch_id` salah, upload dibersihkan, atau path beda. Untuk debug: set `.env` `KEEP_UPLOADS=1`.
* **`Storage folder ... already accessed` (Qdrant)** → ingest & worker jalan bersamaan. Matikan worker saat ingest **atau** gunakan Qdrant Server (lihat di bawah).
* **Model “download lagi”** → pastikan cache `data/hf_cache` dipakai semua proses (lihat log). Pre-warm sekali seperti langkah setup.

---

## 8) (Opsional) Qdrant Server

Jika perlu akses paralel (ingest sambil worker jalan), jalankan Qdrant sebagai server:

```bash
docker run -p 6333:6333 -v "${PWD}\data\qdrant_srv:/qdrant/storage" qdrant/qdrant:latest
```

Tambahkan di `.env`:

```env
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=... (opsional)
```

Kemdian **re-ingest** JD & Rubric karena storage berbeda.

---

## 9) Lisensi

MIT (opsional)

```
```
