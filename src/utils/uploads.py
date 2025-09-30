# src/utils/uploads.py
import os
from typing import List, Tuple
from uuid import uuid4
from fastapi import UploadFile
from src.config import UPLOAD_DIR

def new_batch_id() -> str:
    return str(uuid4())

def _abs(*parts: str) -> str:
    return os.path.abspath(os.path.join(*parts))

def save_uploads(files: List[UploadFile], batch_id: str, subdir: str) -> List[str]:
    """
    Simpan file ke: UPLOAD_DIR/<batch_id>/<subdir>/<filename>
    Return: list path absolut.
    """
    base = _abs(UPLOAD_DIR, batch_id, subdir)
    os.makedirs(base, exist_ok=True)
    paths: List[str] = []
    for f in files:
        name = os.path.basename(f.filename or "upload.bin")
        dst = _abs(base, name)
        with open(dst, "wb") as out:
            out.write(f.file.read())
        paths.append(dst)
    return paths

def list_batch_paths(batch_id: str) -> Tuple[List[str], List[str]]:
    """
    Ambil semua file yang sudah di-upload untuk batch tertentu.
    Mengembalikan (cv_paths, project_paths).
    """
    cv_dir = _abs(UPLOAD_DIR, batch_id, "cv")
    pr_dir = _abs(UPLOAD_DIR, batch_id, "project")

    def _ls(d: str) -> List[str]:
        if not os.path.isdir(d):
            return []
        return [_abs(d, f) for f in os.listdir(d) if os.path.isfile(_abs(d, f))]

    return _ls(cv_dir), _ls(pr_dir)
