#!/usr/bin/env python3
import argparse
from src.config import JOB_ID_DEFAULT, COLL_JOBS_CORPUS
from src.pipeline.ingest import ingest_batch
from src.storage.qdrant_store import close_client

def main():
    parser = argparse.ArgumentParser(
        description="Ingest JD / Rubric into vector DB (Qdrant local mode, Qwen embeddings)."
    )
    parser.add_argument(
        "--job-id", default=JOB_ID_DEFAULT,
        help="Slug job id, e.g., peb-2025"
    )
    parser.add_argument(
        "--source-type", required=True, choices=["jd", "rubric"],
        help="Type of source"
    )
    parser.add_argument(
        "--section", default=None,
        help="Section tag (e.g., overview, about_the_job, rubric_cv, rubric_project). Ignored if --auto-section is used."
    )
    parser.add_argument(
        "--paths", nargs="+", required=True,
        help="File paths (pdf/docx/txt)"
    )
    parser.add_argument(
        "--no-pii-mask", action="store_true",
        help="Disable PII masking"
    )
    parser.add_argument(
        "--auto-section", action="store_true",
        help="Auto-split JD by headings into sections (ignored for source-type=rubric)"
    )

    args = parser.parse_args()

    try:
        # JD dengan auto-section: pecah per heading, set metadata section otomatis
        if args.source_type == "jd" and args.auto_section:
            from src.pipeline.ingest import ingest_jd_auto_sections  # diimport saat diperlukan
            total_chunks, all_ids = 0, []
            for p in args.paths:
                n, ids = ingest_jd_auto_sections(
                    file_path=p,
                    job_id=args.job_id,
                    mask_pii=not args.no_pii_mask,
                )
                total_chunks += n
                all_ids.extend(ids)
            print("Ingest result:", {
                "chunks": total_chunks,
                "ids": all_ids,
                "collection": COLL_JOBS_CORPUS
            })
            return

        # Jalur umum (JD satu section manual / Rubric naratif)
        ok = ingest_batch(
            paths=args.paths,
            job_id=args.job_id,
            source_type=args.source_type,
            section=args.section,
            mask_pii=not args.no_pii_mask,
        )
        print("Ingest result:", ok)

    finally:
        close_client()

if __name__ == "__main__":
    main()
