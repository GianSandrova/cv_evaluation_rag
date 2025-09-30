import logging
import os
import textwrap

def setup_logging(name: str = "app") -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    # force=True supaya format/level kita selalu dipakai (RQ kadang sudah pasang handler)
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    return logging.getLogger(name)

def short(text: str, width: int = 240) -> str:
    return textwrap.shorten((text or "").replace("\n", " "), width=width, placeholder="…")

def hr(title: str) -> str:
    return f"\n======== {title} ========\n"
