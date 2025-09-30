import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(\d{1,4}\)[\s-]?)?\d[\d\s-]{7,}\d)")

def normalize_bullets(text: str) -> str:
    # ubah bullet unicode jadi dash
    text = text.replace("•", "- ").replace("◦", "- ")
    return text

def collapse_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("<email>", text)
    text = PHONE_RE.sub("<phone>", text)
    return text

def normalize_text(text: str, mask_pii_flag: bool = True) -> str:
    text = normalize_bullets(text)
    text = collapse_spaces(text)
    if mask_pii_flag:
        text = mask_pii(text)
    return text
