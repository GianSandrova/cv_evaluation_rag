import re
from typing import List, Tuple

def chunk_by_words(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_words - overlap_words)
    while i < len(words):
        piece = " ".join(words[i:i+chunk_words])
        chunks.append(piece)
        i += step
    return chunks

_HEADING_PATTERNS = [
    r"about the job",
    r"about you",
    r"benefits(?:\s*&\s*perks)?",
    r"responsibilities",
    r"here are some real examples.*",
]

def split_by_headings(text: str) -> List[Tuple[str, str]]:
    """
    Balikkan list (section_key, section_text).
    Jika tak ada heading, kembalikan [('overview', full_text)].
    """
    lines = text.splitlines()
    hits = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        norm = re.sub(r"[:\-–\s]+$", "", line.lower())
        for pat in _HEADING_PATTERNS:
            if re.fullmatch(pat, norm):
                hits.append((i, line))
                break

    if not hits:
        return [("overview", text.strip())]

    sections = []
    # preface sebelum heading pertama = overview
    first_i = hits[0][0]
    pre = "\n".join(lines[:first_i]).strip()
    if pre:
        sections.append(("overview", pre))

    for idx, (start_i, title) in enumerate(hits):
        end_i = hits[idx + 1][0] if idx + 1 < len(hits) else len(lines)
        sec_text = "\n".join(lines[start_i + 1 : end_i]).strip()
        if not sec_text:
            continue
        key = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sections.append((key, sec_text))

    return sections or [("overview", text.strip())]
