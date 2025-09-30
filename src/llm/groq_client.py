# src/llm/groq_client.py
import os
import json
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com")

def call_groq(messages, json_mode=True, timeout=60, temperature=0.2, max_tokens=None):
    """
    Call Groq's OpenAI-compatible Chat Completions API using POST.
    Raises with clear error if anything goes wrong.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment")

    url = f"{GROQ_BASE_URL}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_mode:
        # Many Groq models support this OpenAI-compatible field
        payload["response_format"] = {"type": "json_object"}

    # POST (bukan GET!)
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

    # Debug ringan
    method_used = getattr(resp.request, "method", "UNKNOWN")
    if method_used != "POST":
        raise RuntimeError(f"Expected POST, got {method_used}")

    # Raise jika non-2xx dan tampilkan body dari Groq
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail = resp.text
        raise RuntimeError(f"Groq error {resp.status_code}: {detail}") from e

    data = resp.json()
    return data["choices"][0]["message"]["content"]
