import httpx
from core.config import EMBEDDING_API_URL, EMBEDDING_API_KEY


# =========================================================
# Embedding API
# =========================================================

def get_embedding(text: str) -> list:
    headers = {
        "api-key": EMBEDDING_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    response = httpx.post(
        EMBEDDING_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    print("Embedding API response:", data)
    return data["embedding"]
