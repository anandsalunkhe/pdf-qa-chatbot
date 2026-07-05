import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 1.0


def init_embedding_client() -> OpenAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to .env or export it.")

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    print(f"[Info] Embedding model: {EMBEDDING_MODEL}")
    return client


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_chunks(client: OpenAI, chunks: list[str]) -> np.ndarray:
    all_embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"[Info] Embedding chunks {i + 1}–{i + len(batch)} / {len(chunks)}")
        all_embeddings.extend(_embed_batch(client, batch))

        if i + BATCH_SIZE < len(chunks):
            time.sleep(RATE_LIMIT_DELAY)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return np.array(response.data[0].embedding, dtype=np.float32)
