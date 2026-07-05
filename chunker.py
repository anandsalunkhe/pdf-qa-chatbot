def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    print(f"[Info] Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks
