"""Aegis Document Intelligence Agent - Evidence Retrieval Service.

Provides bounded document text chunking, keyword/phrase evidence search,
and source metadata preservation.
"""

import re
from typing import Any, Dict, List, Optional


def chunk_document_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Split document text into bounded overlapping chunks with start/end character offsets."""
    if not text:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0
    text_len = len(text)
    chunk_id = 1

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append({
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": end,
                "text": chunk_str,
            })
            chunk_id += 1
        start += (chunk_size - overlap)

    return chunks


def retrieve_relevant_chunks(
    query: str,
    docs: Dict[str, Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant text chunks matching a search query across loaded documents."""
    if not query or not docs:
        return []

    query_tokens = set(re.findall(r'\w+', query.lower()))
    if not query_tokens:
        return []

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "of", "with"}
    key_tokens = query_tokens - stopwords
    if not key_tokens:
        key_tokens = query_tokens

    scored_chunks: List[Dict[str, Any]] = []

    for name, doc in docs.items():
        text = str(doc.get("text") or doc.get("content") or "")
        pages = doc.get("pages") or []

        if isinstance(pages, list) and pages:
            for p in pages:
                p_num = p.get("page_number", 1)
                p_text = str(p.get("text", ""))
                p_chunks = chunk_document_text(p_text, chunk_size=400, overlap=40)
                for c in p_chunks:
                    c_tokens = set(re.findall(r'\w+', c["text"].lower()))
                    overlap_count = len(key_tokens & c_tokens)
                    score = overlap_count / len(key_tokens)
                    if score > 0.0:
                        scored_chunks.append({
                            "source_file": name,
                            "location": f"Page {p_num}",
                            "score": round(score, 3),
                            "excerpt": c["text"],
                            "start_char": c["start_char"],
                            "end_char": c["end_char"],
                        })
        else:
            chunks = chunk_document_text(text, chunk_size=500, overlap=50)
            for c in chunks:
                c_tokens = set(re.findall(r'\w+', c["text"].lower()))
                overlap_count = len(key_tokens & c_tokens)
                score = overlap_count / len(key_tokens)
                if score > 0.0:
                    scored_chunks.append({
                        "source_file": name,
                        "location": "Document Body",
                        "score": round(score, 3),
                        "excerpt": c["text"],
                        "start_char": c["start_char"],
                        "end_char": c["end_char"],
                    })

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]
