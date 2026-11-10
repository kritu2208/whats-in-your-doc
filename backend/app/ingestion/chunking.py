from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from app.ingestion.parsers import ParsedPage

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    page_number: int | None


def _split_sentences(text: str) -> list[str]:
    # Keep separators in the following segment so text is not dropped.
    segments = re.split(r"(?<=[.!?])(?=\s)|(?<=\n{2})", text)
    return [segment for segment in segments if segment]


def _split_oversized_segment(segment: str, max_tokens: int) -> list[str]:
    if len(_TOKENIZER.encode(segment)) <= max_tokens:
        return [segment]

    words = re.findall(r"\S+\s*", segment)
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = current + word
        if current and len(_TOKENIZER.encode(candidate)) > max_tokens:
            pieces.append(current)
            current = word
        else:
            current = candidate

        if len(_TOKENIZER.encode(current)) > max_tokens:
            start = 0
            while start < len(current):
                low, high = start + 1, len(current)
                end = low
                while low <= high:
                    middle = (low + high) // 2
                    if len(_TOKENIZER.encode(current[start:middle])) <= max_tokens:
                        end = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                pieces.append(current[start:end])
                start = end
            current = ""

    if current:
        pieces.append(current)
    return pieces


def _page_chunks(
    text: str,
    page_number: int | None,
    *,
    chunk_size: int,
    overlap: int,
    index_start: int,
) -> list[TextChunk]:
    if not text.strip():
        return []

    max_segment_tokens = chunk_size - overlap
    segments = [
        piece
        for segment in _split_sentences(text)
        for piece in _split_oversized_segment(segment, max_segment_tokens)
    ]
    chunk_token_ids: list[int] = []
    chunks: list[TextChunk] = []

    def decode_valid_text(token_ids: list[int]) -> str:
        while token_ids:
            try:
                return _TOKENIZER.decode_bytes(token_ids).decode("utf-8")
            except UnicodeDecodeError:
                token_ids = token_ids[1:]
        return ""

    def emit(token_ids: list[int]) -> None:
        content = decode_valid_text(token_ids).strip()
        if content:
            chunks.append(
                TextChunk(
                    chunk_index=index_start + len(chunks),
                    content=content,
                    page_number=page_number,
                )
            )

    for segment in segments:
        segment_tokens = _TOKENIZER.encode(segment)
        if not segment_tokens:
            continue
        if chunk_token_ids and len(chunk_token_ids) + len(segment_tokens) > chunk_size:
            emit(chunk_token_ids)
            overlap_size = min(overlap, chunk_size - len(segment_tokens))
            overlap_tokens = chunk_token_ids[-overlap_size:] if overlap_size else []
            while overlap_tokens:
                try:
                    _TOKENIZER.decode_bytes(overlap_tokens).decode("utf-8")
                    break
                except UnicodeDecodeError:
                    overlap_tokens = overlap_tokens[1:]
            chunk_token_ids = overlap_tokens
        if len(segment_tokens) > chunk_size:
            # Oversized segments were divided before token packing.
            raise ValueError("A text segment exceeded the configured chunk size.")
        chunk_token_ids.extend(segment_tokens)

    if chunk_token_ids:
        emit(chunk_token_ids)
    return chunks


def chunk_pages(
    pages: list[ParsedPage],
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[TextChunk] = []
    for page in pages:
        page_chunks = _page_chunks(
            page.text,
            page.page_number,
            chunk_size=chunk_size,
            overlap=overlap,
            index_start=len(chunks),
        )
        chunks.extend(page_chunks)
    return chunks
