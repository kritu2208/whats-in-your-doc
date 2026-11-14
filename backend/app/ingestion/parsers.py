from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from importlib import import_module
from math import hypot
from pathlib import PurePosixPath
from typing import Any

from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_PDF_IMAGES = 100
MAX_PDF_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_IMAGES_BYTES = 20 * 1024 * 1024
IMAGE_MEDIA_TYPES = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed."""


@dataclass(frozen=True)
class ParsedPage:
    text: str
    page_number: int | None


@dataclass(frozen=True)
class ParsedImage:
    page_number: int
    image_index: int
    filename: str
    media_type: str
    data: bytes
    associated_text: str = ""


@dataclass(frozen=True)
class ParsedDocument:
    pages: list[ParsedPage]
    media_type: str
    metadata: dict[str, Any]
    images: list[ParsedImage] = field(default_factory=list)


SUPPORTED_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


def get_extension(filename: str) -> str:
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower()


def _normalize_text(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_pdf(data: bytes) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        pages = [
            ParsedPage(text=_normalize_text(page.extract_text() or ""), page_number=index)
            for index, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:
        raise DocumentParseError("The uploaded file is not a readable PDF.") from exc

    images, skipped_image_count = _extract_pdf_images(data)
    return ParsedDocument(
        pages=pages,
        media_type="application/pdf",
        metadata={
            "page_count": len(pages),
            "source_format": "pdf",
            "image_count": len(images),
            "image_skipped_count": skipped_image_count,
        },
        images=images,
    )


def _extract_pdf_images(data: bytes) -> tuple[list[ParsedImage], int]:
    try:
        fitz = import_module("fitz")
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentParseError("Could not inspect PDF image layout.") from exc

    images: list[ParsedImage] = []
    skipped_count = 0
    total_bytes = 0

    with pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_image_index = 0
            text_lines = _pdf_text_lines(page)
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                try:
                    extracted = pdf.extract_image(xref)
                    extension = PurePosixPath(
                        f"image.{extracted['ext']}"
                    ).suffix.lower()
                    if extension == ".jpeg":
                        extension = ".jpg"
                    media_type = IMAGE_MEDIA_TYPES.get(extension)
                    image_data = extracted["image"]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        skipped_count += 1
                        continue

                    for rect in rects:
                        image_index = page_image_index
                        page_image_index += 1
                        if (
                            len(images) >= MAX_PDF_IMAGES
                            or media_type is None
                            or not image_data
                            or len(image_data) > MAX_PDF_IMAGE_BYTES
                            or total_bytes + len(image_data) > MAX_PDF_IMAGES_BYTES
                        ):
                            skipped_count += 1
                            continue

                        filename = (
                            f"page-{page_number}-image-{image_index + 1}{extension}"
                        )
                        images.append(
                            ParsedImage(
                                page_number=page_number,
                                image_index=image_index,
                                filename=filename,
                                media_type=media_type,
                                data=image_data,
                                associated_text=_nearest_image_text(
                                    rect,
                                    text_lines,
                                    page.rect.width,
                                    page.rect.height,
                                ),
                            )
                        )
                        total_bytes += len(image_data)
                except Exception:
                    skipped_count += 1
                    logger.warning(
                        "Could not extract PDF image on page %s",
                        page_number,
                        exc_info=True,
                    )

    return images, skipped_count


def _pdf_text_lines(page: Any) -> list[tuple[tuple[float, float, float, float], str]]:
    lines: dict[
        tuple[int, int],
        tuple[float, float, float, float, list[str]],
    ] = {}
    for word in page.get_text("words"):
        x0, y0, x1, y1, text, block_number, line_number, _ = word
        line_key = (block_number, line_number)
        line = lines.get(line_key)
        if line is None:
            lines[line_key] = (x0, y0, x1, y1, [text])
        else:
            lines[line_key] = (
                min(line[0], x0),
                min(line[1], y0),
                max(line[2], x1),
                max(line[3], y1),
                [*line[4], text],
            )
    return [
        ((x0, y0, x1, y1), " ".join(words))
        for x0, y0, x1, y1, words in lines.values()
    ]


def _nearest_image_text(
    image_rect: Any,
    text_lines: list[tuple[tuple[float, float, float, float], str]],
    page_width: float,
    page_height: float,
) -> str:
    if not text_lines or page_width <= 0 or page_height <= 0:
        return ""

    image_box = (image_rect.x0, image_rect.y0, image_rect.x1, image_rect.y1)
    ranked_lines: list[tuple[float, str]] = []
    for text_box, text in text_lines:
        horizontal_gap = max(
            0.0,
            image_box[0] - text_box[2],
            text_box[0] - image_box[2],
        ) / page_width
        vertical_gap = max(
            0.0,
            image_box[1] - text_box[3],
            text_box[1] - image_box[3],
        ) / page_height
        distance = hypot(horizontal_gap, vertical_gap)
        if distance <= 0.15:
            ranked_lines.append((distance, text))

    ranked_lines.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(text for _, text in ranked_lines[:4])


def _parse_text(data: bytes, *, markdown: bool) -> ParsedDocument:
    try:
        text = _normalize_text(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise DocumentParseError("Text and Markdown files must be UTF-8 encoded.") from exc

    metadata: dict[str, Any] = {
        "source_format": "markdown" if markdown else "text",
        "encoding": "utf-8",
    }
    if markdown:
        title_match = re.search(r"(?m)^\s*#\s+(.+?)\s*#*\s*$", text)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        text = _readable_markdown(text)

    return ParsedDocument(
        pages=[ParsedPage(text=text, page_number=None)],
        media_type="text/markdown" if markdown else "text/plain",
        metadata=metadata,
    )


def _readable_markdown(text: str) -> str:
    text = re.sub(r"(?m)^\s*(```|~~~).*?$", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-+*]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)",
        lambda match: match.group(1) or match.group(2),
        text,
    )
    text = re.sub(r"<[^>]+>", "", text)
    return _normalize_text(text)


def parse_document(data: bytes, filename: str) -> ParsedDocument:
    extension = get_extension(filename)
    if extension not in SUPPORTED_MEDIA_TYPES:
        raise DocumentParseError("Supported file types are PDF, Markdown, and TXT.")
    if not data:
        raise DocumentParseError("The uploaded document is empty.")

    if extension == ".pdf":
        return _parse_pdf(data)
    return _parse_text(data, markdown=extension == ".md")
