from __future__ import annotations

import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Iterator

import pytest
import tiktoken
from fastapi.testclient import TestClient

from app import main
from app.core.config import get_settings
from app.db.session import get_db
from app.ingestion.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingService,
    get_embedding_service,
)
from app.ingestion import parsers
from app.ingestion.chunking import chunk_pages
from app.ingestion.parsers import (
    DocumentParseError,
    ParsedDocument,
    ParsedImage,
    ParsedPage,
    parse_document,
)
from app.ingestion.service import save_document
from app.models import Document, DocumentImage


class FakePdfPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakePdfReader:
    def __init__(self, stream: BytesIO, *, strict: bool) -> None:
        assert strict
        assert stream.read() == b"%PDF-mock"
        self.pages = [FakePdfPage("Page one text"), FakePdfPage(""), FakePdfPage("Page three")]


def make_two_page_pdf() -> bytes:
    stream = b"BT /F1 12 Tf 72 720 Td (Extracted PDF text) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, content in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + content + b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def make_pdf_with_jpeg() -> bytes:
    from PIL import Image

    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(30, 120, 80)).save(image_buffer, format="JPEG")
    image_data = image_buffer.getvalue()
    content = b"q 2 0 0 2 0 0 cm /Im0 Do Q"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ),
        (
            b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length "
            + str(len(image_data)).encode()
            + b" >>\nstream\n"
            + image_data
            + b"\nendstream"
        ),
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, content_object in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + content_object + b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def test_pdf_parser_preserves_page_numbers_and_empty_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parsers, "PdfReader", FakePdfReader)
    monkeypatch.setattr(parsers, "_extract_pdf_images", lambda _data: ([], 0))

    parsed = parse_document(b"%PDF-mock", "report.PDF")

    assert parsed.media_type == "application/pdf"
    assert parsed.metadata == {
        "page_count": 3,
        "source_format": "pdf",
        "image_count": 0,
        "image_skipped_count": 0,
    }
    assert [(page.text, page.page_number) for page in parsed.pages] == [
        ("Page one text", 1),
        ("", 2),
        ("Page three", 3),
    ]


def test_pdf_parser_extracts_text_from_a_real_pdf() -> None:
    parsed = parse_document(make_two_page_pdf(), "real.pdf")

    assert parsed.metadata["page_count"] == 2
    assert parsed.pages[0] == ParsedPage("Extracted PDF text", 1)
    assert parsed.pages[1] == ParsedPage("", 2)


def test_pdf_parser_extracts_images_and_preserves_page_numbers() -> None:
    parsed = parse_document(make_pdf_with_jpeg(), "figures.pdf")

    assert parsed.metadata["page_count"] == 1
    assert parsed.metadata["image_count"] == 1
    assert parsed.metadata["image_skipped_count"] == 0
    assert len(parsed.images) == 1
    image = parsed.images[0]
    assert image.page_number == 1
    assert image.image_index == 0
    assert image.filename == "page-1-image-1.jpg"
    assert image.media_type == "image/jpeg"
    assert image.data.startswith(b"\xff\xd8")


def test_pdf_parser_associates_nearby_text_with_each_image() -> None:
    import fitz
    from PIL import Image

    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), color=(30, 120, 80)).save(image_buffer, format="PNG")

    pdf = fitz.open()
    page = pdf.new_page(width=400, height=400)
    page.insert_image(fitz.Rect(50, 50, 150, 150), stream=image_buffer.getvalue())
    page.insert_text((50, 170), "Ocicat - spotted coat", fontsize=12)
    page.insert_text((300, 350), "Unrelated page text", fontsize=12)
    data = pdf.tobytes()
    pdf.close()

    parsed = parse_document(data, "cat-chart.pdf")

    assert len(parsed.images) == 1
    assert "Ocicat" in parsed.images[0].associated_text
    assert "spotted coat" in parsed.images[0].associated_text
    assert "Unrelated page text" not in parsed.images[0].associated_text


def test_invalid_pdf_is_reported_as_parse_error() -> None:
    with pytest.raises(DocumentParseError, match="readable PDF"):
        parse_document(b"not a pdf", "invalid.pdf")


def test_empty_pdf_file_is_reported_as_parse_error() -> None:
    with pytest.raises(DocumentParseError, match="empty"):
        parse_document(b"", "empty.pdf")


def test_markdown_parser_extracts_readable_text_and_title() -> None:
    parsed = parse_document(
        b"# Project Notes\n\nRead **this guide** and [the appendix](https://example.test).",
        "notes.md",
    )

    assert parsed.media_type == "text/markdown"
    assert parsed.metadata == {
        "source_format": "markdown",
        "encoding": "utf-8",
        "title": "Project Notes",
    }
    assert parsed.pages[0].page_number is None
    assert parsed.pages[0].text == "Project Notes\n\nRead this guide and the appendix."


def test_text_parser_uses_utf8_and_preserves_plain_content() -> None:
    parsed = parse_document(b"\xef\xbb\xbfFirst line\r\nSecond line.", "notes.txt")

    assert parsed.media_type == "text/plain"
    assert parsed.metadata == {"source_format": "text", "encoding": "utf-8"}
    assert parsed.pages[0].text == "First line\nSecond line."
    assert parsed.pages[0].page_number is None


def test_invalid_text_encoding_is_rejected() -> None:
    with pytest.raises(DocumentParseError, match="UTF-8"):
        parse_document(b"\xff\xfe", "bad.txt")


def test_unsupported_file_extension_is_rejected() -> None:
    with pytest.raises(DocumentParseError, match="Supported file types"):
        parse_document(b"data", "notes.docx")


def test_chunking_respects_size_overlap_and_stable_indexes() -> None:
    text = " ".join(
        f"Sentence number {index} contains a few useful words."
        for index in range(80)
    )
    chunks = chunk_pages([ParsedPage(text, 4)], chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    assert all(len(tokenizer.encode(chunk.content)) <= 100 for chunk in chunks)
    for previous, following in zip(chunks, chunks[1:]):
        previous_last_sentence = previous.content.rsplit(". ", 1)[-1]
        assert previous_last_sentence in following.content[: len(previous_last_sentence) + 1]


def test_chunking_preserves_page_metadata_and_skips_empty_pages() -> None:
    chunks = chunk_pages(
        [
            ParsedPage("Text from page one.", 1),
            ParsedPage("   ", 2),
            ParsedPage("Text from page three.", 3),
        ],
        chunk_size=50,
        overlap=5,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 3]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_chunking_splits_long_unbroken_text_without_exceeding_limit() -> None:
    chunks = chunk_pages([ParsedPage("a" * 2500, 7)], chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk.content) > 0 for chunk in chunks)
    assert all(
        len(tiktoken.get_encoding("cl100k_base").encode(chunk.content)) <= 100
        for chunk in chunks
    )
    assert all(chunk.page_number == 7 for chunk in chunks)


def test_chunking_keeps_unicode_valid_at_overlap_boundaries() -> None:
    text = "🙂 " * 1200
    chunks = chunk_pages([ParsedPage(text, 2)], chunk_size=80, overlap=10)

    assert len(chunks) > 1
    assert all("\ufffd" not in chunk.content for chunk in chunks)
    assert all(chunk.page_number == 2 for chunk in chunks)


def test_chunking_does_not_drop_text_when_overlap_is_disabled() -> None:
    source = "First sentence stays intact. A second sentence follows.\n\nFinal paragraph."
    chunks = chunk_pages([ParsedPage(source, None)], chunk_size=8, overlap=0)

    normalize = lambda value: " ".join(value.split())
    assert normalize(" ".join(chunk.content for chunk in chunks)) == normalize(source)


class FakeEncoded:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeEncoder:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.calls = 0

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> FakeEncoded:
        assert batch_size == 16
        assert convert_to_numpy
        assert normalize_embeddings
        assert not show_progress_bar
        self.calls += 1
        return FakeEncoded([[0.0] * self.dimension for _ in sentences])


def test_embeddings_are_lazy_cached_and_384_dimensional() -> None:
    created: list[FakeEncoder] = []

    def factory(model_name: str) -> FakeEncoder:
        assert model_name == "test-model"
        model = FakeEncoder()
        created.append(model)
        return model

    service = EmbeddingService("test-model", model_factory=factory)
    assert created == []
    assert service.embed_texts(["first", "second"]) == [
        [0.0] * EMBEDDING_DIMENSION,
        [0.0] * EMBEDDING_DIMENSION,
    ]
    service.embed_texts(["third"])

    assert len(created) == 1
    assert created[0].calls == 2


def test_embedding_dimension_is_validated() -> None:
    service = EmbeddingService(
        "wrong-dimension",
        model_factory=lambda _: FakeEncoder(dimension=3),
    )

    with pytest.raises(RuntimeError, match="384-dimensional"):
        service.embed_texts(["text"])


def test_save_document_persists_extracted_page_images() -> None:
    session = MemorySession()
    parsed = ParsedDocument(
        pages=[ParsedPage("Source text.", 2)],
        media_type="application/pdf",
        metadata={"source_format": "pdf", "image_count": 1},
        images=[
            ParsedImage(
                page_number=2,
                image_index=0,
                filename="page-2-image-1.png",
                media_type="image/png",
                data=b"image-bytes",
                associated_text="A chart caption",
            )
        ],
    )

    saved = save_document(
        session,  # type: ignore[arg-type]
        filename="figures.pdf",
        parsed=parsed,
        chunks=chunk_pages(parsed.pages, chunk_size=50, overlap=5),
        embeddings=[[0.0] * EMBEDDING_DIMENSION],
        image_embeddings=[[0.0] * 512],
        image_context_embeddings=[[0.1] * EMBEDDING_DIMENSION],
    )

    assert len(saved.images) == 1
    assert isinstance(saved.images[0], DocumentImage)
    assert saved.images[0].page_number == 2
    assert saved.images[0].data == b"image-bytes"
    assert saved.images[0].context_text == "A chart caption"
    assert saved.images[0].context_embedding == [0.1] * EMBEDDING_DIMENSION


class MemorySession:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.pending: Document | None = None

    def add(self, document: Document) -> None:
        self.pending = document

    def flush(self) -> None:
        assert self.pending is not None
        self.pending.id = uuid.uuid4()

    def commit(self) -> None:
        if self.pending is not None:
            self.pending.created_at = datetime.now(timezone.utc)
            self.pending.updated_at = self.pending.created_at
            self.documents.append(self.pending)
            self.pending = None

    def rollback(self) -> None:
        self.pending = None

    def scalars(self, statement: Any) -> Any:
        del statement
        return MemoryScalarResult(self.documents)

    def execute(self, statement: Any) -> Any:
        params = statement.compile().params
        chunk_id = params.get("id_1")
        document_id = params.get("document_id_1")
        row = next(
            (
                (
                    chunk.id,
                    document.id,
                    chunk.chunk_index,
                    chunk.page_number,
                    chunk.content,
                    document.filename,
                )
                for document in self.documents
                if document.id == document_id
                for chunk in document.chunks
                if chunk.id == chunk_id
            ),
            None,
        )
        return type("MemoryResult", (), {"one_or_none": lambda self: row})()

    def get(self, model: type[Document], document_id: uuid.UUID) -> Document | None:
        assert model is Document
        return next((doc for doc in self.documents if doc.id == document_id), None)

    def delete(self, document: Document) -> None:
        self.documents.remove(document)


class MemoryScalarResult:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def all(self) -> list[Document]:
        return list(self.documents)


class StubEmbeddingService(EmbeddingService):
    def __init__(self) -> None:
        super().__init__("not-loaded")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIMENSION for _ in texts]


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, MemorySession]]:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    session = MemorySession()
    main.app.dependency_overrides[get_db] = lambda: session
    main.app.dependency_overrides[get_embedding_service] = StubEmbeddingService

    with TestClient(main.app) as client:
        yield client, session

    main.app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_upload_list_and_delete_document_api(api_client: Any) -> None:
    client, session = api_client
    uploaded = client.post(
        "/api/documents",
        files={"file": ("notes.md", b"# Notes\n\nA small document.", "text/markdown")},
    )

    assert uploaded.status_code == 201
    body = uploaded.json()
    assert body["filename"] == "notes.md"
    assert body["media_type"] == "text/markdown"
    assert body["metadata"]["title"] == "Notes"
    assert len(session.documents[0].chunks) == 1
    assert len(session.documents[0].chunks[0].embedding) == EMBEDDING_DIMENSION

    listed = client.get("/api/documents")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    chunk = session.documents[0].chunks[0]
    chunk.id = uuid.uuid4()
    preview = client.get(f"/api/documents/{body['id']}/chunks/{chunk.id}")
    assert preview.status_code == 200
    assert preview.json() == {
        "document_id": body["id"],
        "document_name": "notes.md",
        "chunk_id": str(chunk.id),
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "content": chunk.content,
    }
    missing_preview = client.get(
        f"/api/documents/{body['id']}/chunks/{uuid.uuid4()}"
    )
    assert missing_preview.status_code == 404

    deleted = client.delete(f"/api/documents/{body['id']}")
    assert deleted.status_code == 204
    assert session.documents == []


def test_document_api_rejects_bad_types_and_invalid_pdf(api_client: Any) -> None:
    client, session = api_client

    unsupported = client.post(
        "/api/documents",
        files={"file": ("notes.docx", b"no", "application/octet-stream")},
    )
    invalid_pdf = client.post(
        "/api/documents",
        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
    )
    empty_text = client.post(
        "/api/documents",
        files={"file": ("empty.txt", b" \n ", "text/plain")},
    )

    assert unsupported.status_code == 415
    assert invalid_pdf.status_code == 422
    assert empty_text.status_code == 422
    assert session.documents == []


def test_document_api_enforces_file_size_limit(
    api_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = api_client
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "4")
    get_settings.cache_clear()

    response = client.post(
        "/api/documents",
        files={"file": ("large.txt", b"12345", "text/plain")},
    )

    assert response.status_code == 413
    assert session.documents == []


def test_document_api_enforces_chunk_limit(
    api_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = api_client
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "5")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "1")
    monkeypatch.setenv("MAX_DOCUMENT_CHUNKS", "1")
    get_settings.cache_clear()

    response = client.post(
        "/api/documents",
        files={"file": ("large.txt", b"First sentence. Second sentence. Third sentence.", "text/plain")},
    )

    assert response.status_code == 413
    assert session.documents == []
