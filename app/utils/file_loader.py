import io
import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from pypdf import PdfReader
from docx import Document

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _detect_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def _read_text_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    ext = _detect_extension(filename)

    if ext == ".txt":
        return _read_text_bytes(data)

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)

    if ext == ".docx":
        doc = Document(io.BytesIO(data))
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported file type: {ext}")


def extract_text_from_path(path: Path) -> str:
    data = path.read_bytes()
    return extract_text_from_bytes(path.name, data)


def get_upload_dir() -> Path:
    root = Path(os.getenv("UPLOAD_DIR", "uploads"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_storage_backend() -> str:
    return os.getenv("STORAGE_BACKEND", "local").strip().lower()


def _generate_filename(original_name: str) -> str:
    ext = _detect_extension(original_name)
    return f"{uuid4().hex}{ext}"


def _validate_extension(filename: str) -> None:
    ext = _detect_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")


async def save_upload_to_disk(upload_file: UploadFile) -> Path:
    if not upload_file.filename:
        raise ValueError("File name is missing")

    _validate_extension(upload_file.filename)

    upload_dir = get_upload_dir()
    filename = _generate_filename(upload_file.filename)
    destination = upload_dir / filename

    with destination.open("wb") as handle:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    await upload_file.close()
    return destination


async def save_upload_to_s3(upload_file: UploadFile) -> Path:
    raise ValueError("S3 storage is not configured. Use STORAGE_BACKEND=local for now.")


async def save_upload(upload_file: UploadFile) -> Path:
    backend = get_storage_backend()

    if backend == "s3":
        return await save_upload_to_s3(upload_file)

    if backend != "local":
        raise ValueError(f"Unsupported storage backend: {backend}")

    return await save_upload_to_disk(upload_file)
