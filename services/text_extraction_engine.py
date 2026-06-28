from typing import Set, Union
from pathlib import Path
import re
from docx import Document

class TextExtractionEngine:
    """
    Extracts raw text from a candidate resume / job description.

    Accepts:
      - a raw string (already-loaded text)
      - a path to a plain-text (.txt) file        -> read with stdlib only
      - a path to a PDF file (.pdf)                -> read with pdfplumber,
                                                        falling back to pypdf

    PDF parsing is the one place this engine reaches outside the standard
    library, since no stdlib module can parse PDF's binary format. Both
    extractors used here are lightweight, pure-Python, and limited strictly
    to read-only text extraction (no PDF creation/editing dependencies).
    """

    @staticmethod
    def extract(source: Union[str, Path]) -> str:
        text = TextExtractionEngine._load(source)
        return TextExtractionEngine._clean(text)

    @staticmethod
    def _load(source: Union[str, Path]) -> str:
        # Decide whether `source` looks like a filesystem path or raw text.
        looks_like_path = isinstance(source, Path) or (
            isinstance(source, str)
            and "\n" not in source
            and len(source) < 260  # typical max filename/path length
        )

        if looks_like_path:
            path = Path(source)
            try:
                file_exists = path.is_file()
            except OSError:
                file_exists = False  # e.g. invalid characters for a path

            if file_exists:
                suffix = path.suffix.lower()
                if suffix == ".docx":
                    doc = Document(path)
                    return "\n".join([paragraph.text for paragraph in doc.paragraphs])
                if suffix == ".txt":
                    return path.read_text(encoding="utf-8", errors="ignore")
                if suffix == ".pdf":
                    return TextExtractionEngine._extract_pdf(path)
                raise ValueError(
                    f"Unsupported file type '{suffix}'. "
                    "Supported file types: .txt, .pdf (or pass raw text directly)."
                )
            if isinstance(source, Path):
                # An explicit Path is an unambiguous signal this should be a
                # file, not raw text - so a missing file is a real error,
                # not something to silently fall back on.
                raise FileNotFoundError(f"No such file: '{source}'")

        # Otherwise treat the input itself as raw text.
        if isinstance(source, str):
            return source

        raise TypeError("source must be a raw text string or a path to a .txt/.pdf file")

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        """
        Extracts text from a PDF resume/JD.

        Tries pdfplumber first (generally the cleanest extraction for
        text-based resumes, including multi-column layouts). Falls back to
        pypdf if pdfplumber fails to load the file or yields no text.
        """
        text = TextExtractionEngine._extract_pdf_with_pdfplumber(path)
        if not text.strip():
            text = TextExtractionEngine._extract_pdf_with_pypdf(path)

        if not text.strip():
            raise ValueError(
                f"No extractable text found in '{path.name}'. This usually means "
                "either (a) the PDF is a scanned/image-based document with no "
                "text layer, which needs OCR rather than direct text extraction "
                "(out of scope for this engine), or (b) the file is corrupted, "
                "encrypted, or not a valid PDF."
            )
        return text

    @staticmethod
    def _extract_pdf_with_pdfplumber(path: Path) -> str:
        try:
            import pdfplumber
        except ImportError:
            return ""

        try:
            page_texts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_texts.append(page.extract_text() or "")
            return "\n".join(page_texts)
        except Exception:
            return ""

    @staticmethod
    def _extract_pdf_with_pypdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "PDF extraction requires pdfplumber and/or pypdf. "
                "Install with: pip install pdfplumber pypdf"
            )

        try:
            reader = PdfReader(str(path))
            page_texts = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(page_texts)
        except Exception:
            # Corrupted/invalid PDF, encrypted file, etc. - let the caller
            # decide what to do with "no text extracted" rather than
            # propagating a confusing low-level parser exception.
            return ""

    @staticmethod
    def _clean(text: str) -> str:
        """Light structural cleanup - collapse whitespace/newlines, strip control chars."""
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

