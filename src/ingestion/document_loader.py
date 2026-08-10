from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def load_pdf(
    file_path: str,
    metadata: dict | None = None,
) -> list[Document]:
    """
    Load a PDF and convert each non-empty page into a LangChain Document.

    Args:
        file_path: Path to the PDF file.
        metadata: Optional metadata to attach to every page.

    Returns:
        A list of Document objects, one per non-empty page.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    documents = []

    base_metadata = {
        "source": path.name,
    }

    if metadata:
        base_metadata.update(metadata)

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            if text.strip():
                page_metadata = {
                    **base_metadata,
                    "page": page_number,
                }

                documents.append(
                    Document(
                        page_content=text,
                        metadata=page_metadata,
                    )
                )

    return documents