import csv
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.document_loader import load_pdf


def load_corpus(
    raw_dir: str = "data/raw",
    inventory_path: str = "metadata/document_inventory.csv",
) -> list[Document]:
    """
    Load all PDF documents listed in the document inventory.

    Returns:
        A list of page-level LangChain Documents with metadata.
    """

    raw_path = Path(raw_dir)
    inventory_file = Path(inventory_path)

    if not inventory_file.exists():
        raise FileNotFoundError(
            f"Document inventory not found: {inventory_path}"
        )

    documents = []

    with inventory_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            pdf_path = raw_path / row["relative_path"]

            metadata = {
                "institution": row["institution"],
                "institution_type": row["institution_type"],
                "loan_type": row["loan_type"],
                "document_type": row["document_type"],
            }

            documents.extend(
                load_pdf(
                    str(pdf_path),
                    metadata=metadata,
                )
            )

    return documents