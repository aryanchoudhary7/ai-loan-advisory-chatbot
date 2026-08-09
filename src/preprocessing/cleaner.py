import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text without changing its meaning.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove standalone page numbers at the beginning
    text = re.sub(r"^\s*\d+\s*\n", "", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Clean the text content of each document while preserving metadata.
    """

    cleaned_documents = []

    for document in documents:
        cleaned_text = clean_text(document.page_content)

        if cleaned_text:
            cleaned_documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata=document.metadata.copy(),
                )
            )

    return cleaned_documents