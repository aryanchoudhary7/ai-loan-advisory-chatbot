from langchain_core.documents import Document

from src.preprocessing.cleaner import clean_documents
from src.preprocessing.chunker import chunk_documents


def preprocess_corpus(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Clean and chunk a collection of documents.

    Metadata is preserved throughout the preprocessing pipeline.

    Args:
        documents: Page-level LangChain Documents.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of cleaned and chunked Documents.
    """

    cleaned_documents = clean_documents(documents)

    chunks = chunk_documents(
        cleaned_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunks