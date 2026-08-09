from langchain_core.documents import Document

from src.preprocessing.cleaner import clean_text, clean_documents
from src.preprocessing.chunker import chunk_documents



def test_clean_text():
    text = "1\n\nHello    world\n\n\nThis is a test."

    cleaned = clean_text(text)

    assert cleaned == "Hello world\n\nThis is a test."


def test_clean_documents_preserves_metadata():
    documents = [
        Document(
            page_content="  Test   document  ",
            metadata={
                "source": "test.pdf",
                "page": 1,
            },
        )
    ]

    cleaned = clean_documents(documents)

    assert cleaned[0].page_content == "Test document"
    assert cleaned[0].metadata["source"] == "test.pdf"
    assert cleaned[0].metadata["page"] == 1


def test_chunk_documents():
    documents = [
        Document(
            page_content="This is a test document. " * 100,
            metadata={
                "source": "test.pdf",
                "page": 1,
            },
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "test.pdf"
    assert chunks[0].metadata["page"] == 1