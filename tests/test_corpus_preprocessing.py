from src.ingestion.corpus_loader import load_corpus
from src.preprocessing.corpus_preprocessor import preprocess_corpus


def test_corpus_preprocessing_creates_chunks():
    documents = load_corpus()

    chunks = preprocess_corpus(documents)

    assert len(chunks) > len(documents)


def test_preprocessing_preserves_metadata():
    documents = load_corpus()

    chunks = preprocess_corpus(documents)

    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert "institution" in chunk.metadata
        assert "loan_type" in chunk.metadata


def test_chunks_have_content():
    documents = load_corpus()

    chunks = preprocess_corpus(documents)

    for chunk in chunks:
        assert chunk.page_content.strip()