from src.embeddings.embedding_generator import EmbeddingGenerator
from src.ingestion.corpus_loader import load_corpus
from src.preprocessing.corpus_preprocessor import preprocess_corpus
from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


def test_corpus_faiss_retrieval():
    documents = load_corpus()
    chunks = preprocess_corpus(documents)

    embedding_generator = EmbeddingGenerator()

    embeddings = embedding_generator.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    vector_store = FAISSVectorStore()
    vector_store.add_documents(chunks, embeddings)

    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
    )

    results = retriever.retrieve(
        "What is the maximum LTV for a loan above 75 lakh?",
        top_k=5,
    )

    assert len(results) == 5

    top_document, top_score = results[0]

    assert top_score > 0
    assert "LTV" in top_document.page_content
    assert top_document.metadata["institution"] == "SBI"
    assert top_document.metadata["loan_type"] == "home_loan"