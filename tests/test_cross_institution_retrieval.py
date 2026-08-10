from src.embeddings.embedding_generator import EmbeddingGenerator
from src.ingestion.corpus_loader import load_corpus
from src.preprocessing.corpus_preprocessor import preprocess_corpus
from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


def build_retriever():
    documents = load_corpus()
    chunks = preprocess_corpus(documents)

    embedding_generator = EmbeddingGenerator()

    embeddings = embedding_generator.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    vector_store = FAISSVectorStore()
    vector_store.add_documents(chunks, embeddings)

    return DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
    )


def test_sbi_home_loan_retrieval():
    retriever = build_retriever()

    results = retriever.retrieve(
        "What is the maximum LTV for a loan above 75 lakh?",
        top_k=1,
    )

    document, score = results[0]

    assert document.metadata["institution"] == "SBI"
    assert document.metadata["loan_type"] == "home_loan"
    assert "LTV" in document.page_content


def test_rbi_penal_charges_retrieval():
    retriever = build_retriever()

    results = retriever.retrieve(
        "What are the RBI rules regarding penal charges on loans?",
        top_k=1,
    )

    document, score = results[0]

    assert document.metadata["institution"] == "RBI"
    assert "penal" in document.page_content.lower()