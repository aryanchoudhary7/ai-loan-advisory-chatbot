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

    vector_store.add_documents(
        chunks,
        embeddings,
    )

    return DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
    )


def test_sbi_home_loan_eligibility_retrieval():
    retriever = build_retriever()

    results = retriever.retrieve(
        "What are the eligibility requirements for an SBI home loan?",
        top_k=5,
    )

    assert len(results) == 5

    top_document, top_score = results[0]

    assert top_score > 0
    assert top_document.metadata["institution"] == "SBI"
    assert top_document.metadata["loan_type"] == "home_loan"


def test_sbi_education_loan_eligibility_retrieval():
    retriever = build_retriever()

    results = retriever.retrieve(
        "Who is eligible for an SBI education loan?",
        top_k=5,
    )

    for rank, (document, score) in enumerate(results, start=1):
        print(
            f"\nRank {rank}"
            f"\nScore: {score:.4f}"
            f"\nInstitution: {document.metadata.get('institution')}"
            f"\nLoan type: {document.metadata.get('loan_type')}"
            f"\nSource: {document.metadata.get('source')}"
            f"\nPage: {document.metadata.get('page')}"
            f"\nText: {document.page_content[:500]}"
        )

    assert len(results) == 5


def test_personal_loan_eligibility_retrieval():
    retriever = build_retriever()

    results = retriever.retrieve(
        "What are the eligibility requirements for a personal loan?",
        top_k=5,
    )

    assert len(results) == 5

    top_document, top_score = results[0]

    assert top_score > 0
    assert top_document.metadata["loan_type"] == "personal_loan"