from langchain_core.documents import Document

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


def test_faiss_retrieval():
    documents = [
        Document(
            page_content=(
                "A maximum permissible LTV ratio of 75% "
                "is applicable on a loan amount above Rs.75 Lacs."
            ),
            metadata={
                "source": "sbi_home_loan_mitc.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content=(
                "The loan is to be repaid in Equated Monthly "
                "Installments over the tenure of the loan."
            ),
            metadata={
                "source": "sbi_home_loan_mitc.pdf",
                "page": 2,
            },
        ),
        Document(
            page_content=(
                "The premium for the optional Home Loan Life "
                "Insurance cover will be added to the loan amount."
            ),
            metadata={
                "source": "sbi_home_loan_mitc.pdf",
                "page": 1,
            },
        ),
    ]

    embedding_generator = EmbeddingGenerator()

    embeddings = embedding_generator.embed_documents(
        [document.page_content for document in documents]
    )

    vector_store = FAISSVectorStore()
    vector_store.add_documents(documents, embeddings)

    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
    )

    results = retriever.retrieve(
        "What is the maximum LTV for a loan above 75 lakh?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0].metadata["page"] == 1
    assert "LTV" in results[0][0].page_content
    assert results[0][1] > results[1][1]