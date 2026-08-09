from langchain_core.documents import Document

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


class FakeGeminiClient:
    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        assert "Maximum LTV is 75%" in prompt

        return "The maximum permissible LTV is 75%."


def test_rag_pipeline():
    documents = [
        Document(
            page_content="Maximum LTV is 75%.",
            metadata={
                "source": "sbi_home_loan_mitc.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="Loan tenure can be up to 30 years.",
            metadata={
                "source": "sbi_home_loan_mitc.pdf",
                "page": 2,
            },
        ),
    ]

    embedding_generator = EmbeddingGenerator()

    embeddings = embedding_generator.embed_documents(
        [document.page_content for document in documents]
    )

    vector_store = FAISSVectorStore()

    vector_store.add_documents(
        documents,
        embeddings,
    )

    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        gemini_client=FakeGeminiClient(),
    )

    response = pipeline.ask(
        "What is the maximum LTV?",
        top_k=1,
    )

    assert response.answer == (
        "The maximum permissible LTV is 75%."
    )

    assert len(response.sources) == 1
    assert response.sources[0].source == (
        "sbi_home_loan_mitc.pdf"
    )
    assert response.sources[0].page == 1