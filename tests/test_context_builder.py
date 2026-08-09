from langchain_core.documents import Document

from src.retrieval.context_builder import build_context


def test_build_context():
    documents = [
        (
            Document(
                page_content="Maximum LTV is 75%.",
                metadata={
                    "source": "sbi_home_loan_mitc.pdf",
                    "page": 1,
                },
            ),
            0.7212,
        )
    ]

    context = build_context(documents)

    assert "Maximum LTV is 75%." in context
    assert "sbi_home_loan_mitc.pdf" in context
    assert "Page: 1" in context
    assert "0.7212" in context


def test_empty_context():
    context = build_context([])

    assert context == ""