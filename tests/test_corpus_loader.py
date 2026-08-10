from src.ingestion.corpus_loader import load_corpus


def test_corpus_loads():
    documents = load_corpus()

    assert len(documents) > 0


def test_corpus_contains_expected_institutions():
    documents = load_corpus()

    institutions = {
        document.metadata["institution"]
        for document in documents
    }

    assert institutions == {"AXIS", "ICICI", "RBI", "SBI"}


def test_corpus_metadata_is_present():
    documents = load_corpus()

    required_fields = {
        "source",
        "page",
        "institution",
        "institution_type",
        "loan_type",
        "document_type",
    }

    for document in documents:
        assert required_fields.issubset(document.metadata.keys())


def test_corpus_contains_expected_loan_types():
    documents = load_corpus()

    bank_documents = [
        document
        for document in documents
        if document.metadata["institution_type"] == "bank"
    ]

    loan_types = {
        document.metadata["loan_type"]
        for document in bank_documents
    }

    assert loan_types == {
        "home_loan",
        "personal_loan",
        "education_loan",
    }


def test_page_numbers_are_valid():
    documents = load_corpus()

    for document in documents:
        assert isinstance(document.metadata["page"], int)
        assert document.metadata["page"] >= 1