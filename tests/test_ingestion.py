import pytest

from src.ingestion.document_loader import load_pdf


PDF_PATH = "data/raw/sbi/home_loan/sbi_home_loan_mitc.pdf"


def test_load_pdf():
    documents = load_pdf(PDF_PATH)

    assert len(documents) == 4
    assert documents[0].page_content.strip()
    assert documents[0].metadata["source"] == "sbi_home_loan_mitc.pdf"
    assert documents[0].metadata["page"] == 1


def test_pdf_not_found():
    with pytest.raises(FileNotFoundError):
        load_pdf("data/raw/sbi/nonexistent.pdf")


def test_invalid_file_type(tmp_path):
    invalid_file = tmp_path / "example.txt"
    invalid_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        load_pdf(str(invalid_file))