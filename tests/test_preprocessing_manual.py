from src.ingestion.document_loader import load_pdf
from src.preprocessing.cleaner import clean_documents
from src.preprocessing.chunker import chunk_documents


PDF_PATH = "data/raw/sbi/sbi_home_loan_mitc.pdf"

documents = load_pdf(PDF_PATH)

cleaned_documents = clean_documents(documents)

chunks = chunk_documents(cleaned_documents)

print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:3], start=1):
    print("\n" + "=" * 80)
    print(f"CHUNK {i}")
    print("Metadata:", chunk.metadata)
    print("Length:", len(chunk.page_content))
    print(chunk.page_content)