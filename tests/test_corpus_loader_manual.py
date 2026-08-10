from collections import Counter

from src.ingestion.corpus_loader import load_corpus


documents = load_corpus()

print(f"Pages loaded: {len(documents)}")

institutions = Counter(
    document.metadata["institution"]
    for document in documents
)

print("\nPages by institution:")

for institution, count in sorted(institutions.items()):
    print(f"{institution}: {count}")

print("\nFirst document:")
print("Metadata:", documents[0].metadata)
print("Text preview:")
print(documents[0].page_content[:500])