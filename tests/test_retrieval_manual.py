from src.embeddings.embedding_generator import EmbeddingGenerator
from src.ingestion.document_loader import load_pdf
from src.preprocessing.cleaner import clean_documents
from src.preprocessing.chunker import chunk_documents
from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


PDF_PATH = "data/raw/sbi/sbi_home_loan_mitc.pdf"


documents = load_pdf(PDF_PATH)

cleaned_documents = clean_documents(documents)

chunks = chunk_documents(cleaned_documents)

embedding_generator = EmbeddingGenerator()

chunk_texts = [chunk.page_content for chunk in chunks]

chunk_embeddings = embedding_generator.embed_documents(chunk_texts)

vector_store = FAISSVectorStore()

vector_store.add_documents(
    chunks,
    chunk_embeddings,
)

retriever = DocumentRetriever(
    vector_store=vector_store,
    embedding_generator=embedding_generator,
)

query = "What is the maximum LTV for a loan above 75 lakh?"

results = retriever.retrieve(
    query=query,
    top_k=5,
)

print(f"\nQuery: {query}")
print(f"Retrieved results: {len(results)}")

for rank, (document, score) in enumerate(results, start=1):
    print("\n" + "=" * 80)
    print(f"Rank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Metadata: {document.metadata}")
    print(f"Text:\n{document.page_content}")