from src.embeddings.embedding_generator import EmbeddingGenerator
from src.ingestion.document_loader import load_pdf
from src.preprocessing.cleaner import clean_documents
from src.preprocessing.chunker import chunk_documents
from src.retrieval.context_builder import build_context
from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


PDF_PATH = "data/raw/sbi/sbi_home_loan_mitc.pdf"


documents = load_pdf(PDF_PATH)
cleaned_documents = clean_documents(documents)
chunks = chunk_documents(cleaned_documents)

embedding_generator = EmbeddingGenerator()

chunk_embeddings = embedding_generator.embed_documents(
    [chunk.page_content for chunk in chunks]
)

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
    top_k=3,
)

context = build_context(results)

print("\nQUERY:")
print(query)

print("\nRETRIEVED CONTEXT:")
print(context)