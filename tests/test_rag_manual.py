from src.embeddings.embedding_generator import EmbeddingGenerator
from src.ingestion.document_loader import load_pdf
from src.llm.gemini_client import GeminiClient
from src.pipeline.rag_pipeline import RAGPipeline

from src.preprocessing.cleaner import clean_documents
from src.preprocessing.chunker import chunk_documents

from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


PDF_PATH = "data/raw/sbi/sbi_home_loan_mitc.pdf"


print("Loading document...")

documents = load_pdf(PDF_PATH)

print(f"Pages loaded: {len(documents)}")

documents = clean_documents(documents)

chunks = chunk_documents(documents)

print(f"Chunks created: {len(chunks)}")


embedding_generator = EmbeddingGenerator()

embeddings = embedding_generator.embed_documents(
    [document.page_content for document in chunks]
)

vector_store = FAISSVectorStore()

vector_store.add_documents(
    chunks,
    embeddings,
)

retriever = DocumentRetriever(
    vector_store=vector_store,
    embedding_generator=embedding_generator,
)

gemini_client = GeminiClient()

pipeline = RAGPipeline(
    retriever=retriever,
    gemini_client=gemini_client,
)


question = "What is the maximum LTV for a loan above 75 lakh?"

response = pipeline.ask(
    question,
    top_k=3,
)


print("\n" + "=" * 80)
print("QUESTION")
print("=" * 80)
print(question)

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)
print(response.answer)

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for index, source in enumerate(response.sources, start=1):
    print(f"\nSource {index}")
    print(f"File: {source.source}")
    print(f"Page: {source.page}")
    print(f"Score: {source.score:.4f}")