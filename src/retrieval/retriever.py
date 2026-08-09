from langchain_core.documents import Document

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectorstore.faiss_store import FAISSVectorStore


class DocumentRetriever:
    """
    Retrieve relevant documents using semantic vector search.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_generator: EmbeddingGenerator,
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve the most relevant documents for a query.
        """

        query_embedding = self.embedding_generator.embed_query(query)

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )