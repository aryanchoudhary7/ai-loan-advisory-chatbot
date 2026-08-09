from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingGenerator:
    """
    Generate embeddings for documents and queries.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a user query.
        """
        return self.embeddings.embed_query(query)
    