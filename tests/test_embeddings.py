from src.embeddings.embedding_generator import EmbeddingGenerator


def test_embedding_generation():
    generator = EmbeddingGenerator()

    embeddings = generator.embed_documents(
        [
            "What is the maximum LTV for a home loan?",
            "The maximum permissible LTV above Rs.75 Lacs is 75%.",
        ]
    )

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_query_embedding():
    generator = EmbeddingGenerator()

    embedding = generator.embed_query(
        "What is the LTV for a loan above 75 lakh?"
    )

    assert len(embedding) == 384