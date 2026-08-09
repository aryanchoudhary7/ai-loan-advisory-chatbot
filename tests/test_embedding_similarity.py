from src.embeddings.embedding_generator import EmbeddingGenerator


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (magnitude_a * magnitude_b)


generator = EmbeddingGenerator()

query = "What is the maximum LTV for a loan above 75 lakh?"

texts = [
    "A maximum permissible LTV ratio of 75% is applicable on a loan amount above Rs.75 Lacs.",
    "The loan is to be repaid in Equated Monthly Installments over the tenure of the loan.",
    "The loan will be sanctioned for the purpose of purchase, construction, extension, repairs or renovation of a residential property.",
    "The premium for the optional Home Loan Life Insurance cover will be added to the loan amount.",
]

query_embedding = generator.embed_query(query)
text_embeddings = generator.embed_documents(texts)

for text, embedding in zip(texts, text_embeddings):
    score = cosine_similarity(query_embedding, embedding)

    print("\nText:", text)
    print("Similarity:", round(score, 4))