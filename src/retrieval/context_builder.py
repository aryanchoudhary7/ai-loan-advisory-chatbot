from langchain_core.documents import Document


def build_context(
    results: list[tuple[Document, float]],
) -> str:
    """
    Build a formatted context string from retrieved documents.
    """

    if not results:
        return ""

    context_parts = []

    for rank, (document, score) in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"""[Source {rank}]
Source: {source}
Page: {page}
Relevance Score: {score:.4f}

{document.page_content}
"""
        )

    return "\n\n".join(context_parts)