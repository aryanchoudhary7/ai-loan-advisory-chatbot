from src.llm.gemini_client import GeminiClient
from src.llm.prompts import LOAN_ADVISORY_SYSTEM_PROMPT
from src.retrieval.context_builder import build_context
from src.retrieval.retriever import DocumentRetriever
from src.pipeline.response import RAGResponse, SourceReference
from src.validation.response_validator import validate_response


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        gemini_client: GeminiClient,
    ):
        self.retriever = retriever
        self.gemini_client = gemini_client

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """
        Retrieve relevant context and generate a grounded answer.
        """

        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        context = build_context(results)

        if not context:
            return RAGResponse(
                answer=(
                    "The available documents do not contain "
                    "enough information to answer this question."
                ),
                sources=[],
            )

        prompt = f"""
Retrieved Context:

{context}

User Question:

{question}

Answer the question using only the retrieved context.

Rules:

- Do not use information outside the retrieved context.
- If the context does not contain enough information, clearly say so.
- Do not invent or assume missing facts.
- Give a concise, direct answer.
- Do not mention source filenames, page numbers, relevance scores, or citations.
- Source information will be displayed separately by the application.
"""

        generated_answer = self.gemini_client.generate(
            prompt=prompt,
            system_instruction=LOAN_ADVISORY_SYSTEM_PROMPT,
        )

        answer = validate_response(
            answer=generated_answer,
            context=context,
        )

        sources = [
            SourceReference(
                source=document.metadata.get(
                    "source",
                    "Unknown",
                ),
                page=document.metadata.get(
                    "page",
                    "Unknown",
                ),
                score=score,
            )
            for document, score in results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
        )