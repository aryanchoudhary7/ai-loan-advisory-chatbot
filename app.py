import streamlit as st

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.ingestion.corpus_loader import load_corpus
from src.llm.gemini_client import GeminiClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.preprocessing.corpus_preprocessor import preprocess_corpus
from src.retrieval.retriever import DocumentRetriever
from src.vectorstore.faiss_store import FAISSVectorStore


st.set_page_config(
    page_title="AI Loan Advisory Chatbot",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def initialize_pipeline() -> RAGPipeline:
    """
    Build and cache the complete RAG pipeline.

    The expensive embedding model and FAISS index are created
    only once during the Streamlit session.
    """

    documents = load_corpus()

    chunks = preprocess_corpus(documents)

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

    return RAGPipeline(
        retriever=retriever,
        gemini_client=gemini_client,
    )


st.title("🏦 AI Loan Advisory Chatbot")

st.markdown(
    """
Ask questions about home loans, personal loans, education loans,
and RBI loan-related guidelines using the provided documents.
"""
)

with st.sidebar:
    st.header("About")

    st.write(
        """
        This chatbot uses Retrieval-Augmented Generation (RAG)
        to answer loan-related questions from a curated document
        corpus.
        """
    )

    st.divider()

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.caption(
        "Answers are generated only from retrieved document context."
    )


try:
    pipeline = initialize_pipeline()
except Exception as exc:
    st.error("Unable to initialize the loan advisory system.")
    st.exception(exc)
    st.stop()


question = st.chat_input(
    "Ask a question about loans..."
)


if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = pipeline.ask(
                    question=question,
                    top_k=top_k,
                )

                st.write(response.answer)

                if response.sources:
                    st.divider()
                    st.subheader("Sources")

                    grouped_sources = {}

                    for source in response.sources:
                        if source.source not in grouped_sources:
                            grouped_sources[source.source] = {
                                "pages": [],
                                "scores": [],
                            }

                        grouped_sources[source.source]["pages"].append(source.page)
                        grouped_sources[source.source]["scores"].append(source.score)

                    for index, (source_name, data) in enumerate(
                        grouped_sources.items(),
                        start=1,
                    ):
                        pages = sorted(set(data["pages"]))
                        max_score = max(data["scores"])

                        with st.expander(
                            f"Source {index}: {source_name}"
                        ):
                            st.write(
                                f"**Pages:** {', '.join(map(str, pages))}"
                            )
                            st.write(
                                f"**Best relevance score:** "
                                f"{max_score:.4f}"
                            )

            except Exception as exc:
                st.error(
                    "An error occurred while generating the answer."
                )
                st.exception(exc)