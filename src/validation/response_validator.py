def validate_response(
    answer: str,
    context: str,
) -> str:
    """
    Validate a generated answer against the retrieved context.

    Returns the original answer when it is non-empty and
    retrieved context is available. Otherwise, returns a
    safe fallback message.
    """

    if not context.strip():
        return (
            "The available documents do not contain enough "
            "information to answer this question."
        )

    if not answer or not answer.strip():
        return (
            "The system could not generate a reliable answer "
            "from the available documents."
        )

    return answer.strip()