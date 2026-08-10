from src.validation.response_validator import validate_response


def test_valid_response_is_returned():
    answer = "The maximum LTV is 75%."
    context = "The maximum LTV for the specified loan is 75%."

    result = validate_response(
        answer=answer,
        context=context,
    )

    assert result == answer


def test_empty_context_returns_safe_fallback():
    answer = "The maximum LTV is 75%."

    result = validate_response(
        answer=answer,
        context="",
    )

    assert (
        result
        == "The available documents do not contain enough "
        "information to answer this question."
    )


def test_empty_answer_returns_safe_fallback():
    context = "The maximum LTV is 75%."

    result = validate_response(
        answer="",
        context=context,
    )

    assert (
        result
        == "The system could not generate a reliable answer "
        "from the available documents."
    )


def test_whitespace_is_removed():
    answer = "  The maximum LTV is 75%.  "
    context = "The maximum LTV is 75%."

    result = validate_response(
        answer=answer,
        context=context,
    )

    assert result == "The maximum LTV is 75%."