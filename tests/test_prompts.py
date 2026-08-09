from src.llm.prompts import loan_advisory_prompt


def test_prompt_contains_context_and_question():
    messages = loan_advisory_prompt.format_messages(
        context="Maximum LTV is 75%.",
        question="What is the maximum LTV?",
    )

    prompt_text = "\n".join(
        message.content for message in messages
    )

    assert "Maximum LTV is 75%." in prompt_text
    assert "What is the maximum LTV?" in prompt_text
    assert "Do not invent loan policies" in prompt_text