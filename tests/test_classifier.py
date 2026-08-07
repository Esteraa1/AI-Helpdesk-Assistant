from app.classifier import (
    check_human_review,
    create_summary,
    detect_category_baseline,
)


def test_login_problem_baseline():

    ticket_text = "I cannot reset my password."

    category = detect_category_baseline(ticket_text)

    assert category == "login problem"


def test_create_summary_for_short_text():

    ticket_text = "My monitor does not turn on."

    summary = create_summary(ticket_text)

    assert summary == "My monitor does not turn on."


def test_create_summary_for_long_text():

    ticket_text = "a" * 150

    summary = create_summary(
        ticket_text,
        maximum_length=100,
    )

    assert len(summary) == 100
    assert summary.endswith("...")


def test_human_review_when_confidence_is_low():

    result = check_human_review(
        category_confidence=0.55,
        priority_confidence=0.80,
    )

    assert result is True


def test_no_human_review_when_confidence_is_high():

    result = check_human_review(
        category_confidence=0.85,
        priority_confidence=0.75,
    )

    assert result is False