from transformers import pipeline


MODEL_NAME = "facebook/bart-large-mnli"

HUMAN_REVIEW_THRESHOLD = 0.60


CATEGORY_DESCRIPTIONS = {
    "login problem": (
        "a problem signing in, resetting a password, "
        "or accessing a user account"
    ),
    "software issue": (
        "a problem with an application, program, "
        "operating system, or production software"
    ),
    "hardware issue": (
        "a problem with a physical device such as a computer, "
        "monitor, keyboard, printer, or laptop"
    ),
    "network issue": (
        "a problem with Wi-Fi, internet, VPN, "
        "or network connectivity"
    ),
    "billing issue": (
        "a problem with an invoice, payment, charge, "
        "billing, or refund"
    ),
    "other": (
        "a helpdesk issue that does not match "
        "the other available categories"
    ),
}


PRIORITY_DESCRIPTIONS = {
    "low priority": (
        "a minor issue or general question that can wait"
    ),
    "medium priority": (
        "an issue affecting one user without an urgent deadline"
    ),
    "high priority": (
        "an urgent issue preventing work, affecting multiple users, "
        "or connected with an important deadline"
    ),
    "critical priority": (
        "a severe outage, security incident, data loss, "
        "or production system failure affecting the whole organization"
    ),
}


def find_short_label(
    descriptions: dict[str, str],
    selected_description: str,
) -> str:

    for short_label, description in descriptions.items():
        if description == selected_description:
            return short_label

    return "other"


def classify_category(classifier, text: str) -> tuple[str, float]:

    category_candidates = list(CATEGORY_DESCRIPTIONS.values())

    result = classifier(
        text,
        candidate_labels=category_candidates,
        hypothesis_template=(
            "This helpdesk ticket describes {}."
        ),
        multi_label=False,
    )

    selected_description = result["labels"][0]
    category_confidence = float(result["scores"][0])

    selected_category = find_short_label(
        CATEGORY_DESCRIPTIONS,
        selected_description,
    )

    return selected_category, category_confidence


def classify_priority(classifier, text: str) -> tuple[str, float]:

    priority_candidates = list(PRIORITY_DESCRIPTIONS.values())

    result = classifier(
        text,
        candidate_labels=priority_candidates,
        hypothesis_template=(
            "This helpdesk ticket describes {}."
        ),
        multi_label=False,
    )

    selected_description = result["labels"][0]
    priority_confidence = float(result["scores"][0])

    selected_priority = find_short_label(
        PRIORITY_DESCRIPTIONS,
        selected_description,
    )

    return selected_priority, priority_confidence


def requires_human_review(
    category_confidence: float,
    priority_confidence: float,
) -> bool:

    category_is_uncertain = (
        category_confidence < HUMAN_REVIEW_THRESHOLD
    )

    priority_is_uncertain = (
        priority_confidence < HUMAN_REVIEW_THRESHOLD
    )

    return category_is_uncertain or priority_is_uncertain


def main() -> None:

    print("Loading AI model...")

    classifier = pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME,
        device=-1,
    )

    print("Model loaded.")

    ticket_text = input(
        "Please enter the ticket text: "
    ).strip()

    if not ticket_text:
        print("\nError:")
        print("Ticket text cannot be empty.")
        return

    category, category_confidence = classify_category(
        classifier,
        ticket_text,
    )

    priority, priority_confidence = classify_priority(
        classifier,
        ticket_text,
    )

    human_review = requires_human_review(
        category_confidence,
        priority_confidence,
    )

    print("\nAI ticket analysis:")
    print(f"Category: {category}")
    print(
        f"Category confidence: "
        f"{category_confidence:.4f}"
    )
    print(f"Priority: {priority}")
    print(
        f"Priority confidence: "
        f"{priority_confidence:.4f}"
    )
    print(
        f"Requires human review: {human_review}"
    )


if __name__ == "__main__":
    main()