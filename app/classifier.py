from transformers import pipeline


MODEL_NAME = "facebook/bart-large-mnli"

HUMAN_REVIEW_THRESHOLD = 0.60


CATEGORY_KEYWORDS = {
    "login problem": [
        "login",
        "log in",
        "password",
        "account access",
        "locked out",
    ],
    "software issue": [
        "software",
        "application",
        "app",
        "program",
        "error message",
    ],
    "hardware issue": [
        "monitor",
        "screen",
        "keyboard",
        "mouse",
        "laptop",
        "computer",
        "printer",
    ],
    "network issue": [
        "wifi",
        "wi-fi",
        "internet",
        "network",
        "vpn",
        "connection",
    ],
    "billing issue": [
        "payment",
        "invoice",
        "charged",
        "billing",
        "refund",
    ],
}


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


SUGGESTED_RESPONSES = {
    "login problem": (
        "Please confirm whether an error message appears during login."
    ),
    "software issue": (
        "Please provide the name of the application and describe "
        "the error message."
    ),
    "hardware issue": (
        "Please check whether the device is connected correctly "
        "and confirm if it receives power."
    ),
    "network issue": (
        "Please check whether other devices can connect to the network."
    ),
    "billing issue": (
        "Please provide the invoice or payment reference number."
    ),
    "other": (
        "Please provide more details about the problem."
    ),
}


def load_ai_model():

    return pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME,
        device=-1,
    )


def detect_category_baseline(text: str) -> str:

    normalized_text = text.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_text:
                return category

    return "other"


def detect_priority_baseline(text: str) -> str:

    normalized_text = text.lower()

    critical_keywords = [
        "security breach",
        "data loss",
        "production is down",
        "all employees",
        "entire company",
    ]

    high_priority_keywords = [
        "urgent",
        "cannot work",
        "important meeting",
        "in one hour",
        "keeps disconnecting",
    ]

    low_priority_keywords = [
        "question",
        "how do i",
        "when possible",
        "minor issue",
    ]

    for keyword in critical_keywords:
        if keyword in normalized_text:
            return "critical priority"

    for keyword in high_priority_keywords:
        if keyword in normalized_text:
            return "high priority"

    for keyword in low_priority_keywords:
        if keyword in normalized_text:
            return "low priority"

    return "medium priority"


def find_short_label(
    descriptions: dict[str, str],
    selected_description: str,
) -> str:

    for short_label, description in descriptions.items():
        if description == selected_description:
            return short_label

    raise ValueError("The model returned an unknown label.")


def detect_category_with_ai(
    classifier,
    text: str,
) -> tuple[str, float]:

    category_candidates = list(CATEGORY_DESCRIPTIONS.values())

    result = classifier(
        text,
        candidate_labels=category_candidates,
        hypothesis_template="This helpdesk ticket describes {}.",
        multi_label=False,
    )

    selected_description = result["labels"][0]
    category_confidence = float(result["scores"][0])

    selected_category = find_short_label(
        CATEGORY_DESCRIPTIONS,
        selected_description,
    )

    return selected_category, category_confidence


def detect_priority_with_ai(
    classifier,
    text: str,
) -> tuple[str, float]:

    priority_candidates = list(PRIORITY_DESCRIPTIONS.values())

    result = classifier(
        text,
        candidate_labels=priority_candidates,
        hypothesis_template="This helpdesk ticket describes {}.",
        multi_label=False,
    )

    selected_description = result["labels"][0]
    priority_confidence = float(result["scores"][0])

    selected_priority = find_short_label(
        PRIORITY_DESCRIPTIONS,
        selected_description,
    )

    return selected_priority, priority_confidence


def create_summary(
    text: str,
    maximum_length: int = 100,
) -> str:

    cleaned_text = " ".join(text.split())

    if len(cleaned_text) <= maximum_length:
        return cleaned_text

    shortened_text = cleaned_text[: maximum_length - 3]

    return shortened_text + "..."


def get_suggested_response(category: str) -> str:

    return SUGGESTED_RESPONSES[category]


def check_human_review(
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


def analyze_ticket_baseline(text: str) -> dict[str, object]:

    category = detect_category_baseline(text)
    priority = detect_priority_baseline(text)

    return {
        "category": category,
        "category_confidence": None,
        "priority": priority,
        "priority_confidence": None,
        "summary": create_summary(text),
        "suggested_response": get_suggested_response(category),
        "requires_human_review": category == "other",
    }


def analyze_ticket_with_ai(
    classifier,
    text: str,
) -> dict[str, object]:

    category, category_confidence = detect_category_with_ai(
        classifier,
        text,
    )

    priority, priority_confidence = detect_priority_with_ai(
        classifier,
        text,
    )

    human_review = check_human_review(
        category_confidence,
        priority_confidence,
    )

    return {
        "category": category,
        "category_confidence": category_confidence,
        "priority": priority,
        "priority_confidence": priority_confidence,
        "summary": create_summary(text),
        "suggested_response": get_suggested_response(category),
        "requires_human_review": human_review,
    }