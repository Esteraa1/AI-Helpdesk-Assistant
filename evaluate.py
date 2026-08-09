import csv
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from app.classifier import (
    detect_category_baseline,
    detect_category_with_ai,
    detect_priority_baseline,
    detect_priority_with_ai,
    load_ai_model,
)


DATA_PATH = Path("data/evaluation_tickets.csv")


CATEGORY_LABELS = [
    "login problem",
    "software issue",
    "hardware issue",
    "network issue",
    "billing issue",
    "other",
]


PRIORITY_LABELS = [
    "low priority",
    "medium priority",
    "high priority",
    "critical priority",
]


def calculate_accuracy(
    correct_predictions: int,
    total_predictions: int,
) -> float:

    if total_predictions == 0:
        return 0.0

    return correct_predictions / total_predictions


def print_errors(
    title: str,
    errors: list[dict[str, object]],
) -> None:

    print(f"\n=== {title} ===")

    if not errors:
        print("No errors.")
        return

    for error_number, error in enumerate(errors, start=1):
        print(f"\nError {error_number}")

        print("Text:")
        print(error["text"])

        print("Expected:")
        print(error["expected"])

        print("Predicted:")
        print(error["predicted"])

        if error["confidence"] is not None:
            print(
                "Confidence:",
                f"{error['confidence']:.4f}",
            )

def save_confusion_matrix(
    expected: list[str],
    predicted: list[str],
    labels: list[str],
    title: str,
    file_path: str,
) -> None:

    matrix = confusion_matrix(
        expected,
        predicted,
        labels=labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot(
        xticks_rotation=45,
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def main() -> None:

    print("Loading evaluation data...")

    with open(
        DATA_PATH,
        mode="r",
        encoding="utf-8",
    ) as file:
        tickets = list(csv.DictReader(file))

    total_tickets = len(tickets)

    print(f"Loaded {total_tickets} tickets.")

    print("\nLoading AI model...")
    classifier = load_ai_model()
    print("AI model loaded.\n")

    baseline_category_correct = 0
    baseline_priority_correct = 0

    ai_category_correct = 0
    ai_priority_correct = 0

    baseline_category_errors = []
    baseline_priority_errors = []

    ai_category_errors = []
    ai_priority_errors = []

    ai_category_expected = []
    ai_category_predicted = []

    ai_priority_expected = []
    ai_priority_predicted = []

    for index, ticket in enumerate(tickets, start=1):
        text = ticket["text"]

        expected_category = ticket["expected_category"]
        expected_priority = ticket["expected_priority"]

        baseline_category = detect_category_baseline(text)
        baseline_priority = detect_priority_baseline(text)

        ai_category, category_confidence = (
            detect_category_with_ai(
                classifier,
                text,
            )
        )

        ai_priority, priority_confidence = (
            detect_priority_with_ai(
                classifier,
                text,
            )
        )

        ai_category_expected.append(expected_category)
        ai_category_predicted.append(ai_category)

        ai_priority_expected.append(expected_priority)
        ai_priority_predicted.append(ai_priority)

        if baseline_category == expected_category:
            baseline_category_correct += 1
        else:
            baseline_category_errors.append(
                {
                    "text": text,
                    "expected": expected_category,
                    "predicted": baseline_category,
                    "confidence": None,
                }
            )

        if baseline_priority == expected_priority:
            baseline_priority_correct += 1
        else:
            baseline_priority_errors.append(
                {
                    "text": text,
                    "expected": expected_priority,
                    "predicted": baseline_priority,
                    "confidence": None,
                }
            )

        if ai_category == expected_category:
            ai_category_correct += 1
        else:
            ai_category_errors.append(
                {
                    "text": text,
                    "expected": expected_category,
                    "predicted": ai_category,
                    "confidence": category_confidence,
                }
            )

        if ai_priority == expected_priority:
            ai_priority_correct += 1
        else:
            ai_priority_errors.append(
                {
                    "text": text,
                    "expected": expected_priority,
                    "predicted": ai_priority,
                    "confidence": priority_confidence,
                }
            )

        print(
            f"Processed ticket "
            f"{index}/{total_tickets}"
        )

    baseline_category_accuracy = calculate_accuracy(
        baseline_category_correct,
        total_tickets,
    )

    baseline_priority_accuracy = calculate_accuracy(
        baseline_priority_correct,
        total_tickets,
    )

    ai_category_accuracy = calculate_accuracy(
        ai_category_correct,
        total_tickets,
    )

    ai_priority_accuracy = calculate_accuracy(
        ai_priority_correct,
        total_tickets,
    )

    print("\nEvaluation results:")

    print("\nBaseline:")

    print(
        "Category accuracy:",
        f"{baseline_category_accuracy:.2%}",
    )

    print(
        "Priority accuracy:",
        f"{baseline_priority_accuracy:.2%}",
    )

    print("\nAI model:")

    print(
        "Category accuracy:",
        f"{ai_category_accuracy:.2%}",
    )

    print(
        "Priority accuracy:",
        f"{ai_priority_accuracy:.2%}",
    )

    print_errors(
        "Baseline category errors",
        baseline_category_errors,
    )

    print_errors(
        "Baseline priority errors",
        baseline_priority_errors,
    )

    print_errors(
        "AI category errors",
        ai_category_errors,
    )

    print_errors(
        "AI priority errors",
        ai_priority_errors,
    )

    print("\nAI category classification report:")

    print(
        classification_report(
            ai_category_expected,
            ai_category_predicted,
            labels=CATEGORY_LABELS,
            zero_division=0,
        )
    )

    print("\nAI priority classification report:")

    print(
        classification_report(
            ai_priority_expected,
            ai_priority_predicted,
            labels=PRIORITY_LABELS,
            zero_division=0,
        )
    )

    print("\nAI category confusion matrix:")

    print("Label order:")
    print(CATEGORY_LABELS)

    print(
        confusion_matrix(
            ai_category_expected,
            ai_category_predicted,
            labels=CATEGORY_LABELS,
        )
    )

    print("\nAI priority confusion matrix:")

    print("Label order:")
    print(PRIORITY_LABELS)

    print(
        confusion_matrix(
            ai_priority_expected,
            ai_priority_predicted,
            labels=PRIORITY_LABELS,
        )
    )

    save_confusion_matrix(
        ai_category_expected,
        ai_category_predicted,
        CATEGORY_LABELS,
        "AI Category Confusion Matrix",
        "screenshots/category_confusion_matrix.png",
    )

    save_confusion_matrix(
        ai_priority_expected,
        ai_priority_predicted,
        PRIORITY_LABELS,
        "AI Priority Confusion Matrix",
        "screenshots/priority_confusion_matrix.png",
    )

    print("\nConfusion matrix images saved in screenshots/")


if __name__ == "__main__":
    main()