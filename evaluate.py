import csv
from pathlib import Path

from app.classifier import (
    detect_category_baseline,
    detect_category_with_ai,
    detect_priority_baseline,
    detect_priority_with_ai,
    load_ai_model,
)


DATA_PATH = Path("data/evaluation_tickets.csv")


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


if __name__ == "__main__":
    main()