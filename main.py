from app.classifier import (
    analyze_ticket_baseline,
    analyze_ticket_with_ai,
    load_ai_model,
)


def print_analysis(
    title: str,
    result: dict[str, object],
) -> None:

    print(f"\n{title}")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main() -> None:

    print("AI Helpdesk Ticket Assistant")

    ticket_text = input(
        "Please enter the ticket text: "
    ).strip()

    if not ticket_text:
        print("\nError:")
        print("Ticket text cannot be empty.")
        return

    baseline_result = analyze_ticket_baseline(ticket_text)

    print_analysis(
        "Baseline analysis:",
        baseline_result,
    )

    print("\nLoading AI model...")
    classifier = load_ai_model()
    print("Model loaded.")

    ai_result = analyze_ticket_with_ai(
        classifier,
        ticket_text,
    )

    print_analysis(
        "AI analysis:",
        ai_result,
    )


if __name__ == "__main__":
    main()