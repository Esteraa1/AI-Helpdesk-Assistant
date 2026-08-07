from fastapi import FastAPI

from app.classifier import (
    analyze_ticket_with_ai,
    load_ai_model,
)
from app.database import (
    create_tickets_table,
    get_all_tickets,
    save_ticket,
)
from app.schemas import (
    StoredTicketResponse,
    TicketRequest,
    TicketResponse,
)


app = FastAPI(
    title="AI Helpdesk Ticket Assistant",
    description="API for analyzing helpdesk tickets.",
    version="0.3.0",
)


# Tworzymy tabelę podczas uruchamiania aplikacji.
create_tickets_table()


# Model AI jest ładowany jeden raz.
print("Loading AI model...")
classifier = load_ai_model()
print("AI model loaded.")


@app.get("/")
def read_root() -> dict[str, str]:
    """Zwraca informację, że API działa."""

    return {
        "message": "AI Helpdesk Ticket Assistant API is running.",
        "status": "ok",
    }


@app.post(
    "/analyze-ticket",
    response_model=TicketResponse,
)
def analyze_ticket_endpoint(
    ticket: TicketRequest,
) -> dict[str, object]:
    """Analizuje zgłoszenie i zapisuje wynik w bazie."""

    analysis_result = analyze_ticket_with_ai(
        classifier,
        ticket.text,
    )

    save_ticket(
        ticket.text,
        analysis_result,
    )

    return analysis_result


@app.get(
    "/tickets",
    response_model=list[StoredTicketResponse],
)
def get_tickets_endpoint() -> list[dict[str, object]]:
    """Zwraca historię przeanalizowanych zgłoszeń."""

    tickets = get_all_tickets()

    return tickets