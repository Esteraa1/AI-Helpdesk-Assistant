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
    version="0.4.0",
)


create_tickets_table()

classifier = None


def get_classifier():

    global classifier

    if classifier is None:
        print("Loading AI model...")
        classifier = load_ai_model()
        print("AI model loaded.")

    return classifier


@app.get("/")
def read_root() -> dict[str, str]:

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

    ai_classifier = get_classifier()

    analysis_result = analyze_ticket_with_ai(
        ai_classifier,
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

    tickets = get_all_tickets()

    return tickets