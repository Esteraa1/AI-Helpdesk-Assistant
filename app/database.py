import sqlite3
from pathlib import Path


# Ścieżka do pliku bazy danych
DATABASE_PATH = Path(__file__).resolve().parent.parent / "tickets.db"


def create_tickets_table() -> None:
    """Tworzy tabelę tickets, jeżeli jeszcze nie istnieje."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_text TEXT NOT NULL,
                category TEXT NOT NULL,
                category_confidence REAL,
                priority TEXT NOT NULL,
                priority_confidence REAL,
                summary TEXT NOT NULL,
                suggested_response TEXT NOT NULL,
                requires_human_review INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_ticket(
    ticket_text: str,
    analysis_result: dict[str, object],
) -> int:
    """Zapisuje przeanalizowane zgłoszenie w bazie i zwraca jego ID."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO tickets (
                ticket_text,
                category,
                category_confidence,
                priority,
                priority_confidence,
                summary,
                suggested_response,
                requires_human_review
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_text,
                analysis_result["category"],
                analysis_result["category_confidence"],
                analysis_result["priority"],
                analysis_result["priority_confidence"],
                analysis_result["summary"],
                analysis_result["suggested_response"],
                int(analysis_result["requires_human_review"]),
            ),
        )

        ticket_id = cursor.lastrowid

    return ticket_id


def get_all_tickets() -> list[dict[str, object]]:
    """Pobiera wszystkie zapisane zgłoszenia z bazy."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                ticket_text,
                category,
                category_confidence,
                priority,
                priority_confidence,
                summary,
                suggested_response,
                requires_human_review,
                created_at
            FROM tickets
            ORDER BY id DESC
            """
        ).fetchall()

    tickets = []

    for row in rows:
        ticket = dict(row)

        ticket["requires_human_review"] = bool(
            ticket["requires_human_review"]
        )

        tickets.append(ticket)

    return tickets