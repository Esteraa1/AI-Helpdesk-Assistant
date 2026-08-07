from pydantic import BaseModel, Field, field_validator


class TicketRequest(BaseModel):
    """Opisuje dane wysyłane do analizy zgłoszenia."""

    text: str = Field(
        min_length=1,
        max_length=2000,
        description="Treść zgłoszenia helpdesk.",
        examples=[
            "I cannot log in to my account."
        ],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, text: str) -> str:
        """Usuwa spacje i odrzuca pustą treść zgłoszenia."""

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "Ticket text cannot be empty."
            )

        return cleaned_text


class TicketResponse(BaseModel):
    """Opisuje wynik analizy zgłoszenia."""

    category: str

    category_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    priority: str

    priority_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    summary: str
    suggested_response: str
    requires_human_review: bool


class StoredTicketResponse(BaseModel):
    """Opisuje zgłoszenie zapisane w bazie danych."""

    id: int
    ticket_text: str
    category: str

    category_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    priority: str

    priority_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    summary: str
    suggested_response: str
    requires_human_review: bool
    created_at: str