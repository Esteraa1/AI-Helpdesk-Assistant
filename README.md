# AI Helpdesk Ticket Assistant

AI-powered helpdesk ticket analysis application built with Python.

The application can:

- classify helpdesk ticket categories using zero-shot classification,
- classify ticket priority,
- return confidence scores,
- flag uncertain predictions for human review,
- generate a simple ticket summary,
- suggest a response based on the detected category,
- expose the analysis through FastAPI,
- store analyzed tickets in SQLite,
- provide a simple Streamlit interface.

## Technologies

- Python
- FastAPI
- Pydantic
- Hugging Face Transformers
- PyTorch
- SQLite
- Streamlit
- Requests

## Project status

Work in progress.

Currently implemented:

- rule-based baseline classifier
- zero-shot AI classification
- FastAPI backend
- SQLite ticket storage
- Streamlit interface

Planned:

- pytest tests
- baseline vs AI evaluation
- improved documentation