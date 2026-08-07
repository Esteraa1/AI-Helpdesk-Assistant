import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="AI Helpdesk Ticket Assistant",
    page_icon="🎫",
)


st.title("🎫 AI Helpdesk Ticket Assistant")

st.write(
    "Enter a helpdesk ticket below. "
    "The ticket will be sent to the FastAPI backend for analysis."
)


ticket_text = st.text_area(
    "Ticket text",
    placeholder=(
        "Example: I cannot log in to my account. "
        "Password reset does not work."
    ),
    height=150,
)


if st.button("Analyze ticket"):
    cleaned_text = ticket_text.strip()

    if not cleaned_text:
        st.warning(
            "Please enter a ticket before analyzing."
        )

    else:
        try:
            response = requests.post(
                f"{API_URL}/analyze-ticket",
                json={
                    "text": cleaned_text,
                },
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()

                st.success(
                    "Ticket analyzed successfully."
                )

                st.subheader("Analysis result")

                st.write(
                    "**Category:**",
                    result["category"],
                )

                st.write(
                    "**Category confidence:**",
                    f"{result['category_confidence']:.2%}",
                )

                st.write(
                    "**Priority:**",
                    result["priority"],
                )

                st.write(
                    "**Priority confidence:**",
                    f"{result['priority_confidence']:.2%}",
                )

                st.write(
                    "**Summary:**",
                    result["summary"],
                )

                st.write(
                    "**Suggested response:**",
                    result["suggested_response"],
                )

                if result["requires_human_review"]:
                    st.warning(
                        "This ticket requires human review."
                    )
                else:
                    st.success(
                        "Human review is not required."
                    )

            else:
                st.error(
                    f"API returned error: "
                    f"{response.status_code}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Make sure the backend is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The request took too long."
            )


st.divider()

st.subheader("Ticket history")


try:
    history_response = requests.get(
        f"{API_URL}/tickets",
        timeout=10,
    )

    if history_response.status_code == 200:
        tickets = history_response.json()

        if not tickets:
            st.info("No tickets have been saved yet.")

        else:
            for ticket in tickets:
                title = (
                    f"Ticket #{ticket['id']} - "
                    f"{ticket['category']}"
                )

                with st.expander(title):
                    st.write(
                        "**Original ticket:**",
                        ticket["ticket_text"],
                    )

                    st.write(
                        "**Category:**",
                        ticket["category"],
                    )

                    st.write(
                        "**Category confidence:**",
                        f"{ticket['category_confidence']:.2%}",
                    )

                    st.write(
                        "**Priority:**",
                        ticket["priority"],
                    )

                    st.write(
                        "**Priority confidence:**",
                        f"{ticket['priority_confidence']:.2%}",
                    )

                    st.write(
                        "**Summary:**",
                        ticket["summary"],
                    )

                    st.write(
                        "**Suggested response:**",
                        ticket["suggested_response"],
                    )

                    st.write(
                        "**Created at:**",
                        ticket["created_at"],
                    )

                    if ticket["requires_human_review"]:
                        st.warning(
                            "Requires human review."
                        )
                    else:
                        st.success(
                            "Human review is not required."
                        )

    else:
        st.error(
            f"Could not load ticket history. "
            f"API returned: {history_response.status_code}"
        )

except requests.exceptions.ConnectionError:
    st.error(
        "Cannot load ticket history because "
        "FastAPI is not running."
    )

except requests.exceptions.Timeout:
    st.error(
        "Loading ticket history took too long."
    )