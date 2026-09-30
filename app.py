import streamlit as st

from src.agent import run_agent


MODEL = "PetrosStav/gemma3-tools:4b"


st.set_page_config(
    page_title="Clinical Copilot",
    page_icon="",
    layout="wide",
)


st.title("Clinical Copilot")

st.markdown(
    """
    **Agentic RAG for clinical and scientific literature**

    Ask questions about the clinical knowledge base and scientific papers.
    """
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("Clinical Copilot")

    st.markdown(
        """
        **Architecture**

        - Local LLM
        - Agentic RAG
        - Semantic retrieval
        - Scientific papers
        - Tool calling
        - Bibliographic metadata
        """
    )

    st.divider()

    if st.button(
        "New conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Model: PetrosStav/gemma3-tools:4b"
    )


# ---------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("sources"):

                with st.expander("📚 Sources"):

                    for i, source in enumerate(
                        message["sources"],
                        1,
                    ):

                        st.markdown(
                            f"**{i}. {source['title']}**"
                        )

                        if source.get("year"):
                            st.write(
                                f"Year: {source['year']}"
                            )

                        if source.get("pmid"):
                            st.write(
                                f"PMID: {source['pmid']}"
                            )

                        if source.get("doi"):
                            st.write(
                                f"DOI: {source['doi']}"
                            )

            if message.get("trace"):

                with st.expander("Agent activity"):

                    for item in message["trace"]:

                        st.write(
                            f"Step {item['step']}: "
                            f"`{item['tool']}`"
                        )

                        if item.get("arguments"):
                            st.json(item["arguments"])


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask Clinical Copilot a question..."
)


if question:

    # Show user message immediately.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Build conversational history for the agent.
    conversation_history = []

    for message in st.session_state.messages[:-1]:

        conversation_history.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    # Generate response.
    with st.chat_message("assistant"):

        with st.spinner(
            "Clinical Copilot is searching the literature..."
        ):

            result = run_agent(
                question=question,
                model=MODEL,
                conversation_history=conversation_history,
            )

        st.markdown(result["answer"])

        # Sources
        if result["sources"]:

            with st.expander("Sources", expanded=True):

                for i, source in enumerate(
                    result["sources"],
                    1,
                ):

                    st.markdown(
                        f"**{i}. {source['title']}**"
                    )

                    if source.get("year"):
                        st.write(
                            f"Year: {source['year']}"
                        )

                    if source.get("pmid"):
                        st.write(
                            f"PMID: {source['pmid']}"
                        )

                    if source.get("doi"):
                        st.write(
                            f"DOI: {source['doi']}"
                        )

        # Agent trace
        if result["trace"]:

            with st.expander("🔧 Agent activity"):

                for item in result["trace"]:

                    st.write(
                        f"Step {item['step']}: "
                        f"`{item['tool']}`"
                    )

                    if item.get("arguments"):
                        st.json(item["arguments"])

    # Store assistant response.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "trace": result["trace"],
        }
    )