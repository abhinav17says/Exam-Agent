"""Run with: python -m streamlit run exam_agent_frontend.py"""
import hashlib
import logging
import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import exam_agent_backend as backend

st.set_page_config(page_title="Exam Agent", page_icon="📚", layout="wide")
st.title("📚 Exam Agent")
st.caption("Turn an exam notification into a conversation. Explore eligibility, dates, syllabus and more.")

for key, value in {"agent": None, "messages": [], "document_id": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def show_error(exc, action):
    logging.exception("Exam Agent failed to %s", action)
    if isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        st.error(f"Couldn't {action} ({type(exc).__name__}). "
                 "Check Manage app → Logs on Streamlit Cloud, or your local terminal, for the traceback.")


with st.sidebar:
    st.header("Your document")
    configured_key = os.getenv("OPENAI_API_KEY", "")
    if not configured_key:
        try:
            configured_key = st.secrets.get("OPENAI_API_KEY", "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass
    entered_key = st.text_input("OpenAI API key", type="password",
                               help="Leave empty to use the key in .env or Streamlit secrets.")
    api_key = entered_key or configured_key
    if api_key:
        st.caption("API key available")
    else:
        st.info("Enter an API key to process a PDF and ask questions.")
    uploaded = st.file_uploader("Upload an exam notification (PDF or Docling Markdown)", type=["pdf", "md"])
    low_memory = st.checkbox("Lower-memory PDF processing", value=True,
                             help="Skips OCR and processes smaller batches while retaining table parsing. "
                                  "Use for PDFs with selectable text. Turn off for scanned documents.")
    st.caption("If PDF processing crashes this host, convert the PDF locally with export_pdf.py "
               "and upload the resulting .md file. This avoids loading PDF models here.")
    sample = Path(__file__).with_name("noti_cds.pdf")
    use_sample = st.checkbox("Use the included CDS notification", disabled=not sample.exists())
    data, filename = None, None
    if use_sample and sample.exists():
        data, filename = sample.read_bytes(), sample.name
    elif uploaded is not None:
        data, filename = uploaded.getvalue(), uploaded.name
    selected_id = hashlib.sha256(data).hexdigest() if data is not None else None
    if selected_id and st.session_state.document_id and selected_id != st.session_state.document_id:
        st.warning("Process the selected PDF to switch documents. Chat still uses the active document below.")
    if st.button("Process document", type="primary", use_container_width=True,
                 disabled=data is None or not api_key):
        try:
            with st.spinner("Reading the PDF and preparing document sections..."):
                if filename.lower().endswith(".md"):
                    agent = backend.process_uploaded_markdown(data, filename, api_key)
                else:
                    agent = backend.process_uploaded_pdf(data, filename, api_key, low_memory=low_memory)
            st.session_state.agent = agent
            st.session_state.document_id = selected_id
            st.session_state.messages = []
            st.success("Document ready. Ask your first question!")
        except Exception as exc:
            show_error(exc, "process the document")
    st.caption("First use may download document parsing and reranking models. "
               "Document text is sent to OpenAI for embeddings and answers; API usage is billed to your key.")
    if st.session_state.agent:
        st.divider()
        st.write("**Active document**")
        st.write(st.session_state.agent.filename)
        st.caption(f"{st.session_state.agent.chunk_count} searchable sections")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
        transcript = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.messages)
        st.download_button("Download conversation", transcript, file_name="exam-agent-chat.txt",
                           mime="text/plain", disabled=not transcript, use_container_width=True)


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View supporting document excerpts"):
                for i, source in enumerate(message["sources"], 1):
                    meta = source["metadata"]
                    section = meta.get("Header 3") or meta.get("Header 2") or meta.get("Header 1") or "General Content"
                    st.markdown(f"**[{i}] {section}**")
                    st.text(source["content"])


if not st.session_state.agent:
    st.info("Upload a PDF or Docling Markdown file in the sidebar, or choose the included CDS notification, then select Process document.")

suggestion = None
if not st.session_state.messages:
    st.subheader("What would you like to know?")
    prompts = ["What are the eligibility criteria?", "What are the important dates?", "Summarize the exam pattern and syllabus."]
    for column, prompt in zip(st.columns(3), prompts):
        if column.button(prompt, use_container_width=True, disabled=st.session_state.agent is None):
            suggestion = prompt

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask about your exam notification...", disabled=st.session_state.agent is None)
if question or suggestion:
    question = question or suggestion
    history = [HumanMessage(content=m["content"]) if m["role"] == "user"
               else AIMessage(content=m["content"]) for m in st.session_state.messages]
    render_message({"role": "user", "content": question})
    try:
        with st.spinner("Searching your document..."):
            answer, sources = st.session_state.agent.ask(question, history)
        st.session_state.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "sources": sources},
        ])
        st.rerun()
    except Exception as exc:
        show_error(exc, "answer the question")
        st.info("Your conversation was preserved. Submit the question again to retry.")
