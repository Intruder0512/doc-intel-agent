"""
Streamlit UI: upload PDFs (auto-ingested into the knowledge base) and chat with the agent.
Shows which of the 3 agents (Cache / Search / Fallback) answered each question.
"""
import streamlit as st
from google.cloud import storage

from src import config
from src.ingest import ingest_document
from src.agents import answer_query

st.set_page_config(page_title="PMP Document Intelligence Agent", layout="wide")
st.title("📄 PMP Document Intelligence Agent")
st.caption("RAG + 3-agent orchestration on GCP — Cache Agent, Search Agent (your books), Fallback Agent (general PMP knowledge).")

# ---------- Sidebar: upload ----------

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader("Add PDFs to the knowledge base", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Ingest uploaded files"):
        storage_client = storage.Client()
        bucket = storage_client.bucket(config.BUCKET_NAME)

        for uploaded_file in uploaded_files:
            gcs_path = f"uploads/{uploaded_file.name}"
            blob = bucket.blob(gcs_path)

            with st.spinner(f"Uploading {uploaded_file.name} to Cloud Storage..."):
                blob.upload_from_file(uploaded_file, content_type="application/pdf")

            gcs_uri = f"gs://{config.BUCKET_NAME}/{gcs_path}"
            with st.spinner(f"Extracting, chunking, embedding {uploaded_file.name}... (large scanned books can take several minutes)"):
                try:
                    ingest_document(gcs_uri)
                    st.success(f"Indexed {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to ingest {uploaded_file.name}: {e}")

        from src import agents
        agents._metadata_cache = {}

    st.divider()
    st.caption("Auto-ingestion is also live via a Cloud Function — files uploaded directly to the bucket get indexed automatically.")
    st.divider()
    st.subheader("Agents in this pipeline")
    st.markdown("""
    1. **Cache Agent** — instant answer for repeat questions
    2. **Search Agent** — retrieves from your uploaded PMP books
    3. **Fallback Agent** — agentic general PMP knowledge when the books don't cover it
    """)

# ---------- Main: chat ----------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

AGENT_ICONS = {
    "Cache Agent (instant repeat answer)": "⚡",
    "Search Agent (your uploaded PMP books)": "📚",
    "Fallback Agent (general PMP knowledge, not from your books)": "🧠",
}

def _render_timings(total_ms, timings):
    if total_ms is None:
        return
    parts = [f"**{total_ms}ms total**"]
    for label, ms in (timings or {}).items():
        parts.append(f"{label.replace('_ms', '')}: {ms}ms")
    st.caption("⏱️ " + " · ".join(parts))


for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("agent_used"):
            icon = AGENT_ICONS.get(entry["agent_used"], "🤖")
            st.caption(f"{icon} Answered by: {entry['agent_used']}")
        if entry.get("sources"):
            st.caption("Sources: " + ", ".join(entry["sources"]))
        _render_timings(entry.get("total_latency_ms"), entry.get("timings_ms"))

user_query = st.chat_input("Ask a PMP question...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Routing through agents..."):
            try:
                result = answer_query(user_query)
                st.markdown(result["answer"])
                icon = AGENT_ICONS.get(result["agent_used"], "🤖")
                st.caption(f"{icon} Answered by: {result['agent_used']}")
                if result["sources"]:
                    st.caption("Sources: " + ", ".join(result["sources"]))
                _render_timings(result.get("total_latency_ms"), result.get("timings_ms"))
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "agent_used": result["agent_used"],
                    "total_latency_ms": result.get("total_latency_ms"),
                    "timings_ms": result.get("timings_ms"),
                })
            except Exception as e:
                st.error(f"Something went wrong: {e}")
