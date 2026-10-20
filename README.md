# GCP Document Intelligence Agent

A hands-on RAG + multi-agent system on Google Cloud that ingests PDFs (including scanned books), answers questions with citations, and keeps its knowledge base updated as new documents are uploaded.

> **Status: reference build, not deployed.** I built and tested this end to end on GCP, then deleted the project to stop an always-on Vector Search endpoint from billing by the hour. The code is kept as a reference; running it needs your own GCP project and the resource IDs in `src/config.py`.

## Architecture

```
PDF upload (Cloud Storage: uploads/)
   -> Eventarc -> Cloud Function (main.py)
   -> src/ingest.py: split large PDFs, Document AI OCR, chunk (1000 words / 150 overlap),
      Vertex AI embeddings (text-embedding-005, 768 dims), upsert to Vector Search,
      chunk metadata saved to Cloud Storage

Question -> src/agents.py (three agents, each response reports which one answered)
   1. Cache Agent     Firestore, keyed by normalized question (survives restarts)
   2. Search Agent    Vector Search retrieval + Gemini answer with citations
   3. Fallback Agent  Gemini general knowledge when book results are weak
UI: Streamlit (app.py) with upload, chat and per-stage latency
```

## Key design choices

- **Two-layer confidence filter:** a cheap distance threshold first, then an LLM check that the draft actually answers the question; a keyword safety net covers cases where LLM self-judgment was unreliable.
- **Planner step** classifies a question as lookup, compare or summarize and changes retrieval depth.
- **Persistent cache** in Firestore for instant repeat answers.
- **Latency tracking** per stage (cache, planner, retrieval, generation, confidence check).
- **Scanned PDFs over Document AI page limits** are split into sub-batches (pypdf) and processed in imageless mode.

## Services used

Cloud Storage, Document AI, Vertex AI (embeddings, Gemini 2.5 Flash, Vector Search), Firestore, Eventarc and Cloud Functions (2nd gen), IAM service accounts, Streamlit.

## Files

| File | Purpose |
| --- | --- |
| `src/config.py` | Project, model and Vector Search settings (IDs are from the deleted project) |
| `src/ingest.py` | PDF to chunks to embeddings to index |
| `src/agents.py` | Planner, cache, search and fallback agents |
| `app.py` | Streamlit UI |
| `main.py` | Cloud Function entry point (auto-ingest on upload) |
| `clear_cache.py`, `clear_old_data.py` | Maintenance scripts |
| `index-metadata.json` | Vector Search index config |

## Lessons learned

- Vector Search endpoints bill per hour while deployed; tear them down when not demoing.
- Event-triggered functions have a timeout ceiling; long OCR jobs may need manual ingestion.
- Honest note: orchestration is custom Python, not LangChain or LangGraph. See my FinOps project for a LangGraph build.
