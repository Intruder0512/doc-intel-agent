"""
Cloud Function (2nd gen): triggered on Cloud Storage object finalize.
Auto-ingests any PDF uploaded to uploads/ into the knowledge base.
"""
import functions_framework
from cloudevents.http import CloudEvent

from src.ingest import ingest_document


@functions_framework.cloud_event
def on_document_upload(cloud_event: CloudEvent):
    data = cloud_event.data
    bucket = data["bucket"]
    name = data["name"]

    # Avoid re-triggering on our own metadata writes, and skip non-PDFs.
    if not name.startswith("uploads/") or not name.lower().endswith(".pdf"):
        print(f"Skipping {name} (not a PDF under uploads/)")
        return

    gcs_uri = f"gs://{bucket}/{name}"
    print(f"Auto-ingesting {gcs_uri} ...")
    try:
        ingest_document(gcs_uri)
        print(f"Successfully ingested {gcs_uri}")
    except Exception as e:
        print(f"Failed to ingest {gcs_uri}: {e}")
        raise
