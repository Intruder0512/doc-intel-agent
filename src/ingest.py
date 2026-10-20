"""
Ingestion pipeline: PDF -> Document AI extraction -> chunking -> embeddings -> Vector Search upsert.
Handles PDFs over Document AI's 30-page sync limit by splitting into sub-batches.
"""
import hashlib
import io
from google.cloud import documentai
from google.cloud import storage
from pypdf import PdfReader, PdfWriter
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform

from src import config

DOCAI_PAGE_LIMIT = 30
SAFE_BATCH_SIZE = 28  # a little under the 30-page imageless-mode limit


def _split_pdf_bytes(pdf_bytes: bytes, batch_size: int = SAFE_BATCH_SIZE) -> list[bytes]:
    """Split PDF bytes into a list of smaller PDF byte-strings, each under batch_size pages."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    if total_pages <= DOCAI_PAGE_LIMIT:
        return [pdf_bytes]

    chunks = []
    for start in range(0, total_pages, batch_size):
        writer = PdfWriter()
        for page in reader.pages[start:start + batch_size]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
    return chunks


def _ocr_pdf_bytes(client: documentai.DocumentProcessorServiceClient, pdf_bytes: bytes) -> str:
    """Run Document AI OCR on a single PDF byte-string (must be <=30 pages, imageless mode)."""
    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(
        name=config.DOCAI_PROCESSOR_PATH,
        raw_document=raw_document,
        imageless_mode=True,  # required to raise the sync page limit from 15 to 30
    )
    result = client.process_document(request=request)
    return result.document.text


def extract_text_from_pdf(gcs_uri: str) -> str:
    """Extract text from a PDF stored in Cloud Storage using Document AI, splitting if needed."""
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{config.DOCAI_LOCATION}-documentai.googleapis.com"}
    )

    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    pdf_bytes = blob.download_as_bytes()

    batches = _split_pdf_bytes(pdf_bytes)
    if len(batches) > 1:
        print(f"  Document exceeds {DOCAI_PAGE_LIMIT} pages, splitting into {len(batches)} batches...")

    texts = []
    for i, batch_bytes in enumerate(batches):
        if len(batches) > 1:
            print(f"  OCR batch {i + 1}/{len(batches)}...")
        texts.append(_ocr_pdf_bytes(client, batch_bytes))

    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Simple sliding-window chunker with overlap, splitting on whitespace boundaries."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of text chunks using Vertex AI's embedding model."""
    vertexai.init(project=config.PROJECT_ID, location=config.REGION)
    model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)

    embeddings = []
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        results = model.get_embeddings(batch)
        embeddings.extend([r.values for r in results])
    return embeddings


def _chunk_id(source_doc: str, chunk_index: int) -> str:
    """Deterministic ID so re-ingesting the same doc/chunk overwrites rather than duplicates."""
    raw = f"{source_doc}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def upsert_to_vector_search(chunk_texts: list[str], embeddings: list[list[float]], source_doc: str):
    """Upsert chunk embeddings into the streaming Vector Search index."""
    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
    index = aiplatform.MatchingEngineIndex(index_name=config.VECTOR_INDEX_RESOURCE_NAME)

    datapoints = []
    for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
        datapoints.append({
            "datapoint_id": _chunk_id(source_doc, i),
            "feature_vector": embedding,
        })

    index.upsert_datapoints(datapoints=datapoints)
    _save_chunk_metadata(source_doc, chunk_texts)


def _save_chunk_metadata(source_doc: str, chunk_texts: list[str]):
    """Store chunk text + source alongside datapoint IDs in Cloud Storage as a simple lookup file."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.BUCKET_NAME)

    import json
    metadata = [
        {"id": _chunk_id(source_doc, i), "text": text, "source": source_doc}
        for i, text in enumerate(chunk_texts)
    ]
    safe_name = source_doc.replace("gs://", "").replace("/", "_")
    blob = bucket.blob(f"_metadata/{safe_name}.json")
    blob.upload_from_string(json.dumps(metadata), content_type="application/json")


def ingest_document(gcs_uri: str):
    """Full pipeline for one document: extract -> chunk -> embed -> upsert."""
    print(f"Extracting text from {gcs_uri} ...")
    text = extract_text_from_pdf(gcs_uri)
    print(f"Extracted {len(text)} characters.")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = embed_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings.")

    upsert_to_vector_search(chunks, embeddings, source_doc=gcs_uri)
    print("Upserted to Vector Search and saved chunk metadata.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.ingest gs://bucket/path/to/file.pdf")
        sys.exit(1)
    ingest_document(sys.argv[1])
