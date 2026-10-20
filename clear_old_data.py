"""One-off script: remove all datapoints/metadata for a given source doc from the index."""
import sys
import json
from google.cloud import storage
from google.cloud import aiplatform

sys.path.insert(0, ".")
from src import config

def clear_document(gcs_uri: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.BUCKET_NAME)

    safe_name = gcs_uri.replace("gs://", "").replace("/", "_")
    meta_blob = bucket.blob(f"_metadata/{safe_name}.json")

    if not meta_blob.exists():
        print(f"No metadata found for {gcs_uri} — nothing to clear.")
        return

    entries = json.loads(meta_blob.download_as_text())
    ids = [e["id"] for e in entries]
    print(f"Found {len(ids)} datapoints for {gcs_uri}")

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
    index = aiplatform.MatchingEngineIndex(index_name=config.VECTOR_INDEX_RESOURCE_NAME)
    index.remove_datapoints(datapoint_ids=ids)
    print("Removed datapoints from Vector Search index.")

    meta_blob.delete()
    print("Deleted metadata file.")

    # also delete the original uploaded PDF from the bucket
    original_path = gcs_uri.replace(f"gs://{config.BUCKET_NAME}/", "")
    orig_blob = bucket.blob(original_path)
    if orig_blob.exists():
        orig_blob.delete()
        print(f"Deleted original file {original_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clear_old_data.py gs://bucket/path/to/file.pdf")
        sys.exit(1)
    clear_document(sys.argv[1])
