"""One-off script: delete all documents in the Firestore answer_cache collection."""
import sys
sys.path.insert(0, ".")
from google.cloud import firestore
from src import config

client = firestore.Client(project=config.PROJECT_ID)
docs = client.collection("answer_cache").stream()
count = 0
for doc in docs:
    doc.reference.delete()
    count += 1
print(f"Deleted {count} cached answers.")
