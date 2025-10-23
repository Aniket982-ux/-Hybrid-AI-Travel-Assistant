import json
import time
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
import requests
from hybrid_tourist_planner import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = 768  # ✅ embedding dimension for HF model

# -----------------------------
# Initialize clients
# -----------------------------
pc = Pinecone(api_key=config.PINECONE_API_KEY)
HF_API_URL = f"https://api-inference.huggingface.co/models/{config.HF_MODEL_NAME}"
HF_HEADERS = {"Authorization": f"Bearer {config.HF_API_KEY}"}

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
# ✅ Ensure index is created with correct VECTOR_DIM
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME} with dimension {VECTOR_DIM}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts):
    """Generate embeddings using Hugging Face API."""
    embeddings = []
    for text in texts:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and isinstance(data[0], list):
            embeddings.append(data[0])
        else:
            embeddings.append(data)
    return embeddings

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)  # ✅ uses Hugging Face

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.2)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()
