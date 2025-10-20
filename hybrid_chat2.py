# hybrid_chat.py
import json
from typing import List
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import requests
import config
from google import genai
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)


# -----------------------------
# Config
# -----------------------------
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = 768  # match your embeddings (HF/Gemini)

# -----------------------------
# Initialize clients
# -----------------------------
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # free-tier safe
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# Hugging Face API for embeddings
HF_API_URL = f"https://api-inference.huggingface.co/models/{config.HF_MODEL_NAME}"
HF_HEADERS = {"Authorization": f"Bearer {config.HF_API_KEY}"}

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Get embedding from Hugging Face."""
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and isinstance(data[0], list):
        return data[0]
    return data

async def pinecone_query(query_text: str, top_k=TOP_K):
    loop = asyncio.get_running_loop()
    def _work():
        vec = embed_text(query_text)
        return index.query(vector=vec, top_k=top_k, include_metadata=True, include_values=False)
    res = await loop.run_in_executor(_executor, _work)
    return res["matches"]


def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    return facts

async def async_fetch_graph(node_ids: List[str]):
    loop = asyncio.get_running_loop()
    # Use original per-id graph fetch function
    return await loop.run_in_executor(_executor, lambda: fetch_graph_context(node_ids))

async def retrieve_parallel(query_text: str, top_k=TOP_K):
    matches = await pinecone_query(query_text, top_k=top_k)
    match_ids = [m["id"] for m in matches]
    graph_facts = await async_fetch_graph(match_ids)
    return matches, graph_facts


def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful travel assistant that plans trips for users by using semantic search results from a vector database and graph facts from a knowledge graph. Use these data sources to provide answers that are concise, accurate, and relevant to the user's query."
        "Use semantic search results to identify relevant places, attractions, or entities related to the query."
        "Use graph facts to understand relationships between entities and ensure itineraries are realistic and feasible within the user’s timeframe."
        "Provide answers in a friendly, professional tone, keeping responses under 300-450 words without unnecessary elaboration."
        "Where possible, include 2–3 concrete itinerary steps or tips per location with practical details. Suggest how much time to allocate or highlight top must-see attractions. If relevant, include travel or pacing advice to help users plan their days realistically."
        "If exact information is not available from the provided data, say you dont know. Do not invent or guess."
        "Cite node IDs when referencing any specific places, attractions, or entities."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", None)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
         "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
         "Based on the above, answer the user's question."}
    ]
    return prompt

def call_chat(prompt_messages):
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    user_prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in prompt_messages])

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt
    )

    return resp.text




# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit","quit"):
            break

        matches, graph_facts = asyncio.run(retrieve_parallel(query, top_k=TOP_K))
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)

        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

if __name__ == "__main__":
    interactive_chat()
