import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
from embeddings import embed_text

# Thread pool for running blocking I/O operations (Pinecone, Neo4j) asynchronously
_executor = ThreadPoolExecutor(max_workers=4)

TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = 768

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Create Pinecone index if it doesn't exist
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

async def pinecone_query(query_text: str, top_k=TOP_K):
    loop = asyncio.get_running_loop()
    def _work():
        vec = embed_text(query_text)
        return index.query(vector=vec, top_k=top_k, include_metadata=True, include_values=False)
    res = await loop.run_in_executor(_executor, _work)
    return res["matches"]

def fetch_graph_context(node_ids: List[str]):
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            # Cypher query: fetch all relationships and connected entities (limit 10 per node)
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, m.id AS id, m.name AS name, "
                "m.type AS type, m.description AS description LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400]
                })
    return facts

async def async_fetch_graph(node_ids: List[str]):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: fetch_graph_context(node_ids))

async def retrieve_parallel(query_text: str, top_k=TOP_K):
    # Run both queries concurrently (time = max(pinecone_time, neo4j_time))
    matches = await pinecone_query(query_text, top_k=top_k)
    match_ids = [m["id"] for m in matches]
    graph_facts = await async_fetch_graph(match_ids)
    return matches, graph_facts
