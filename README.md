# Hybrid Travel Assistant

An AI-powered conversational travel assistant that combines vector search (Pinecone), knowledge graphs (Neo4j), and LLM reasoning (Gemini 2.5 Flash) to generate personalized travel itineraries with conversational follow-up support.

---

## Overview

This system implements a hybrid retrieval architecture that merges semantic search with graph-based relationship mapping. When a user requests travel recommendations, the system queries both a vector database for semantically similar entities and a graph database for structural relationships, then uses an LLM to synthesize coherent travel plans from the combined context.

The assistant maintains conversation context through a sliding window memory system (last 2 exchanges) and uses LLM-controlled tool calling to retrieve conversation history only when needed. Queries to Pinecone and Neo4j run in parallel using async execution, reducing retrieval latency by ~45%. An LRU cache stores the 128 most recent query embeddings, reducing API calls by ~30-40%.

---

## Key Features

**Hybrid Retrieval**: Combines semantic search (Pinecone) with graph relationships (Neo4j) to provide comprehensive travel context. Pinecone identifies semantically similar entities using cosine similarity on 768-dimensional embeddings, while Neo4j provides structured relationship information.

**LRU Embedding Cache**: Stores up to 128 recent query embeddings in memory with least-recently-used eviction policy. Eliminates redundant API calls for repeated queries during development and testing.

**Conversational Memory**: Maintains last 2 user-assistant exchanges (4 messages) using a sliding window. Automatically evicts oldest messages to prevent unbounded token growth while preserving recent context for follow-up queries.

**Adaptive Tool Calling**: LLM decides when to retrieve conversation history via `get_conversation_history()` tool. Memory is only included in prompts when queries reference previous context (e.g., "make it different", "add beaches"), saving ~14% tokens on standalone questions.

**Parallel Execution**: Async queries to Pinecone and Neo4j run concurrently using Python's asyncio and ThreadPoolExecutor, reducing total retrieval time to the maximum of individual query times rather than their sum.

**Citation Support**: All responses reference specific node IDs from the knowledge graph, enabling traceability of generated information.

---

## Installation

1. **Clone repository and setup environment:**

git clone https://github.com/Aniket982-ux/-Hybrid-AI-Travel-Assistant.git
cd -Hybrid-AI-Travel-Assistant
python -m venv venv
source venv/bin/activate # Linux/macOS
pip install -r requirements.txt


2. **Set API keys in `config.py`:**

NEO4J_URI = "your-uri"
NEO4J_USER = "your-user_name"
NEO4J_PASSWORD = "your_password"
GEMINI_API_KEY = "your_api-key"
HF_API_KEY = "your-api-key"
HF_MODEL_NAME = "your-model-name"
PINECONE_API_KEY = " your Pinecone API key " 
PINECONE_ENV = "your-env"
PINECONE_INDEX_NAME = "index-name"


3. **Load data:**

python load_to_neo4j.py
python visualize_graph.py # Optional: view graph
python pinecone_upload.py


4. **Run assistant:**

cd hybrid_tourist_planner
python hybrid_chat.py


5. **Example usage:**

You: Create a romantic 4-day itinerary for Vietnam
Assistant: [Generates itinerary with node IDs]

You: Make it more adventurous
Assistant: [Modifies using conversation context]

You: clear
Conversation cleared.


---

## Technical Stack

- **Embeddings**: BAAI/bge-base-en-v1.5 (768 dimensions, 512 max tokens)
- **Vector DB**: Pinecone Serverless (cosine similarity)
- **Graph DB**: Neo4j (relationship traversal)
- **LLM**: Gemini 2.5 Flash (function calling)
- **Async**: Python asyncio + ThreadPoolExecutor

---

## Evaluation Focus

This project demonstrates:
1. Hybrid retrieval combining vector and graph approaches
2. Production optimizations (caching, async execution, bounded memory)
3. Modern LLM features (tool calling, conversational context)
4. Clean modular architecture

See IMPROVEMENTS.md for detailed technical decisions and future enhancements.