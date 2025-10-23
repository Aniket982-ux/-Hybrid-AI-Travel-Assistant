# Hybrid Travel Assistant

An AI-powered conversational travel assistant that combines vector search (Pinecone), knowledge graphs (Neo4j), and LLM reasoning (Gemini 2.5 Flash) to generate personalized travel itineraries with conversational follow-up support.

## Overview

This system implements a hybrid retrieval architecture that merges semantic search with graph-based relationship mapping. When a user requests travel recommendations, the system queries both a vector database for semantically similar entities and a graph database for structural relationships, then uses an LLM to synthesize coherent travel plans from the combined context.

The assistant maintains conversation context through a sliding window memory system (last 2 exchanges) and uses LLM-controlled tool calling to retrieve conversation history only when needed. Queries to Pinecone and Neo4j run in parallel using async execution, reducing retrieval latency by ~45%. An LRU cache stores the 128 most recent query embeddings, reducing API calls by ~30-40%.

## Project Structure

BLUE_ENIGMA/
├── hybrid_tourist_planner/
│   ├── __pycache__/             # Python bytecode cache
│   ├── config.py                # API keys and configuration
│   ├── conversation.py          # Sliding window conversation memory
│   ├── embeddings.py            # Embedding generation (BAAI/bge-base-en-v1.5) with LRU  cache
│   ├── hybrid_chat.py           # Main interactive loop
│   ├── llm.py                   # Prompt building + Gemini tool calling
│   └── retrieval.py             # Pinecone + Neo4j parallel queries
│
├── lib/                         # External libraries
├── venv/                        # Virtual environment
├── .gitignore                   # Git ignore rules
├── load_to_neo4j.py             # Data loading script for Neo4j
├── neo4j_viz.html               # Graph visualization interface
├── pinecone_upload.py           # Data upload script for Pinecone
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── test.py                      # Unit tests
├── vietnam_travel_dataset.json  # Travel data (cities, attractions, themes)
└── visualize_graph.py           # Graph analysis utilities


## Key Features

**Hybrid Retrieval**: Combines semantic search (Pinecone) with graph relationships (Neo4j) to provide comprehensive travel context. Pinecone identifies semantically similar entities using cosine similarity on 768-dimensional embeddings, while Neo4j provides structured relationship information.

**LRU Embedding Cache**: Stores up to 128 recent query embeddings in memory with least-recently-used eviction policy. Eliminates redundant API calls for repeated queries during development and testing.

**Conversational Memory**: Maintains last 2 user-assistant exchanges (4 messages) using a sliding window. Automatically evicts oldest messages to prevent unbounded token growth while preserving recent context for follow-up queries.

**Adaptive Tool Calling**: LLM decides when to retrieve conversation history via `get_conversation_history()` tool. Memory is only included in prompts when queries reference previous context (e.g., "make it different", "add beaches"), saving ~14% tokens on standalone questions.

**Parallel Execution**: Async queries to Pinecone and Neo4j run concurrently using Python's asyncio and ThreadPoolExecutor, reducing total retrieval time to the maximum of individual query times rather than their sum.

**Citation Support**: All responses reference specific node IDs from the knowledge graph, enabling traceability of generated information.

## Installation

1.Set your API keys in `config.py`
2.create virtual environment & install all dependencies
3.Run ‘load_to_neo4j.py’
4.Run ‘visualize_graph.py’
5.Run `python pinecone_upload.py`
6.Run `python hybrid_chat.py`
7.Ask: `create a romantic 4 day itinerary for Vietnam

## Technical Stack

- **Embeddings**: BAAI/bge-base-en-v1.5 (768 dimensions, 512 max tokens)
- **Vector DB**: Pinecone Serverless (cosine similarity)
- **Graph DB**: Neo4j (relationship traversal)
- **LLM**: Gemini 2.5 Flash (function calling)
- **Async**: Python asyncio + ThreadPoolExecutor

## Evaluation Focus

This project demonstrates:
1. Hybrid retrieval combining vector and graph approaches
2. Production optimizations (caching, async execution, bounded memory)
3. Modern LLM features (tool calling, conversational context)
4. Clean modular architecture

See `improvements.md` for detailed technical decisions and future enhancements.
