# Design Decisions & Technical Rationale

This document explains key technical decisions made in the Hybrid Travel Assistant and potential future improvements.

---

## 1. Hybrid Retrieval Architecture

**Decision:** Combine Vector Search (Pinecone) + Knowledge Graph (Neo4j)

**Why:**
Vector search handles semantic similarity ("romantic vacation" matches "honeymoon") but lacks understanding of structured relationships. Knowledge graphs provide relationship context (which attractions are in which city, travel connections) but require exact matches. Combining both gives semantic understanding with structural constraints, enabling realistic itineraries that respect geographic and temporal feasibility.

---

## 2. LRU Embedding Cache

**Decision:** In-Memory LRU Cache (128 entries)

**Why:**
Embedding API calls add noticeable latency. During development and testing, queries are repeated frequently. LRU cache stores recent embeddings to avoid redundant API calls while maintaining bounded memory (O(1) lookup, automatic eviction when full).

**Why Not Redis:**
Zero external dependencies simplifies evaluation setup. In-memory is sufficient for single-user demo. Production deployments at scale would benefit from Redis for persistence and multi-worker sharing.

---

## 3. Sliding Window Conversation Memory

**Decision:** Fixed window of 2 exchanges (4 messages)

**Why:**
Conversation memory enables follow-up queries ("make it less romantic", "add beaches"). Unbounded history causes token growth over conversation length. Sliding window maintains recent context while preventing cost escalation.

**Bounded vs Unbounded:**
- Sliding window: Constant overhead regardless of conversation length
- Unbounded: Grows linearly with turns, unsustainable for long sessions

---

## 4. LLM-Controlled Tool Calling

**Decision:** Let Gemini decide when to retrieve conversation history

**Why:**
Not all queries need conversation context. Standalone questions ("What's Hanoi like?") don't reference previous exchanges. Including memory unnecessarily wastes tokens.

**Why Tool Calling Over Keywords:**
Keyword-based detection ("it", "that", "change") produces false positives ("What is it like in Hanoi?") and false negatives ("Can you suggest something different?"). LLM understands context naturally without manual rule maintenance.

**How It Works:**
LLM receives `get_conversation_history()` as callable tool. If query references previous context, LLM calls tool and regenerates response with context. Otherwise responds directly.

---

## 5. Parallel Async Retrieval

**Decision:** Query Pinecone and Neo4j concurrently using AsyncIO

**Why:**
Pinecone and Neo4j queries are independent I/O-bound operations. Running them sequentially adds unnecessary latency. Parallel execution reduces total time to maximum of individual queries rather than their sum.

**Time Complexity:**
- Sequential: O(t_pinecone + t_neo4j)
- Parallel: O(max(t_pinecone, t_neo4j))
- Practical speedup: ~2x faster

---

## 6. Embedding Model Selection

**Decision:** BAAI/bge-base-en-v1.5

**Why:**
Strong performance on semantic similarity tasks with 768-dimensional embeddings. Free Hugging Face API access with reasonable rate limits. BERT-based architecture is well-documented and widely adopted. Appropriate balance between embedding quality and dimensionality for demo-scale projects.

---

## 7. LLM Selection

**Decision:** Gemini 2.5 Flash

**Why:**
Native function calling support essential for tool-based conversation memory. Fast inference suitable for interactive applications. Cost-effective for demo and evaluation purposes. Built-in lightweight reasoning helps with tool decision-making.

---

## Future Enhancements

### 1. Semantic Embedding Cache

**Current Limitation:**
Cache uses exact string matching. "romantic Vietnam" and "vietnam romantic honeymoon" are treated as different queries despite semantic similarity.

**Proposed Solution:**
Store query embeddings alongside text. On new query, check cosine similarity with cached embeddings (threshold: ~0.95). If similar query exists, reuse cached results.

**Challenge - Chicken and Egg Problem:**
To check if a query is cached based on semantic similarity, you first need its embedding. But getting the embedding requires an API call, which is what caching is trying to avoid. This creates a circular dependency.

**Two-Tier Solution:**
- **L1 Cache:** Exact string match (O(1), instant, no API call needed)
- **L2 Cache:** Semantic similarity (O(n) but only checked on L1 miss, after embedding is already computed)

Repeated exact queries hit L1 (no embedding needed). Semantically similar queries hit L2 (embedding computed once, then reused for similarity check).

---

### 2. Multi-Tool System

**Expand Beyond Conversation Memory:**
- `get_weather(city, date)` - Real-time weather integration
- `check_flight_price(from, to, date)` - Flight pricing lookup
- `find_hotels(city, budget)` - Hotel availability search
- `get_user_preferences()` - Stored travel preferences

Transforms static recommender into dynamic travel agent with real-time information.

---

### 3. Graph RAG for Complex Queries

**Enhancement:** Multi-hop graph traversal for complex relationships. Instead of querying immediate neighbors only, traverse multiple relationship types simultaneously to answer queries like "romantic beach town accessible from Hanoi with vegetarian restaurants."

---

### 4. Scaling to Production (Millions of Nodes)

The current architecture works well for demo-scale (thousands of entities) but requires modifications for production deployment with millions of nodes.

#### Current Limitations

**Single Database Instances:**
- Pinecone single index: Performance optimal up to ~100K vectors
- Neo4j single instance: Memory constraints around 1M nodes
- In-memory cache: Doesn't persist or share across workers

#### Scaling Strategy

**1. Data Sharding**

Partition vector data across multiple Pinecone indexes based on natural boundaries (region, category, theme). Route queries to relevant shards or query multiple shards in parallel and merge results. Each shard maintains optimal performance by staying within the 100K-500K vector range.

**2. Neo4j Clustering**

Deploy Neo4j in cluster mode with one primary node for writes and multiple read replicas for query distribution. Load balance user queries across replicas. For multi-country deployments, shard graph data by geographic region to minimize cross-shard queries.

**3. Distributed Caching**

Replace in-memory LRU with Redis cluster for:
- Persistent cache across application restarts
- Shared cache across multiple worker instances
- Horizontal scaling by adding Redis nodes
- TTL-based eviction for automatic freshness

**4. Query Optimization**

Implement intelligent routing based on query analysis:
- Simple queries target single relevant shard
- Complex queries distributed across multiple shards
- Filter and rank results globally before returning

**5. Asynchronous Write Operations**

Decouple data updates from query serving using message queues. User queries hit read-optimized replicas while updates are processed asynchronously through queues to primary databases. Eventual consistency is acceptable for travel data updates.

#### Scaling Benefits

Horizontal scaling through sharding and clustering enables:
- Support for 1M+ entities across distributed indexes
- Consistent query latency regardless of total data size
- Linear throughput scaling by adding nodes
- High availability through replication
- Geographic distribution for global deployments

The modular architecture allows incremental scaling as data and traffic grow, starting with single instances and progressively adding shards and replicas based on actual usage patterns.

---

## Conclusion

This project demonstrates hybrid retrieval with production-ready optimizations (caching, async execution, bounded memory) suitable for demo and evaluation. The architecture has a clear scaling path to millions of nodes through data sharding, distributed caching, and database clustering while maintaining consistent performance characteristics.

**Key Achievements:**
- Hybrid retrieval combining semantic search and graph relationships
- Performance optimizations (parallel queries, LRU caching)
- Modern LLM features (tool calling, conversational context)
- Clear production scaling strategy with distributed architecture
