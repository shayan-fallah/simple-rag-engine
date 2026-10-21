# Building a High-Performance RAG Solution with Pgvectorscale and Python

This tutorial will guide you through setting up and using `pgvectorscale` with Docker and Python, leveraging OpenAI's `text-embedding-3-small` model for embeddings. You'll learn to build a RAG (Retrieval-Augmented Generation) solution, combining advanced retrieval techniques (including hybrid search) with intelligent answer generation based on the retrieved context. Perfect for AI engineers looking to enhance their projects with state-of-the-art vector search and generation capabilities with the power of PostgreSQL.

> **2026 update:** The project now runs on PostgreSQL 18 with pgvector 0.8 and pgvectorscale 0.9. It talks to Postgres directly with `psycopg` and plain SQL, because the `timescale-vector` Python client is no longer released. See [What changed in the 2026 update](#what-changed-in-the-2026-update).

## YouTube Tutorial
You can watch the full tutorial here on [YouTube](https://youtu.be/hAdEuDBN57g). The video shows the original 2024 version, so some code differs. The core concepts are the same.

## Pgvectorscale Documentation

For more information about using PostgreSQL as a vector database in AI applications, check out these resources (Timescale is now called Tiger Data):

- [GitHub Repository: pgvectorscale](https://github.com/timescale/pgvectorscale)
- [GitHub Repository: pgvector](https://github.com/pgvector/pgvector)
- [GitHub Repository: pg_textsearch (BM25)](https://github.com/timescale/pg_textsearch)
- [Blog Post: PostgreSQL and Pgvector: Now Faster Than Pinecone, 75% Cheaper, and 100% Open Source](https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/)
- [Blog Post: RAG Is More Than Just Vector Search](https://www.timescale.com/blog/rag-is-more-than-just-vector-search/)

## Why PostgreSQL?

Using PostgreSQL with pgvectorscale as your vector database offers several key advantages over dedicated vector databases:

- PostgreSQL is a robust, open-source database with a rich ecosystem of tools, drivers, and connectors. This ensures transparency, community support, and continuous improvements.

- By using PostgreSQL, you can manage both your relational and vector data within a single database. This reduces operational complexity, as there's no need to maintain and synchronize multiple databases.

- Pgvectorscale enhances pgvector with faster search capabilities, higher recall, and filtered search. It uses a DiskANN-inspired index (StreamingDiskANN) to significantly speed up Approximate Nearest Neighbor (ANN) searches.

- Keyword search (BM25) lives in the same database too, so hybrid search is a single SQL query.

Pgvectorscale builds on top of [pgvector](https://github.com/pgvector/pgvector), offering improved performance and additional features, making PostgreSQL a powerful and versatile choice for AI applications.

## Prerequisites

- Docker
- Python 3.10+
- OpenAI API key (for embeddings; answers can come from OpenAI or Anthropic)
- PostgreSQL GUI client (optional)

## Steps

1. Set up Docker environment
2. Connect to the database using a PostgreSQL GUI client (I use TablePlus)
3. Create a Python script to insert document chunks as vectors using OpenAI embeddings
4. Create a Python function to perform similarity search

## Detailed Instructions

### 1. Set up Docker environment

Create a `docker-compose.yml` file with the following content:

```yaml
services:
  timescaledb:
    image: timescale/timescaledb-ha:pg18
    container_name: timescaledb
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - pgdata_pg18:/home/postgres/pgdata
    restart: unless-stopped

volumes:
  pgdata_pg18:
```

The `timescaledb-ha:pg18` image includes pgvector, pgvectorscale and pg_textsearch.

Run the Docker container:

```bash
docker compose up -d
```

### 2. Connect to the database using a PostgreSQL GUI client

- Open client
- Create a new connection with the following details:
  - Host: localhost
  - Port: 5432
  - User: postgres
  - Password: password
  - Database: postgres

### 3. Create a Python script to insert document chunks as vectors

See `insert_vectors.py` for the implementation. This script uses OpenAI's `text-embedding-3-small` model to generate embeddings, in batches.

### 4. Create a Python function to perform similarity search

See `similarity_search.py` for the implementation. This script also uses OpenAI's `text-embedding-3-small` model for query embedding.

## Usage

1. Create a copy of `app/example.env` and rename it to `app/.env`
2. Open `.env` and fill in your OpenAI API key. Leave the database settings as is
3. Optional: set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` to have Claude write the answers
4. Run the Docker container
5. Install the required Python packages using `pip install -r requirements.txt`
6. From the `app` folder, run `python insert_vectors.py` to populate the database
7. Play with `python similarity_search.py` to perform similarity searches

## The database schema

`VectorStore.create_tables()` creates:

```sql
CREATE TABLE embeddings (
    id uuid PRIMARY KEY DEFAULT uuidv7(),   -- time-ordered id (Postgres 18)
    metadata jsonb NOT NULL DEFAULT '{}',
    contents text NOT NULL,
    embedding vector(1536) NOT NULL,
    labels smallint[] NOT NULL DEFAULT '{}' -- DiskANN filter labels
);
CREATE TABLE embeddings_labels (id smallint PRIMARY KEY, name text UNIQUE);
```

`VectorStore.create_index()` creates:

```sql
CREATE INDEX ON embeddings USING diskann (embedding vector_cosine_ops, labels);
CREATE INDEX ON embeddings USING bm25 (contents) WITH (text_config = 'english');
```

A basic search is plain SQL:

```sql
SELECT id, metadata, contents, embedding <=> $1 AS distance
FROM embeddings
ORDER BY embedding <=> $1
LIMIT 5;
```

## Using ANN search indexes to speed up queries

For large vector datasets (10k+ vectors), an index on the embedding column makes similarity queries much faster:

1. Supported indexes:
   - pgvectorscale's StreamingDiskANN (`USING diskann`, used here): a DiskANN-inspired graph index with compressed vectors (Statistical Binary Quantization)
   - pgvector's HNSW: Hierarchical Navigable Small World graph index
   - pgvector's IVFFlat: Inverted file index

2. The DiskANN index is "streaming": filtered queries keep searching until `LIMIT` is filled. If you use HNSW instead, turn on pgvector 0.8's iterative scans (`SET hnsw.iterative_scan = relaxed_order`) to get the same behaviour.

3. Tuning: `SET diskann.query_rescore = 50` and `SET diskann.query_search_list_size = 100` (the defaults) trade speed for recall. See the [pgvectorscale README](https://github.com/timescale/pgvectorscale).

## Filtering

`VectorStore.search()` supports four kinds of filters, which can be combined:

| Filter | Example | How it works |
|---|---|---|
| `labels` | `labels=["Shipping"]` | Runs inside the DiskANN index (filtered DiskANN). The fastest option. Labels come from the `category` metadata key. |
| `metadata_filter` | `{"category": "Shipping"}` | jsonb containment (`metadata @> ...`) |
| `predicates` | `Predicate("price", ">", 100) & Predicate("rating", ">=", 4.5)` | Comparisons on metadata keys, combined with `&` / `\|` |
| `time_range` | `(datetime(2026, 1, 1), datetime(2026, 2, 1))` | Range on the UUIDv7 `id`, which stores the insert time. Uses the primary key index. |

To store a document under a specific date, give it an id from `uuid7_from_time(your_datetime)`.

## Hybrid search

`VectorStore.hybrid_search()` runs a vector search and a BM25 keyword search (pg_textsearch), then merges them with Reciprocal Rank Fusion. Keyword search finds exact terms such as product names or order numbers that embeddings can miss. It takes the same filters as `search()`.

```python
results = vec.hybrid_search("Do you offer express shipping?", limit=3)
```

## What changed in the 2026 update

- **Postgres 16 → 18**: native `uuidv7()`, faster reads with asynchronous I/O.
- **pgvectorscale 0.4 → 0.9**: label-based filtered DiskANN, L2 and inner-product distance, up to 16,000 dimensions. DiskANN now requires a typed `vector(N)` column.
- **pgvector 0.7 → 0.8**: iterative index scans for filtered HNSW/IVFFlat queries, plus many fixes.
- **`timescale-vector` removed**: replaced by `psycopg` + `pgvector` and plain SQL (`app/database/vector_store.py`). `client.Predicates` is now `database.filters.Predicate`; `uuid_from_time` is now `uuid7_from_time`.
- **`instructor` removed**: OpenAI (Responses API) and Anthropic both support structured outputs natively.
- **Default models**: `gpt-5.5` and `claude-opus-5-5`, configurable with `OPENAI_MODEL` / `ANTHROPIC_MODEL`.
- **Docker volume fixed**: the old compose file mounted `/var/lib/postgresql/data`, but this image stores data in `/home/postgres/pgdata/data`, so data wasn't persisted. The new volume (`pgdata_pg18`) also keeps Postgres 18 separate from any old Postgres 16 data. Re-run `insert_vectors.py` after upgrading.

## Cosine Similarity in Vector Search

### What is Cosine Similarity?

Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It's a measure of orientation rather than magnitude.

- Range: -1 to 1 (for normalized vectors, which is typical in text embeddings)
- 1: Vectors point in the same direction (most similar)
- 0: Vectors are orthogonal (unrelated)
- -1: Vectors point in opposite directions (most dissimilar)

### Cosine Distance

In pgvector, the `<=>` operator computes cosine distance, which is 1 - cosine similarity.

- Range: 0 to 2
- 0: Identical vectors (most similar)
- 1: Orthogonal vectors
- 2: Opposite vectors (most dissimilar)

### Interpreting Results

When you get results from similarity_search:

- Lower distance values indicate higher similarity.
- A distance of 0 would mean exact match (rarely happens with embeddings).
- Distances closer to 0 indicate high similarity.
- Distances around 1 suggest little to no similarity.
- Distances approaching 2 indicate opposite meanings (rare in practice).
