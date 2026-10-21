from datetime import datetime
from pathlib import Path

import pandas as pd
from database.uuid7 import uuid7_from_time
from database.vector_store import VectorStore

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "faq_dataset.csv", sep=";")

# Embed all rows with a few batched API calls instead of one call per row
contents = [f"Question: {q}\nAnswer: {a}" for q, a in zip(df["question"], df["answer"])]
embeddings = vec.get_embeddings(contents)


# Prepare data for insertion
def prepare_record(category: str, content: str, embedding: list) -> dict:
    """Prepare a record for insertion into the vector store.

    The id is a UUIDv7, which stores the current time (or a time you choose),
    so search(time_range=...) can filter by it.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time, pass a datetime to uuid7_from_time:

        Example:
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid7_from_time(specific_time))

        This is useful when your content already has an associated datetime.
        Leave the id as None to let Postgres generate one with uuidv7().
    """
    return {
        "id": str(uuid7_from_time(datetime.now())),
        "metadata": {
            "category": category,
            "created_at": datetime.now().isoformat(),
        },
        "contents": content,
        "embedding": embedding,
    }


records_df = pd.DataFrame(
    [prepare_record(c, t, e) for c, t, e in zip(df["category"], contents, embeddings)]
)

# Create tables, insert data, then build the indexes
vec.create_tables()
vec.upsert(records_df)
vec.create_index()  # StreamingDiskANN vector index + BM25 keyword index
