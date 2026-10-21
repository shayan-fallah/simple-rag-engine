from datetime import datetime, timedelta

from database.filters import Predicate
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer

# Initialize VectorStore
vec = VectorStore()


def print_response(response):
    print(f"\n{response.answer}")
    print("\nThought process:")
    for thought in response.thought_process:
        print(f"- {thought}")
    print(f"\nContext: {response.enough_context}")


# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = "What are your shipping options?"
results = vec.search(relevant_question, limit=3)

response = Synthesizer.generate_response(question=relevant_question, context=results)
print_response(response)

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

irrelevant_question = "What is the weather in Tokyo?"

results = vec.search(irrelevant_question, limit=3)

response = Synthesizer.generate_response(question=irrelevant_question, context=results)
print_response(response)

# --------------------------------------------------------------
# Label filtering (runs inside the DiskANN index — fastest filter)
# --------------------------------------------------------------

results = vec.search(relevant_question, limit=3, labels=["Shipping"])

response = Synthesizer.generate_response(question=relevant_question, context=results)
print_response(response)

# --------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------

metadata_filter = {"category": "Shipping"}

results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)

# --------------------------------------------------------------
# Advanced filtering using Predicates
# --------------------------------------------------------------

predicates = Predicate("category", "==", "Shipping")
results = vec.search(relevant_question, limit=3, predicates=predicates)


predicates = Predicate("category", "==", "Shipping") | Predicate(
    "category", "==", "Services"
)
results = vec.search(relevant_question, limit=3, predicates=predicates)


predicates = Predicate("category", "==", "Shipping") & Predicate(
    "created_at", ">", "2024-09-01"
)
results = vec.search(relevant_question, limit=3, predicates=predicates)

# --------------------------------------------------------------
# Time-based filtering (uses the time stored in the UUIDv7 id)
# --------------------------------------------------------------

# Last 7 days — returns results if you ran insert_vectors.py this week
time_range = (datetime.now() - timedelta(days=7), datetime.now() + timedelta(minutes=1))
results = vec.search(relevant_question, limit=3, time_range=time_range)

# A range before the data was inserted — no results
time_range = (datetime(2024, 8, 1), datetime(2024, 8, 30))
results = vec.search(relevant_question, limit=3, time_range=time_range)

# --------------------------------------------------------------
# Hybrid search (vector + BM25 keyword search)
# --------------------------------------------------------------

keyword_question = "Do you offer express shipping?"
results = vec.hybrid_search(keyword_question, limit=3)
print(results[["content", "score", "semantic_rank", "keyword_rank"]])

response = Synthesizer.generate_response(question=keyword_question, context=results)
print_response(response)
