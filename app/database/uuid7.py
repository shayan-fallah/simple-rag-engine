"""Time-ordered UUIDs (version 7).

A UUIDv7 starts with a millisecond Unix timestamp, so sorting by id sorts by
time and a time range becomes a simple `id` range on the primary key index.
Postgres 18 can generate these itself with `uuidv7()`; this helper lets you
create one for a specific datetime (e.g. when a document has its own date).
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

_VERSION_7 = 0x7 << 76
_VARIANT_RFC = 0b10 << 62


def _unix_ms(dt: datetime) -> int:
    # Naive datetimes are treated as local time, like datetime.timestamp().
    return int(dt.timestamp() * 1000)


def uuid7_from_time(dt: Optional[datetime] = None) -> uuid.UUID:
    """Create a random UUIDv7 for the given time (default: now)."""
    ms = _unix_ms(dt or datetime.now(timezone.utc))
    rand = int.from_bytes(os.urandom(10), "big")  # 80 random bits
    rand_a = rand >> 68  # 12 bits
    rand_b = rand & ((1 << 62) - 1)  # 62 bits
    return uuid.UUID(int=(ms << 80) | _VERSION_7 | (rand_a << 64) | _VARIANT_RFC | rand_b)


def uuid7_lower_bound(dt: datetime) -> uuid.UUID:
    """The smallest UUIDv7 for the given time. Use with >= / < for time ranges."""
    return uuid.UUID(int=(_unix_ms(dt) << 80) | _VERSION_7 | _VARIANT_RFC)
