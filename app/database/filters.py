"""Metadata filters that compile to SQL.

Replaces `timescale_vector.client.Predicates`:

    Predicate("price", ">", 100)
    Predicate("category", "==", "Books") & Predicate("rating", ">=", 4.5)
    Predicate("category", "==", "Shipping") | Predicate("category", "==", "Services")
"""

from datetime import date, datetime
from typing import Any, List, Tuple

from psycopg import sql

_OPERATORS = {"==": "=", "!=": "<>", ">": ">", ">=": ">=", "<": "<", "<=": "<="}


class Predicate:
    """A comparison on one metadata key. Combine with & (AND) and | (OR)."""

    def __init__(self, key: str, operator: str, value: Any):
        if operator not in _OPERATORS:
            raise ValueError(
                f"Unsupported operator {operator!r}. Use one of: {', '.join(_OPERATORS)}"
            )
        self.key = key
        self.operator = operator
        self.value = value

    def __and__(self, other: "Predicate") -> "Predicate":
        return _Group("AND", self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return _Group("OR", self, other)

    def to_sql(self) -> Tuple[sql.Composable, List[Any]]:
        """Return the SQL condition and its query parameters (%s placeholders)."""
        # Cast the JSON text to match the value's type, so 10 > 9 numerically.
        if isinstance(self.value, bool):
            cast = "::boolean"
        elif isinstance(self.value, (int, float)):
            cast = "::numeric"
        elif isinstance(self.value, (datetime, date)):
            cast = "::timestamptz"
        else:
            cast = ""
        condition = sql.SQL(f"(metadata->>%s){cast} {_OPERATORS[self.operator]} %s")
        return condition, [self.key, self.value]


class _Group(Predicate):
    def __init__(self, joiner: str, left: Predicate, right: Predicate):
        self.joiner = joiner
        self.left = left
        self.right = right

    def to_sql(self) -> Tuple[sql.Composable, List[Any]]:
        left_sql, left_params = self.left.to_sql()
        right_sql, right_params = self.right.to_sql()
        condition = sql.SQL("({} {} {})").format(
            left_sql, sql.SQL(self.joiner), right_sql
        )
        return condition, left_params + right_params
