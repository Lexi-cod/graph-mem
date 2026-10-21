"""Decay score, defined once in Python (reference) and once as a Cypher expression.

    decay = weight * (1 + 0.1 * ln(1 + access_count)) * exp(-lambda * dt_days)

Retrieval and pruning use min(decay, 1) so it is on the same [0, 1] scale as
cosine similarity before the alpha-mix.

dt_days is measured from last_accessed and clamped at 0 so that a clock running
"behind" a record (e.g. replaying a dataset with simulated time) never inflates it.
"""

import math
from datetime import datetime

ACCESS_BOOST = 0.1
SECONDS_PER_DAY = 86400.0


def decay_score(
    weight: float, access_count: int, last_accessed: datetime, now: datetime, lam: float
) -> float:
    dt_days = max(0.0, (now - last_accessed).total_seconds() / SECONDS_PER_DAY)
    return weight * (1 + ACCESS_BOOST * math.log1p(access_count)) * math.exp(-lam * dt_days)


def normalized_decay(*args, **kwargs) -> float:
    """Decay capped to [0, 1]; this is what enters the alpha-mix and prune threshold."""
    return min(1.0, decay_score(*args, **kwargs))


def cypher_cap(expr: str) -> str:
    """min(expr, 1) in Cypher. Pass a variable, not a long expression (it is repeated)."""
    return f"(CASE WHEN {expr} > 1.0 THEN 1.0 ELSE {expr} END)"


def cypher_cosine(score_expr: str) -> str:
    """Neo4j vector scores are (1 + cos) / 2; recover raw cosine, clamped at 0."""
    return f"(CASE WHEN {score_expr} < 0.5 THEN 0.0 ELSE 2 * {score_expr} - 1 END)"


def cypher_decay(var: str) -> str:
    """Cypher expression for the decay score of node/relationship `var`.

    Expects query parameters $now (datetime) and $lambda (float).
    """
    dt = f"duration.inSeconds({var}.last_accessed, $now).seconds / {SECONDS_PER_DAY}"
    return (
        f"({var}.weight * (1 + {ACCESS_BOOST} * log(1 + {var}.access_count))"
        f" * exp(-$lambda * CASE WHEN {dt} < 0 THEN 0.0 ELSE {dt} END))"
    )
