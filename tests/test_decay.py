import math
from datetime import datetime, timedelta, timezone

import pytest

from graphmem.decay import decay_score
from graphmem.schema import NodeType, RelType, SchemaViolation, check_relation

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_fresh_unaccessed_equals_weight():
    assert decay_score(2.0, 0, T0, T0, 0.05) == pytest.approx(2.0)


def test_access_boost_is_logarithmic():
    assert decay_score(1.0, 9, T0, T0, 0.05) == pytest.approx(1 + 0.1 * math.log(10))


def test_half_life():
    lam = 0.05
    half = math.log(2) / lam
    assert decay_score(1.0, 0, T0, T0 + timedelta(days=half), lam) == pytest.approx(0.5)


def test_future_last_accessed_is_clamped():
    assert decay_score(1.0, 0, T0 + timedelta(days=3), T0, 0.05) == pytest.approx(1.0)


def test_lambda_zero_disables_decay():
    assert decay_score(1.0, 0, T0, T0 + timedelta(days=1000), 0.0) == pytest.approx(1.0)


def test_relation_domain_range():
    check_relation(RelType.WORKS_AT, NodeType.PERSON, NodeType.ORGANIZATION)
    with pytest.raises(SchemaViolation):
        check_relation(RelType.WORKS_AT, NodeType.PERSON, NodeType.PERSON)
    with pytest.raises(SchemaViolation):
        check_relation(RelType.PARTICIPATED_IN, NodeType.PERSON, NodeType.CONCEPT)
    check_relation(RelType.RELATED_TO, NodeType.EVENT, NodeType.CONCEPT)  # unconstrained
