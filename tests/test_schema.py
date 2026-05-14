from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from italiabench.schema import AnyOf, Category, Difficulty, Question


def _valid_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fisco-iva-001",
        "category": "fisco",
        "difficulty": "easy",
        "question": "Qual è l'aliquota IVA ridotta applicata ai libri?",
        "ground_truth": "4%",
        "must_mention": ["4%"],
        "must_not_mention": ["10%"],
        "source": ["DPR 633/72"],
        "last_verified": "2026-04-01",
    }
    base.update(overrides)
    return base


def test_minimal_valid_question() -> None:
    q = Question.model_validate(_valid_payload())
    assert q.id == "fisco-iva-001"
    assert q.category is Category.FISCO
    assert q.difficulty is Difficulty.EASY
    assert q.last_verified == date(2026, 4, 1)
    assert q.canary is False


def test_any_of_constraint_parses() -> None:
    payload = _valid_payload(
        must_mention=[
            "4%",
            {"any_of": ["aliquota minima", "super-ridotta"]},
        ],
    )
    q = Question.model_validate(payload)
    assert isinstance(q.must_mention[1], AnyOf)
    assert q.must_mention[1].any_of == ["aliquota minima", "super-ridotta"]


def test_any_of_requires_at_least_two_alternatives() -> None:
    payload = _valid_payload(
        must_mention=[{"any_of": ["solo uno"]}],
    )
    with pytest.raises(ValidationError):
        Question.model_validate(payload)


def test_id_format_rejects_invalid() -> None:
    bad_ids = [
        "fisco-iva-1",  # number not zero-padded
        "Fisco-IVA-001",  # uppercase
        "fisco_iva_001",  # missing dashes between segments
        "fisco-iva",  # missing number
    ]
    for bad in bad_ids:
        with pytest.raises(ValidationError):
            Question.model_validate(_valid_payload(id=bad))


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_valid_payload(category="economia"))


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_valid_payload(extra_field="boom"))


def test_source_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_valid_payload(source=[]))


def test_unsupported_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_valid_payload(schema_version=999))
