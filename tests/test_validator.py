from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from italiabench.validator import DatasetError, validate_dataset


def _write_question(
    root: Path,
    folder: str,
    qid: str,
    *,
    declared_category: str | None = None,
    **overrides: object,
) -> Path:
    """Write a YAML question file. ``folder`` is where on disk; ``declared_category``
    is what the YAML payload says (defaults to ``folder``)."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "id": qid,
        "category": declared_category or folder,
        "difficulty": "easy",
        "question": "Qual è l'aliquota IVA ridotta applicata ai libri?",
        "ground_truth": "4%",
        "must_mention": ["4%"],
        "must_not_mention": ["10%"],
        "source": ["DPR 633/72"],
        "last_verified": "2026-04-01",
    }
    payload.update(overrides)
    target_dir = root / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{qid}.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def test_validates_a_clean_dataset(tmp_path: Path) -> None:
    _write_question(tmp_path, "fisco", "fisco-iva-001")
    _write_question(tmp_path, "diritto", "diritto-cc-001")
    questions = validate_dataset(tmp_path)
    assert len(questions) == 2


def test_empty_root_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no YAML files"):
        validate_dataset(tmp_path)


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        validate_dataset(tmp_path / "ghost")


def test_category_must_match_directory(tmp_path: Path) -> None:
    _write_question(tmp_path, "fisco", "diritto-cc-001", declared_category="diritto")
    with pytest.raises(DatasetError, match="move the file or fix the category"):
        validate_dataset(tmp_path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    _write_question(tmp_path, "fisco", "fisco-iva-001")
    _write_question(tmp_path, "diritto", "fisco-iva-001")
    with pytest.raises(DatasetError, match="duplicate id"):
        validate_dataset(tmp_path)


def test_invalid_yaml_is_reported_with_path(tmp_path: Path) -> None:
    bad = tmp_path / "fisco" / "fisco-iva-001.yaml"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(": this is not: valid: yaml:", encoding="utf-8")
    with pytest.raises(DatasetError) as excinfo:
        validate_dataset(tmp_path)
    assert "fisco-iva-001.yaml" in str(excinfo.value)
