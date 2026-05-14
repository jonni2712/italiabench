"""Validate the ItaliaBench question dataset.

Run as a module:
    python -m italiabench.validator data/questions

Exit code is 0 only if every YAML file is valid AND ids are globally unique
AND the on-disk path matches the declared category.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml
from pydantic import ValidationError

from italiabench.schema import Question


class DatasetError(Exception):
    """Raised when one or more dataset files fail validation."""


def _iter_yaml_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.yaml") if p.is_file())


def _load_question(path: Path) -> Question:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise DatasetError(f"{path}: top-level YAML node must be a mapping, got {type(raw).__name__}")
    return Question.model_validate(raw)


def validate_dataset(root: Path) -> list[Question]:
    if not root.exists():
        raise DatasetError(f"dataset root does not exist: {root}")

    files = _iter_yaml_files(root)
    if not files:
        raise DatasetError(f"no YAML files found under {root}")

    questions: list[Question] = []
    errors: list[str] = []

    for path in files:
        try:
            q = _load_question(path)
        except (yaml.YAMLError, DatasetError) as e:
            errors.append(f"{path}: {e}")
            continue
        except ValidationError as e:
            errors.append(f"{path}:\n{e}")
            continue

        # Path-on-disk must match declared category.
        try:
            on_disk_category = path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            on_disk_category = "<root>"
        if on_disk_category != q.category.value:
            errors.append(
                f"{path}: file is in '{on_disk_category}/' but category={q.category.value!r}; "
                f"move the file or fix the category.",
            )
            continue

        questions.append(q)

    # Globally unique ids.
    id_counts = Counter(q.id for q in questions)
    duplicates = {qid: count for qid, count in id_counts.items() if count > 1}
    if duplicates:
        for qid, count in duplicates.items():
            errors.append(f"duplicate id {qid!r} appears {count} times")

    if errors:
        raise DatasetError(
            f"{len(errors)} validation error(s) in {len(files)} file(s):\n\n"
            + "\n\n".join(errors),
        )

    return questions


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m italiabench.validator <dataset-root>", file=sys.stderr)
        return 2

    root = Path(args[0]).resolve()
    try:
        questions = validate_dataset(root)
    except DatasetError as e:
        print(str(e), file=sys.stderr)
        return 1

    public = sum(1 for q in questions if not q.canary)
    canary = sum(1 for q in questions if q.canary)
    print(f"OK — {len(questions)} questions ({public} public, {canary} canary) in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
