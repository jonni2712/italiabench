"""Pydantic schema for ItaliaBench question files.

Each question is a single YAML file under ``data/questions/<category>/``.
The schema is versioned: future incompatible changes bump
``Question.schema_version`` and require a dataset migration.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

SCHEMA_VERSION = 1


class Category(str, Enum):
    DIRITTO = "diritto"
    FISCO = "fisco"
    GEOGRAFIA_PA = "geografia_pa"
    STORIA_CULTURA = "storia_cultura"
    PROCEDURE = "procedure"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


_ID_PATTERN = re.compile(r"^[a-z0-9_]+-[a-z0-9_]+-\d{3}$")


class AnyOf(BaseModel):
    """An OR-constraint: at least one of the alternatives must appear."""

    model_config = ConfigDict(extra="forbid")

    any_of: list[Annotated[str, StringConstraints(min_length=1)]] = Field(min_length=2)


MentionItem = str | AnyOf


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    id: Annotated[str, StringConstraints(min_length=5, max_length=64)]
    category: Category
    difficulty: Difficulty
    question: Annotated[str, StringConstraints(min_length=10)]
    ground_truth: Annotated[str, StringConstraints(min_length=1)]
    must_mention: list[MentionItem] = Field(default_factory=list)
    must_not_mention: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        default_factory=list,
    )
    source: list[Annotated[str, StringConstraints(min_length=3)]] = Field(min_length=1)
    last_verified: date

    canary: bool = Field(
        default=False,
        description="Canary questions are excluded from the public dataset and used to detect contamination.",
    )
    tags: list[Annotated[str, StringConstraints(min_length=1)]] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"id must match '<area>-<topic>-<NNN>' (lowercase, snake_case, 3-digit number); got {v!r}",
            )
        return v

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {v} is not supported by this italiabench version (expected {SCHEMA_VERSION})",
            )
        return v
