"""Hybrid scoring engine for ItaliaBench answers.

Strategy:
1. Normalize the model answer (lowercase, strip diacritics, collapse whitespace).
2. Run deterministic checks: every ``must_mention`` item must match (with
   ``AnyOf`` requiring at least one alternative), and no ``must_not_mention``
   term may appear.
3. ``must_not_mention`` violations are a hard fail — the LLM judge cannot
   override them.
4. If deterministic checks pass: passed.
5. If only ``must_mention`` is partially missed AND a judge is supplied,
   call the judge with the question and answer; the judge may flip a fail
   to a pass.
6. Otherwise: failed, with a fractional score for partial-credit reporting.

Term matching uses word-boundary semantics: a term matches only if it is not
preceded or followed by an alphanumeric character. This avoids false
positives like ``"4%"`` matching inside ``"104%"`` or ``"iva"`` matching
inside ``"ivassicurazione"``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from italiabench.schema import AnyOf, Question


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace.

    Keeps all alphanumerics and symbols (``%``, ``€``, ``.``, etc.) since
    they often carry meaning (``"4%"`` vs ``"4"``).
    """
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(text.split())


def _term_matches(term: str, normalized_answer: str) -> bool:
    """Return True iff ``term`` appears in ``normalized_answer`` as an isolated
    token (not glued to surrounding alphanumerics)."""
    norm_term = _normalize(term)
    if not norm_term:
        return False

    start = 0
    while True:
        pos = normalized_answer.find(norm_term, start)
        if pos == -1:
            return False

        before_ok = pos == 0 or not normalized_answer[pos - 1].isalnum()
        end = pos + len(norm_term)
        after_ok = end >= len(normalized_answer) or not normalized_answer[end].isalnum()

        if before_ok and after_ok:
            return True
        start = pos + 1


@dataclass(frozen=True)
class JudgeVerdict:
    passed: bool
    reasoning: str = ""


Judge = Callable[[Question, str], JudgeVerdict]
"""A judge takes a question and the model's raw answer and returns a verdict."""


@dataclass
class MentionCheck:
    """Result of evaluating a single ``must_mention`` constraint."""

    constraint: str | AnyOf
    satisfied: bool
    matched_term: str | None = None


@dataclass
class ScoreResult:
    passed: bool
    score: float
    mentions: list[MentionCheck] = field(default_factory=list)
    forbidden_violations: list[str] = field(default_factory=list)
    judge_used: bool = False
    judge_verdict: JudgeVerdict | None = None

    @property
    def deterministic_passed(self) -> bool:
        """True iff the answer would pass without any judge intervention."""
        return all(m.satisfied for m in self.mentions) and not self.forbidden_violations


def score_answer(
    question: Question,
    answer: str,
    *,
    judge: Judge | None = None,
) -> ScoreResult:
    """Score ``answer`` against the constraints declared in ``question``."""
    normalized_answer = _normalize(answer)

    mentions: list[MentionCheck] = []
    for item in question.must_mention:
        if isinstance(item, AnyOf):
            matched: str | None = None
            for alternative in item.any_of:
                if _term_matches(alternative, normalized_answer):
                    matched = alternative
                    break
            mentions.append(
                MentionCheck(constraint=item, satisfied=matched is not None, matched_term=matched)
            )
        else:
            satisfied = _term_matches(item, normalized_answer)
            mentions.append(
                MentionCheck(
                    constraint=item,
                    satisfied=satisfied,
                    matched_term=item if satisfied else None,
                )
            )

    forbidden_violations = [
        term for term in question.must_not_mention if _term_matches(term, normalized_answer)
    ]

    if forbidden_violations:
        # Hard fail: judge cannot override an explicit forbidden mention.
        return ScoreResult(
            passed=False,
            score=0.0,
            mentions=mentions,
            forbidden_violations=forbidden_violations,
        )

    all_mentions_satisfied = all(m.satisfied for m in mentions)

    if all_mentions_satisfied:
        return ScoreResult(
            passed=True,
            score=1.0,
            mentions=mentions,
            forbidden_violations=[],
        )

    fraction_satisfied = (
        sum(1 for m in mentions if m.satisfied) / len(mentions) if mentions else 1.0
    )

    if judge is not None:
        verdict = judge(question, answer)
        return ScoreResult(
            passed=verdict.passed,
            score=1.0 if verdict.passed else fraction_satisfied,
            mentions=mentions,
            forbidden_violations=[],
            judge_used=True,
            judge_verdict=verdict,
        )

    return ScoreResult(
        passed=False,
        score=fraction_satisfied,
        mentions=mentions,
        forbidden_violations=[],
    )
