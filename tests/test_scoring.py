from __future__ import annotations

from typing import Any

import pytest

from italiabench.schema import Question
from italiabench.scoring import (
    JudgeVerdict,
    MentionCheck,
    _normalize,
    _term_matches,
    score_answer,
)


def _question(**overrides: Any) -> Question:
    base: dict[str, Any] = {
        "id": "fisco-iva-001",
        "category": "fisco",
        "difficulty": "easy",
        "question": "Qual è l'aliquota IVA ridotta applicata ai libri?",
        "ground_truth": "4%",
        "must_mention": ["4%"],
        "must_not_mention": ["10%", "22%"],
        "source": ["DPR 633/72"],
        "last_verified": "2026-04-01",
    }
    base.update(overrides)
    return Question.model_validate(base)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_strips_accents() -> None:
    assert _normalize("Perché È così?") == "perche e cosi?"


def test_normalize_collapses_whitespace() -> None:
    assert _normalize("a   b\tc\n d") == "a b c d"


def test_normalize_preserves_percent_and_currency() -> None:
    assert _normalize("Il 4% e 10€") == "il 4% e 10€"


# ---------------------------------------------------------------------------
# Term matching with word boundaries
# ---------------------------------------------------------------------------


def test_term_matches_basic() -> None:
    assert _term_matches("4%", "l'aliquota e il 4% per i libri")


def test_term_matches_normalizes_term_against_already_normalized_answer() -> None:
    # _term_matches normalizes the term internally; the answer is expected to
    # already be normalized (this is what score_answer does for us).
    assert _term_matches("perché", _normalize("Perche tu lo dici"))
    assert _term_matches("ALIQUOTA", _normalize("L'aliquota è 4%"))


def test_score_answer_handles_uppercase_and_accents_in_real_answer() -> None:
    # End-to-end: caller passes raw answer, normalization is internal.
    q = _question(must_mention=["perché", "aliquota"])
    result = score_answer(q, "PERCHÉ l'ALIQUOTA è ridotta sui libri.")
    assert result.passed is True


def test_term_does_not_match_when_substring_of_larger_number() -> None:
    # The infamous case: "4%" must NOT match inside "104%".
    assert not _term_matches("4%", "l'aliquota corretta e 104%")


def test_term_does_not_match_when_substring_of_larger_word() -> None:
    assert not _term_matches("iva", "l'ivassicurazione e cara")
    assert not _term_matches("aliquota", "l'aliquotario e una persona")


def test_term_matches_at_start_and_end_of_string() -> None:
    assert _term_matches("aliquota", "aliquota minima 4%")
    assert _term_matches("ridotta", "iva ridotta")


def test_empty_term_never_matches() -> None:
    assert not _term_matches("", "qualunque cosa")


# ---------------------------------------------------------------------------
# score_answer — deterministic happy path
# ---------------------------------------------------------------------------


def test_perfect_answer_passes() -> None:
    q = _question(must_mention=["4%", "iva"])
    result = score_answer(q, "L'aliquota IVA per i libri è del 4%.")
    assert result.passed is True
    assert result.score == 1.0
    assert all(m.satisfied for m in result.mentions)
    assert result.forbidden_violations == []
    assert result.judge_used is False


def test_missing_required_mention_fails() -> None:
    q = _question(must_mention=["4%", "iva"])
    result = score_answer(q, "L'aliquota è del 4%.")  # missing "iva"
    assert result.passed is False
    assert result.score == 0.5  # 1 of 2 satisfied
    assert any(not m.satisfied for m in result.mentions)


def test_must_not_mention_is_hard_fail_even_with_all_required() -> None:
    q = _question(must_mention=["4%"], must_not_mention=["10%"])
    result = score_answer(q, "L'IVA è del 4% (non del 10%).")
    assert result.passed is False
    assert result.score == 0.0
    assert "10%" in result.forbidden_violations


def test_no_must_mention_means_only_forbidden_check_matters() -> None:
    q = _question(must_mention=[], must_not_mention=["10%"])
    pass_result = score_answer(q, "L'IVA è del 4%.")
    assert pass_result.passed is True
    assert pass_result.score == 1.0

    fail_result = score_answer(q, "L'IVA è del 10%.")
    assert fail_result.passed is False


# ---------------------------------------------------------------------------
# score_answer — AnyOf semantics
# ---------------------------------------------------------------------------


def test_any_of_passes_when_at_least_one_alternative_matches() -> None:
    q = _question(
        must_mention=[
            "4%",
            {"any_of": ["aliquota minima", "super-ridotta"]},
        ],
    )
    result = score_answer(q, "Il 4% è la cosiddetta aliquota minima.")
    assert result.passed is True
    any_of_check = result.mentions[1]
    assert any_of_check.satisfied is True
    assert any_of_check.matched_term == "aliquota minima"


def test_any_of_fails_when_no_alternative_matches() -> None:
    q = _question(
        must_mention=[
            "4%",
            {"any_of": ["aliquota minima", "super-ridotta"]},
        ],
    )
    result = score_answer(q, "Il 4% si applica ai libri.")
    assert result.passed is False
    assert result.mentions[1].satisfied is False


# ---------------------------------------------------------------------------
# score_answer — judge interaction
# ---------------------------------------------------------------------------


def test_judge_can_flip_partial_fail_to_pass() -> None:
    q = _question(must_mention=["4%", "aliquota minima"])
    # Answer satisfies "4%" but uses "ridottissima" instead of "aliquota minima".
    answer = "Il 4% è l'aliquota ridottissima sui libri."

    def judge(question: Question, ans: str) -> JudgeVerdict:
        return JudgeVerdict(passed=True, reasoning="Sinonimo accettabile.")

    result = score_answer(q, answer, judge=judge)
    assert result.passed is True
    assert result.score == 1.0
    assert result.judge_used is True
    assert result.judge_verdict is not None
    assert result.judge_verdict.reasoning == "Sinonimo accettabile."
    assert result.deterministic_passed is False  # but with judge override, passed


def test_judge_can_keep_fail_as_fail() -> None:
    q = _question(must_mention=["4%", "aliquota minima"])
    answer = "Non lo so."

    def judge(question: Question, ans: str) -> JudgeVerdict:
        return JudgeVerdict(passed=False, reasoning="Risposta vuota.")

    result = score_answer(q, answer, judge=judge)
    assert result.passed is False
    assert result.score == 0.0  # 0 of 2 mentions satisfied
    assert result.judge_used is True


def test_judge_is_not_called_when_deterministic_passes() -> None:
    q = _question(must_mention=["4%"])
    calls: list[tuple[Question, str]] = []

    def judge(question: Question, ans: str) -> JudgeVerdict:
        calls.append((question, ans))
        return JudgeVerdict(passed=False)

    result = score_answer(q, "Il 4% è giusto.", judge=judge)
    assert result.passed is True
    assert result.judge_used is False
    assert calls == []


def test_judge_is_not_called_on_forbidden_violation() -> None:
    q = _question(must_mention=["4%"], must_not_mention=["10%"])
    calls: list[tuple[Question, str]] = []

    def judge(question: Question, ans: str) -> JudgeVerdict:
        calls.append((question, ans))
        return JudgeVerdict(passed=True, reasoning="should never see this")

    result = score_answer(q, "Il 4% per i libri (e il 10% per altri).", judge=judge)
    assert result.passed is False
    assert result.judge_used is False
    assert calls == []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_mention_check_records_matched_term_for_plain_string() -> None:
    q = _question(must_mention=["4%"])
    result = score_answer(q, "Il 4% va bene.")
    assert result.mentions == [MentionCheck(constraint="4%", satisfied=True, matched_term="4%")]


@pytest.mark.parametrize(
    "answer,expected_violations",
    [
        ("L'IVA è del 22%.", ["22%"]),
        ("Il 10% sui generi alimentari, non il 22%.", ["10%", "22%"]),
        ("Solo il 4%.", []),
    ],
)
def test_forbidden_violations_listed_in_order(answer: str, expected_violations: list[str]) -> None:
    q = _question(must_mention=[], must_not_mention=["10%", "22%"])
    result = score_answer(q, answer)
    assert result.forbidden_violations == expected_violations
