"""``skills.dedupe`` — is this auto-skill candidate new, a duplicate, or an update?

The existing judge is one LLM turn on the shared background session
(``HistoryConsolidator._dedupe_judge``, 60s, fail-open to "new"), parsed by
``skills_dedupe.parse_dedupe_verdict``. This hook puts the same candidate and
the same existing-skill set to the oracle as one closed choice and records
whether the two verdicts match.

SHADOW ONLY. The call site schedules this alongside the judge's own verdict and
never reads the result, so what gets staged, dropped or merged is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from kiro_crew.decisions.points import MAX_KEY_CHARS

logger = logging.getLogger(__name__)

POINT = "skills.dedupe"

MAX_DESCRIPTION_CHARS = 400
MAX_TRIGGERS_CHARS = 400
#: Cap on how many existing skills are described. The set is already bounded
#: (live auto-skills plus staged candidates), so this only guards a pathological
#: workspace.
MAX_EXISTING = 100

#: Verdict vocabulary. ``NONE`` is "no match — stage as new"; the other two carry
#: the matched key, so the option list is built per call.
VERDICT_NONE = "NONE"
VERDICT_DUP_PREFIX = "DUP:"
VERDICT_UPDATE_PREFIX = "UPDATE:"

#: Test seam — see :mod:`kiro_crew.decisions.points.skills_select`.
_decide: Any = None


def build_state(candidate: dict[str, Any], existing: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The state: the candidate's metadata and the metadata it competes with."""
    return {
        "candidate": {
            "key": str(candidate.get("key", ""))[:MAX_KEY_CHARS],
            "description": str(candidate.get("description", ""))[:MAX_DESCRIPTION_CHARS],
            "triggers": str(candidate.get("triggers", ""))[:MAX_TRIGGERS_CHARS],
        },
        "existing": [
            {
                "key": str(row.get("key", ""))[:MAX_KEY_CHARS],
                "description": str(row.get("description", ""))[:MAX_DESCRIPTION_CHARS],
            }
            for row in list(existing)[:MAX_EXISTING]
            if isinstance(row, dict)
        ],
    }


def verdict_options(existing: Sequence[dict[str, Any]]) -> list[str]:
    """The closed answer set: no match, or one key under one of two relations."""
    keys = [
        str(row.get("key", ""))[:MAX_KEY_CHARS]
        for row in list(existing)[:MAX_EXISTING]
        if isinstance(row, dict)
    ]
    keys = [k for k in keys if k]
    return (
        [VERDICT_NONE]
        + [f"{VERDICT_DUP_PREFIX}{k}" for k in keys]
        + [f"{VERDICT_UPDATE_PREFIX}{k}" for k in keys]
    )


def baseline_verdict(verdict: str, key: str | None) -> str:
    """Spell the existing judge's ``(verdict, key)`` in the answer's vocabulary.

    ``kiro_crew.skills_dedupe`` names the three outcomes ``new`` / ``dup`` /
    ``update``; the option list needs the key attached, so the two spellings are
    reconciled here rather than at the call site.
    """
    normalized = (verdict or "").strip().lower()
    if normalized == "dup" and key:
        return f"{VERDICT_DUP_PREFIX}{key}"
    if normalized == "update" and key:
        return f"{VERDICT_UPDATE_PREFIX}{key}"
    return VERDICT_NONE


def build_baseline(verdict: str, key: str | None) -> dict[str, Any]:
    return {"verdict": baseline_verdict(verdict, key)}


def row_agree(answers: Any, baseline: Any) -> bool | None:
    """Adapter handing :func:`agree` to the log, in the gate's calling shape.

    Value equality over a shared id set -- the generic rule -- already gives the
    right answer here, since this point asks one question whose vocabulary the
    baseline is normalised into. The adapter is passed anyway so all three
    points read the same at their call to ``decide`` and a later second question
    cannot silently turn this row's ``agree`` into ``null``.
    """
    if not answers or not isinstance(baseline, dict):
        return None
    answer = answers.get("verdict")
    if answer is None:
        return None
    expected = baseline.get("verdict")
    if not isinstance(expected, str):
        return None
    value = getattr(answer, "value", None)
    return agree(str(value) if value is not None else None, expected)


def agree(oracle_verdict: str | None, baseline: str | None) -> bool:
    """Whether both verdicts name the same relation to the same key.

    Exact string equality, unlike ``skills.select``: the answer set is closed and
    every member is a distinct action, so "duplicate of A" and "update of A" are
    a disagreement even though both found A.
    """
    if oracle_verdict is None or baseline is None:
        return False
    return oracle_verdict == baseline


async def shadow_skills_dedupe(
    candidate: dict[str, Any],
    existing: Sequence[dict[str, Any]],
    verdict: str,
    key: str | None,
    *,
    session_key: str | None = None,
) -> Any:
    """Ask the oracle for a dedupe verdict; never act on the answer.

    *verdict* / *key* are what the LLM judge just returned — they travel as the
    baseline so agreement is logged, and are otherwise untouched.
    """
    try:
        from kiro_crew import decisions as _core
        from kiro_crew.decisions.types import Choice
    except ImportError:
        return None
    # ``decide`` is read off the package rather than imported by name: on a tree
    # where the core has not landed, ``kiro_crew.decisions`` still resolves as a
    # namespace package, so the by-name import is a type error rather than the
    # ImportError above. A core without the symbol degrades like an absent one.
    fn = _decide or getattr(_core, "decide", None)
    if fn is None:
        return None
    try:
        state = build_state(candidate, existing)
        options = verdict_options(existing)
        baseline = build_baseline(verdict, key)
        if baseline["verdict"] not in options:
            return None
        questions = [
            Choice(
                "verdict",
                "Is this candidate skill new, a duplicate of an existing one, or "
                "an update to one? Answer NONE when it is new.",
                options=options,
            )
        ]
        return await fn(
            POINT,
            state,
            questions,
            session_key=session_key,
            baseline=baseline,
            agree=row_agree,
        )
    except Exception:
        logger.debug("skills.dedupe shadow failed", exc_info=True)
        return None
