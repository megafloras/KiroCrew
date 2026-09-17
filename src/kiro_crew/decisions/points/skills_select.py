"""``skills.select`` — which skills should this message load?

The existing selector is word-overlap trigger matching
(``SkillsLoader.get_triggered_skills`` scored by ``trigger_match.trigger_score``
against ``MIN_TRIGGER_OVERLAP``). This hook asks the oracle the same question
over the same candidate set and records whether the two agree.

SHADOW ONLY. The call site in ``context.py`` schedules this fire-and-forget and
never reads the result, so the injected skill list is byte-identical to main.
``build_message`` is synchronous; an arm that actually consumes an answer needs
an async budget there first, which is deliberately out of scope here.

Agreement is not something the transport can define generically, so it lives
here: :func:`agree` treats the oracle as agreeing when its pick is one of the
skills trigger matching already chose (or when both say "nothing applies").

One fact about the baseline the log cannot show on its own: ``skills.max_triggered``
defaults to 0, so on a stock install trigger matching selects NOTHING and
``chosen`` arrives empty. Agreement then measures how often the oracle also
answers ``none``; comparing two non-empty selections needs that cap raised first.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from kiro_crew.decisions.points import MAX_KEY_CHARS

logger = logging.getLogger(__name__)

POINT = "skills.select"

#: Cap on how many skills are described to the oracle. Trigger matching scores
#: every visible skill, but the state has to stay a small prompt, so the tail is
#: dropped rather than sent.
MAX_CANDIDATES = 100
MAX_MESSAGE_CHARS = 2000
MAX_DESCRIPTION_CHARS = 200

#: The "nothing applies" option. Kept as an explicit choice rather than an empty
#: answer so a refusal is distinguishable from a transport failure.
#:
#: Deliberately outside the skill-key namespace. Keys come from directory names
#: and the ``auto/`` prefix, so none can contain a space or a parenthesis; a bare
#: ``none`` collided with a skill literally called ``none``, which made the
#: oracle picking that skill indistinguishable from it refusing to pick one.
NONE_OPTION = "(no skill applies)"

#: Test seam. ``None`` means "use :func:`kiro_crew.decisions.decide`". Tests
#: monkeypatch this with an ``AsyncMock`` so this module can be exercised on a
#: tree where the core package does not exist yet.
_decide: Any = None

#: Test seam for :func:`kiro_crew.decisions.is_armed`, same contract as
#: :data:`_decide`.
_is_armed: Any = None


def candidates_from_loader(
    skills_loader: Any, project_dir: str | Path | None = None
) -> list[dict[str, str]]:
    """Describe every trigger-eligible skill as ``{key, description}``.

    Only skills that CAN be trigger matched are offered: ``always: true`` skills
    are injected unconditionally (never selected), and a skill with no
    ``triggers`` can never be picked by the baseline, so offering it would
    manufacture disagreements the baseline had no way to avoid.

    ``triggers`` is read through the loader's mtime-cached frontmatter reader,
    which trigger matching has already warmed for this same message, so this
    adds parsing work for no skill and one dict per row.
    """
    rows: list[dict[str, str]] = []
    try:
        listed = skills_loader.list_skills(project_dir)
    except Exception:
        logger.debug("skills.select: candidate listing failed", exc_info=True)
        return rows
    for row in listed:
        if not isinstance(row, dict) or row.get("always"):
            continue
        key = str(row.get("key") or "")
        if not key:
            continue
        if not _has_triggers(skills_loader, row):
            continue
        rows.append(
            {
                "key": key[:MAX_KEY_CHARS],
                "description": str(row.get("description") or "")[:MAX_DESCRIPTION_CHARS],
            }
        )
        if len(rows) >= MAX_CANDIDATES:
            break
    return rows


def _has_triggers(skills_loader: Any, row: dict) -> bool:
    """Whether *row* declares any trigger phrase.

    A read failure reads as "no triggers": a skill whose metadata cannot be
    parsed is also one trigger matching will not select.
    """
    path = row.get("path")
    if not path:
        return False
    try:
        meta = skills_loader._cached_frontmatter(Path(path), within=row.get("confine_root"))
    except Exception:
        return False
    return bool(str(meta.get("triggers", "")).strip())


def build_state(text: str, candidates: Sequence[dict[str, str]]) -> dict[str, Any]:
    """The state sent to the oracle: the message and the menu, nothing else."""
    return {
        "message": (text or "")[:MAX_MESSAGE_CHARS],
        "candidates": [
            {
                "key": str(c.get("key", ""))[:MAX_KEY_CHARS],
                "description": str(c.get("description", ""))[:MAX_DESCRIPTION_CHARS],
            }
            for c in list(candidates)[:MAX_CANDIDATES]
        ],
    }


def build_baseline(chosen: Sequence[str]) -> dict[str, Any]:
    """What the existing selector answered, in the answer's own vocabulary."""
    return {"pick": list(chosen)}


def row_agree(answers: Any, baseline: Any) -> bool | None:
    """Adapter handing :func:`agree` to the log, in the gate's calling shape.

    The generic rule the gate falls back to compares values over a shared
    question-id set, and the second half of that does not hold here: ``pick`` is
    one key against a baseline LIST, so the test is set membership rather than
    equality. Without this the point would log ``agree: null`` on every row.

    ``None`` when the row is not comparable at all -- no answers, or a baseline
    that is not the ``{"pick": [...]}`` this point sends.
    """
    if not answers or not isinstance(baseline, dict):
        return None
    answer = answers.get("pick")
    if answer is None:
        return None
    chosen = baseline.get("pick")
    if not isinstance(chosen, (list, tuple)):
        return None
    value = getattr(answer, "value", None)
    return agree(str(value) if value is not None else None, chosen)


def agree(pick: str | None, chosen: Sequence[str]) -> bool:
    """Whether the oracle's ``pick`` matches what trigger matching chose.

    Set membership rather than equality: the baseline returns a ranked LIST and
    the question asks for one skill, so picking any member of that list is the
    oracle reproducing the baseline's judgement. ``none`` agrees only when the
    baseline selected nothing at all.
    """
    chosen_list = [str(c) for c in chosen]
    if pick is None:
        return False
    if pick == NONE_OPTION:
        return not chosen_list
    return pick in chosen_list


async def shadow_skills_select(
    text: str,
    candidates: Sequence[dict[str, str]] | Callable[[], Sequence[dict[str, str]]],
    chosen: Sequence[str],
    *,
    session_key: str | None = None,
) -> Any:
    """Ask the oracle which skills this message needs; never act on the answer.

    *candidates* may be a list or a zero-argument callable returning one. The
    callable form is what the synchronous call site passes, so enumerating the
    skill tree happens on this task rather than on the turn's critical path.

    Returns whatever the core returns — ``None`` in every shadow, gated-off,
    or failed case. The caller discards it.
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
    # Ask BEFORE enumerating. ``candidates`` is a callable that walks the skill
    # tree and reads each skill's frontmatter, and building the state is the only
    # expensive thing this hook does -- so doing it above the gate meant the
    # DEFAULT configuration paid for it once per eligible message and then handed
    # the result to a ``decide`` that refuses on its first line. ``decide``
    # re-checks all five refusals, so this is a cost check and not a second gate.
    armed = _is_armed or getattr(_core, "is_armed", None)
    if armed is not None and not armed(POINT, session_key=session_key):
        return None
    try:
        rows = (
            await asyncio.to_thread(lambda: list(candidates()))
            if callable(candidates)
            else list(candidates)
        )
        state = build_state(text, rows)
        keys = [c["key"] for c in state["candidates"]]
        if any(choice not in keys for choice in chosen):
            return None
        # ONE question, deliberately. A second one would carry its own ``p``,
        # and the report buckets a row by the HIGHEST ``p`` across its answers --
        # so an answer that no baseline covers and no agreement rule reads would
        # still decide which calibration bucket this row lands in, which is the
        # curve that decides whether the point is ever promoted.
        questions = [
            Choice(
                "pick",
                "Which skills should be loaded for this message? "
                f"Answer {NONE_OPTION} if none apply.",
                options=keys + [NONE_OPTION],
            ),
        ]
        return await fn(
            POINT,
            state,
            questions,
            session_key=session_key,
            baseline=build_baseline(chosen),
            agree=row_agree,
        )
    except Exception:
        logger.debug("skills.select shadow failed", exc_info=True)
        return None
