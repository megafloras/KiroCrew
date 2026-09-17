"""``cron.novelty`` — does this cron result carry news worth delivering?

The existing test is byte equality: ``slack/gateway.py`` hashes the result and
suppresses delivery when the hash repeats. Two results that differ by a
timestamp are therefore both delivered. This hook asks the oracle whether the
new result actually says anything the previous one did not, and records the
probability beside the delivery that happened anyway.

SHADOW ONLY, and asked only on the hash-DIFFERS path — the path that delivers.
Nothing here can suppress a delivery: the answer is never read, so a "no news"
verdict costs a log line and nothing else.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

POINT = "cron.novelty"

MAX_RESULT_CHARS = 3000

#: Test seam — see :mod:`kiro_crew.decisions.points.skills_select`.
_decide: Any = None


def build_state(job_id: str, title: str, last_result: str, new_result: str) -> dict[str, Any]:
    """The state: which job, and the two result texts being compared."""
    return {
        "job_id": str(job_id or ""),
        "title": str(title or ""),
        "last_result": (last_result or "")[:MAX_RESULT_CHARS],
        "new_result": (new_result or "")[:MAX_RESULT_CHARS],
    }


def build_baseline() -> dict[str, Any]:
    """The existing logic's answer on this path is always "deliver".

    Fixed, not measured: the hook is only reached when the hashes differ, and
    that branch has no other outcome. Agreement is therefore a measure of how
    often byte inequality was also a real difference.
    """
    return {"delivered": True}


#: Probability at or above which the Noul answer reads as "yes, there is news".
#: The answer IS a probability, and the baseline on this path is the boolean
#: ``delivered: True``, so a threshold is the only way the two can be compared at
#: all. 0.5 rather than a tuned value: this log exists to find out where the
#: split should be, so the initial reading must not presume one.
NOVELTY_THRESHOLD = 0.5


def row_agree(answers: Any, baseline: Any) -> bool | None:
    """Adapter handing :func:`agree` to the log, in the gate's calling shape.

    The generic rule cannot judge this row: the answer is a probability and the
    baseline is ``True``, so equality is False for every value including 1.0.
    Thresholding at :data:`NOVELTY_THRESHOLD` makes the comparison meaningful;
    the raw ``p`` stays in the row, so a different split can be re-derived from
    the log without a code change.
    """
    if not answers or not isinstance(baseline, dict):
        return None
    answer = answers.get("has_new_info")
    if answer is None:
        return None
    delivered = baseline.get("delivered")
    if not isinstance(delivered, bool):
        return None
    value = getattr(answer, "value", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        # A Noul answer is a probability. Anything else -- a string, None, a
        # bool from an implementation that answered the wrong question type --
        # is not comparable, and null says so rather than guessing a side.
        return None
    return agree(float(value) >= NOVELTY_THRESHOLD, delivered)


def agree(has_new_info: Any, delivered: bool = True) -> bool:
    """Whether the oracle's novelty verdict matches the delivery that happened."""
    if has_new_info is None:
        return False
    return bool(has_new_info) == bool(delivered)


async def shadow_cron_novelty(
    job_id: str,
    title: str,
    last_result: str,
    new_result: str,
    *,
    session_key: str | None = None,
) -> Any:
    """Ask the oracle whether the new result is news; never act on the answer."""
    try:
        from kiro_crew import decisions as _core
        from kiro_crew.decisions.types import Noul
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
        state = build_state(job_id, title, last_result, new_result)
        questions = [
            Noul(
                "has_new_info",
                "Does new_result contain information not in last_result worth " "delivering?",
            )
        ]
        return await fn(
            POINT,
            state,
            questions,
            session_key=session_key,
            baseline=build_baseline(),
            agree=row_agree,
        )
    except Exception:
        logger.debug("cron.novelty shadow failed", exc_info=True)
        return None
