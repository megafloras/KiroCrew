"""The LLM comparison lane: the same questions, answered by a chat model.

This exists so the seam can be measured before a Jev key exists, and so that
once one does, the two rows are comparable field-for-field. It reuses
``llm_helpers.run_bg_oneliner``, which is the established tool-free background
one-liner path -- so the questions run on the cheap model, in an ephemeral
``_bg`` session, with permission requests denied and SEL-audited under
``sel_source="decisions"``.

Why a module-level ``sessions`` setter rather than a parameter
------------------------------------------------------------
``run_bg_oneliner`` needs a ``SessionManager``-like object, and the seam's
callers are three hot paths that have no reason to know one exists -- one of
them (``skills.select``) runs inside a synchronous ``build_message``. Threading
``sessions`` through ``decide`` would put a dashboard object in the signature of
every point file and make the gate's own contract depend on it.

So the dashboard registers it ONCE at startup (``DashboardState.__init__``, the
single funnel both construction sites pass through) and this module holds a weak
reference to it. Unset -> ``ask`` raises, the gate logs it and returns ``None``:
a CLI or test process that never built a dashboard state simply has no LLM lane,
which is the correct outcome and not an error to fix at the call site.

The reference is weak so that holding it here cannot keep a torn-down
``SessionManager`` (and the ACP runtimes behind it) alive for the life of the
process. A dead reference reads as unset.

Why strict JSON out, and why a parse failure raises
--------------------------------------------------
The prompt asks for one JSON object and nothing else. A chat model will
sometimes wrap it in a fence or a sentence, so the parser tolerates those two
shapes -- and nothing further. Anything else raises, because the alternative is
inventing an answer: a row that says ``value: "NONE"`` because the parser fell
back to a default is worse than a row that says ``error``, since only one of the
two is visible as a problem in the report.

Confidence is NOT asked for. A chat model's self-reported certainty is not
calibrated and putting it in the same column as Jev's derived confidence would
make the two look comparable when they are not. ``p`` for a Choice comes back as
``null`` -> 0.0, which the report reads as "not reported"; for a Noul the
model returns the probability itself, which is the one number it can give
meaningfully because the question already asks for one.
"""

from __future__ import annotations

import json
import logging
import math
import re
import weakref
from typing import Any

from kiro_crew.decisions.oracle import OracleResult
from kiro_crew.decisions.types import Answer, Answers, Choice, Noul, Question

logger = logging.getLogger(__name__)

#: SEL attribution for permission denials on this lane's background turns.
_SEL_SOURCE = "decisions"

#: Weak reference to the process's ``SessionManager``, or ``None``. See the
#: module docstring on why weak.
_SESSIONS_REF: weakref.ReferenceType | None = None


def set_bg_sessions(sessions: Any) -> None:
    """Register the ``SessionManager`` this lane runs its background turns on.

    Called once by the dashboard at startup. Idempotent, and safe to call with
    ``None`` to clear -- a test that installs a fake manager clears it in
    teardown rather than leaving a stale one for the next test.
    """
    global _SESSIONS_REF
    if sessions is None:
        _SESSIONS_REF = None
        return
    try:
        _SESSIONS_REF = weakref.ref(sessions)
    except TypeError:
        # Not weak-referenceable (a plain object with __slots__, some test
        # doubles). Keeping a strong reference is the lesser evil: the lane
        # working matters more here than the teardown nicety the weak ref buys,
        # and such an object is not the real SessionManager.
        _SESSIONS_REF = _StrongRef(sessions)  # type: ignore[assignment]


class _StrongRef:
    """A ``weakref.ref``-shaped holder for objects that cannot be weak-referenced."""

    __slots__ = ("_obj",)

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __call__(self) -> Any:
        return self._obj


def bg_sessions() -> Any | None:
    """The registered ``SessionManager``, or ``None`` if unset or collected."""
    ref = _SESSIONS_REF
    return ref() if ref is not None else None


def build_prompt(state: dict | str, questions: list[Question]) -> str:
    """One prompt covering every question, asking for one JSON object back.

    All questions go in a single turn -- the same shape Jev's API takes -- so the
    two lanes see identical state and the latency numbers describe the same unit
    of work. Asking them one at a time would also multiply the cost by the state
    size, which is the largest part of the payload.
    """
    state_text = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
    )
    lines: list[str] = [
        "You are a fast classifier. Answer every question below about the STATE.",
        "Reply with ONE JSON object and nothing else: no prose, no code fence.",
        "",
        "STATE:",
        state_text,
        "",
        "QUESTIONS:",
    ]
    shape: list[str] = []
    for q in questions:
        if isinstance(q, Choice):
            lines.append(f"- {q.id}: {q.prompt} Pick exactly one of: {', '.join(q.options)}")
            shape.append(f'  "{q.id}": {{"value": "<one of the listed options>", "p": <0..1>}}')
        elif isinstance(q, Noul):
            lines.append(f"- {q.id}: {q.prompt} Answer with the probability it is true.")
            shape.append(f'  "{q.id}": {{"value": <0..1>, "p": <0..1>}}')
    lines += [
        "",
        "Reply in exactly this shape:",
        "{",
        ",\n".join(shape),
        "}",
        "",
        '"p" is how much probability you put on your own answer, 0 to 1.',
    ]
    return "\n".join(lines)


#: Leading/trailing markdown fence a model adds around JSON despite being told
#: not to. Stripped rather than refused because it is the single most common
#: deviation and carries no ambiguity about what was meant.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_reply(text: str, questions: list[Question]) -> Answers:
    """Parse the model's reply into :data:`~.types.Answers`. Raises on anything else.

    Two tolerances and no more: a markdown fence, and prose around a single
    top-level object (recovered by taking the outermost brace pair). Both are
    unambiguous. A missing question, a non-numeric ``p``, or a Choice answer
    outside its own option list all raise -- see the module docstring on why a
    default is worse than an error here.
    """
    raw = _FENCE_RE.sub("", text or "").strip()
    if not raw:
        raise ValueError("empty reply")
    try:
        parsed = json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("reply is not JSON") from None
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"reply is {type(parsed).__name__}, not an object")

    answers: Answers = {}
    for q in questions:
        entry = parsed.get(q.id)
        if not isinstance(entry, dict):
            raise ValueError(f"no answer for {q.id!r}")
        answers[q.id] = _answer_from_entry(q, entry)
    return answers


def _answer_from_entry(q: Question, entry: dict) -> Answer:
    """One answer, validated against the question that asked for it."""
    raw_value = entry.get("value")
    parsed_p = _as_float_or_none(entry.get("p"))
    p = parsed_p if parsed_p is not None else 0.0

    if isinstance(q, Choice):
        if not isinstance(raw_value, str):
            raise ValueError(f"{q.id!r}: value is not a string")
        if q.options and raw_value not in q.options:
            # Refused, not snapped to the nearest option: a model that answered
            # outside the domain did not answer this question, and a snap would
            # record a verdict nobody produced.
            raise ValueError(f"{q.id!r}: {raw_value!r} is not one of {q.options}")
        return Answer(id=q.id, value=raw_value, p=p, confidence=None)

    if isinstance(q, Noul):
        value = _as_float_strict(raw_value, q.id)
        value = min(1.0, max(0.0, value))
        # Same identity as the Jev lane: for a Noul the answer IS a probability,
        # so ``p`` mirrors it when the model did not report one separately.
        return Answer(
            id=q.id,
            value=value,
            p=value if parsed_p is None else parsed_p,
            confidence=None,
        )

    raise ValueError(f"unsupported question type {type(q).__name__}")


def _as_float_or_none(raw: Any) -> float | None:
    """*raw* as a finite float, or ``None`` when absent or not numeric."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _as_float(raw: Any) -> float:
    """*raw* as a finite float, or 0.0 when it is unusable."""
    return _as_float_or_none(raw) or 0.0


def _as_float_strict(raw: Any, qid: str) -> float:
    """*raw* as a float, raising when it is not numeric.

    Strict where :func:`_as_float` is lenient: ``p`` is metadata a missing value
    only degrades, but ``value`` IS the answer and a 0.0 stand-in for it would be
    a fabricated verdict at the low end of every scale.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"{qid!r}: value is not numeric")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{qid!r}: value {raw!r} is not numeric") from None


class LlmOracle:
    """Implements :class:`~kiro_crew.decisions.oracle.DecisionOracle` via a chat model.

    *provider* is accepted (and its ``model`` honoured when set to something
    other than a Jev model id) so the gate can construct either implementation
    with one signature. ``timeout_ms`` is not read here: the gate's ``wait_for``
    is the budget, and ``run_bg_oneliner``'s own ``timeout`` is passed the same
    value so the background turn stops rather than outliving the decision.
    """

    def __init__(self, provider: Any = None) -> None:
        self._timeout_ms = getattr(provider, "timeout_ms", 1000) if provider is not None else 1000

    async def ask(self, state: dict | str, questions: list[Question]) -> OracleResult:
        """Ask every question in one background turn. Raises on any failure."""
        if not questions:
            raise ValueError("no questions to ask")
        sessions = bg_sessions()
        if sessions is None:
            # Not a warning: a process with no dashboard state (CLI, tests) has
            # no LLM lane by construction, and this is the message that says so
            # in the row's ``error``.
            raise RuntimeError("no background sessions registered")

        from kiro_crew.llm_helpers import run_bg_oneliner

        text = await run_bg_oneliner(
            sessions,
            build_prompt(state, questions),
            sel_source=_SEL_SOURCE,
            timeout=max(0.001, _as_float(self._timeout_ms) / 1000.0),
        )
        # Token usage is not reported by this path, so the row carries 0/0.0 --
        # the "not reported" convention from oracle.py. Guessing from prompt
        # length would put a number in the same column as Jev's billed count and
        # make the two silently incomparable.
        return OracleResult(answers=parse_reply(text, questions))
