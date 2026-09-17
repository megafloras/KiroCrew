"""The DecisionOracle seam: one function, five gates in front of it, off by default.

A caller asks typed questions about a state and gets typed answers, or ``None``:

    from kiro_crew.decisions import decide, Choice

    answers = await decide(
        "skills.dedupe",
        {"candidate": text, "existing": keys},
        [Choice(id="verdict", prompt="Is this a duplicate?", options=["NONE", "DUP"])],
        session_key=session_key,
        baseline={"verdict": existing_verdict},
    )
    if answers is None:
        ...  # exactly what the code did before
    else:
        ...  # only reached when the point is armed ``live``

``None`` covers every refusal and every failure -- see :func:`decide` -- so a
call site needs no try/except, no feature check and no arm check of its own. That
is what makes this safe to place in a hot path: with ``decisions.preview`` false
(the default) the call performs no ``await``, no IO and no import beyond this
module.

Import this package lazily, inside the function that calls ``decide``. Nothing
here imports the config loader, ``aiohttp`` or the session layer at module scope,
but a top-level import in a hot module would still put this package on that
module's import path for no benefit.

The package deliberately does NOT import from ``decisions.points``: points
depend on the seam, never the reverse, so the dependency stays acyclic and a
broken point file cannot make ``decide`` unimportable.
"""

from __future__ import annotations

from kiro_crew.decisions.gate import (
    ARM_LIVE,
    ARM_OFF,
    ARM_SHADOW,
    IMPL_JEV,
    IMPL_LLM,
    decide,
    is_armed,
)
from kiro_crew.decisions.oracle import DecisionOracle, OracleResult
from kiro_crew.decisions.types import Answer, Answers, Choice, Noul, Question

__all__ = [
    "ARM_LIVE",
    "ARM_OFF",
    "ARM_SHADOW",
    "IMPL_JEV",
    "IMPL_LLM",
    "Answer",
    "Answers",
    "Choice",
    "DecisionOracle",
    "Noul",
    "OracleResult",
    "Question",
    "decide",
    "is_armed",
]
