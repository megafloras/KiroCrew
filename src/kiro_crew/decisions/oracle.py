"""The one seam every decision implementation is behind.

``decide`` (gate.py) resolves a :class:`DecisionOracle` from config and awaits
exactly one method on it. That is the whole extension surface: adding a second
provider is a new module with an ``ask``, not a change to the gate.

Why ``ask`` returns a result object rather than :data:`~.types.Answers`
----------------------------------------------------------------------
The log line carries ``in_tokens`` and ``cost_usd``, and only the
implementation knows them -- the token count comes back in the provider's own
``usage`` block, and the price per token is a property of the provider, not of
the seam. Returning a bare mapping would force the gate to reach back into the
implementation for the numbers (a second, order-dependent call) or to drop them.
:class:`OracleResult` keeps one round trip and one return value.

Both fields default to zero so an implementation with no usage reporting is
written without ceremony, and a zero in the log reads as "not reported" -- the
same convention the report side already has to apply to a row from a run where
the provider omitted usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kiro_crew.decisions.types import Answers, Question


@dataclass
class OracleResult:
    """One implementation's reply: the answers plus what they cost."""

    answers: Answers = field(default_factory=dict)
    #: Input tokens the provider charged for. 0 = not reported.
    in_tokens: int = 0
    #: Cost in USD, computed by the implementation from its own price. 0.0 =
    #: not reported (or genuinely free, e.g. a local model).
    cost_usd: float = 0.0


@runtime_checkable
class DecisionOracle(Protocol):
    """Answer *questions* against *state*, or raise.

    An implementation NEVER returns a partial or empty result to signal
    failure: it raises, and the gate converts that into ``None`` plus a logged
    ``error``. Two reasons the direction matters. A caller that got an empty
    mapping back could not tell "the provider said nothing" from "there was
    nothing to ask", so it would have to re-derive the question list to find
    out. And the error TEXT is the only thing that makes a shadow run
    diagnosable -- swallowing the exception inside the implementation would put
    ``error: null`` on a row that in fact never reached the model.

    Timeouts are the gate's job, not the implementation's: the gate wraps this
    call in ``asyncio.wait_for(provider.timeout_ms)`` so one budget governs
    every implementation. An implementation MAY pass its own transport timeout
    as well, but must not make the deadline longer than the gate's.
    """

    async def ask(self, state: dict | str, questions: list[Question]) -> OracleResult:
        """Evaluate every question in *questions* against *state*."""
        ...
