"""``decide`` -- the one function a point calls, and every refusal in front of it.

Five gates in a fixed order, cheapest first, each of which returns ``None``:

1. ``decisions.preview`` off -> refuse with ZERO awaits and zero IO.
2. the point is not configured, or its ``arm`` is ``off``.
3. the session hashes outside the point's ``bucket``.
4. the state carries something that looks like a credential.
5. the implementation raised, or outran ``provider.timeout_ms``.

Gates 1-3 write nothing at all. Gates 4-5 write a log row, because both are
findings an operator needs to see (see log.py's own note on why a silent scrub
is the worst available outcome).

The order is the contract, not an optimisation
---------------------------------------------
``preview`` is first so a disabled seam costs one attribute read: not "fast",
but *provably inert*, which is what lets this land in three hot paths at once.
Concretely, ``decide`` is an ``async def`` whose body performs no ``await``
before that check, so awaiting a refused call never yields to the loop and never
reaches an implementation -- ``test_decisions_gate`` pins that with an
implementation whose ``ask`` fails the test if it is ever entered.

Scrub is last of the cheap gates and BEFORE the network, which is the whole
point of it: a credential in the state must never reach a provider, so no
transport may be constructed above it.

Why an unresolvable config means off
-----------------------------------
The config is read from the live watcher's snapshot, never from disk -- a disk
read here would be IO on the path that promises none. Before the watcher is
primed the snapshot is ``None``, and that resolves to OFF rather than to a
``load()`` fallback. Fail-closed is the only safe direction for a gate whose
open state sends conversation text to a third party, and it costs nothing real:
the snapshot is primed at boot, long before any point fires.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from hashlib import sha256
from typing import Any

from kiro_crew import credential_patterns as _cred
from kiro_crew.decisions import log as _log
from kiro_crew.decisions.oracle import DecisionOracle, OracleResult
from kiro_crew.decisions.types import Answer, Answers, Choice, Noul, Question

logger = logging.getLogger(__name__)

ARM_OFF = "off"
ARM_SHADOW = "shadow"
ARM_LIVE = "live"

IMPL_JEV = "jev"
IMPL_LLM = "llm"


def _in_closed_range(value: object, lo: float, hi: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and lo <= float(value) <= hi
    )


def _answers_are_valid(answers: Answers, questions: list[Question]) -> bool:
    by_id = {question.id: question for question in questions}
    if set(answers) != set(by_id):
        return False
    for question_id, question in by_id.items():
        answer = answers.get(question_id)
        if not isinstance(answer, Answer) or answer.id != question_id:
            return False
        if not _in_closed_range(answer.p, 0.0, 1.0):
            return False
        if isinstance(question, Choice):
            if answer.value not in question.options:
                return False
        elif isinstance(question, Noul):
            if not _in_closed_range(answer.value, 0.0, 1.0):
                return False
    return True


#: The bucket modulus. A session's first 4 digest bytes are reduced mod this, so
#: ``bucket`` reads directly as a percentage of sessions.
_BUCKET_MOD = 100

#: Credential spellings this module refuses on ITS OWN, before consulting the
#: canonical scanner. Compiled from ``credential_patterns``, which exports
#: pattern SOURCE strings rather than compiled objects (it is an import leaf that
#: does not even import ``re``), so the compile has to happen at a consumer.
#:
#: This is the scrubber-side AWS spelling (``AKIA``/``ASIA``), not the wider
#: redaction one -- widening it here would change what a request-blocking gate
#: refuses, which is documented at the source as a security behaviour change
#: rather than a stricter-is-better tweak.
#:
#: It is NOT the whole of gate 4. ``VENDOR_TOKEN_PATTERNS`` carries a generic
#: ``sk-[A-Za-z0-9]{20,}`` spelling that ``redact_credentials`` does not, which is
#: why both run and why this list is not simply replaced by the call below.
_CREDENTIAL_RE = re.compile(
    "|".join(
        [_cred.AWS_KEY_ID, _cred.JWT_MULTI_SEGMENT]
        + [frag for _label, frag in _cred.VENDOR_TOKEN_PATTERNS]
    )
)


def _snapshot() -> Any:
    """The live config snapshot, or ``None``.

    Imported inside the function, not at module scope: ``config.live`` pulls the
    whole loader in, and ``decisions`` is imported lazily from hot paths
    precisely so that it adds nothing to their import cost.
    """
    from kiro_crew.config import live

    return live.snapshot()


def _decisions_config(config: Any | None) -> Any | None:
    """The ``decisions`` section of *config*, or of the live snapshot.

    *config* is an injection seam for tests and for a caller that already holds
    a config; ``None`` means "read the snapshot". Returns ``None`` when there is
    no config at all or it predates the section, which gate 1 reads as off.
    """
    cfg = config if config is not None else _snapshot()
    if cfg is None:
        return None
    return getattr(cfg, "decisions", None)


def in_bucket(session_key: str | None, bucket: int) -> bool:
    """Whether *session_key* falls inside a *bucket*-percent sample.

    The digest is the SAME one the log's ``session`` field carries, so a row is
    enough to re-derive why it was sampled without keeping the key.

    Both ends are closed forms, not approximations: ``bucket=0`` admits nothing
    (every residue is ``>= 0``) and ``bucket=100`` admits everything (no residue
    reaches 100). A value outside 0..100 is clamped rather than rejected --
    ``arm`` is the switch, and a typo'd bucket must not become a third,
    undocumented way to disable a point.
    """
    bucket = max(0, min(_BUCKET_MOD, int(bucket)))
    digest = sha256((session_key or "").encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _BUCKET_MOD < bucket


def state_text(state: dict | str) -> str:
    """*state* as one string, for the credential scan.

    A ``dict`` is rendered with ``json.dumps`` rather than ``str()``: ``str()``
    on a nested structure can elide content behind a ``__repr__``, and a
    credential hidden inside an object whose repr is ``<Foo object at 0x...>``
    would pass the scan and then be serialised onto the wire by the
    implementation. Rendering the way the wire will renders what the wire sees.
    """
    if isinstance(state, str):
        return state
    import json

    try:
        return json.dumps(state, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        # Unserialisable state cannot be scanned honestly, and gate 4's job is to
        # refuse what it cannot clear -- ``repr`` here only feeds the scan, and a
        # scan that finds nothing in a repr it could not read is why the
        # implementation will hit the same failure and raise.
        return repr(state)


def scrub_reason(state: dict | str) -> str | None:
    """Why *state* must not leave the machine, or ``None`` when both scans clear.

    The cheap local credential regex runs first. The canonical credential and
    exfiltration-URL scanners run only after gates 1-3 admitted the point, and
    either scanner's warnings refuse the whole state. Scanner failures also
    refuse, because an external request cannot be cleared by a scan that did not
    complete.
    """
    text = state_text(state)
    if _CREDENTIAL_RE.search(text) is not None:
        return "scrubbed: credential"
    try:
        from kiro_crew.security.redaction import redact_credentials

        _cleaned, warnings = redact_credentials(text)
    except Exception:  # pragma: no cover - defensive
        logger.warning("decisions: credential scan failed; refusing the state")
        return "scrubbed: credential-scan-failed"
    if warnings:
        return "scrubbed: credential"
    try:
        from kiro_crew.security import redact_exfiltration_urls

        _cleaned, warnings = redact_exfiltration_urls(text)
    except Exception:  # pragma: no cover - defensive
        logger.warning("decisions: exfiltration URL scan failed; refusing the state")
        return "scrubbed: exfiltration-scan-failed"
    return "scrubbed: exfiltration-url" if warnings else None


def _resolve_impl(name: str, provider: Any) -> DecisionOracle:
    """The implementation named *name*. Raises ``ValueError`` on an unknown name.

    Both imports are function-local so that arming a point on ``llm`` does not
    drag ``aiohttp`` and the vault in, and arming it on ``jev`` does not drag
    the session/dashboard side in. They are also the reason an unknown name
    raises rather than falling back: a silent fallback would send state to a
    provider the operator did not name.
    """
    if name == IMPL_JEV:
        from kiro_crew.decisions.impl_jev import JevOracle

        return JevOracle(provider)
    if name == IMPL_LLM:
        from kiro_crew.decisions.impl_llm import LlmOracle

        return LlmOracle(provider)
    raise ValueError(f"unknown decisions impl {name!r}")


def _armed_entry(point: str, session_key: str | None, config: Any | None) -> Any | None:
    """The point's config entry when gates 1-3 all admit it, else ``None``.

    Gates 1-3 are the three cheap refusals -- ``preview`` off, the point absent
    or ``arm: off``, the session outside the ``bucket`` -- and they are the ones a
    caller can ask about BEFORE building a state. Extracted rather than copied so
    ``is_armed`` and ``decide`` cannot drift: a fourth cheap gate added here is
    added to both at once.

    Performs no ``await``, no IO and no import of an implementation, exactly as
    ``decide``'s own preamble did before this was lifted out of it.
    """
    decisions = _decisions_config(config)
    if decisions is None or not getattr(decisions, "preview", False):
        return None

    points = getattr(decisions, "points", None) or {}
    entry = points.get(point) if isinstance(points, dict) else None
    if entry is None:
        # Not a warning: an unconfigured point is the normal state of a point
        # that shipped after the operator's config was written, and the loader
        # already warns about names it does not recognise.
        return None
    if str(getattr(entry, "arm", ARM_OFF) or ARM_OFF) == ARM_OFF:
        return None

    if not in_bucket(session_key, getattr(entry, "bucket", _BUCKET_MOD)):
        return None
    return entry


def is_armed(point: str, *, session_key: str | None = None, config: Any | None = None) -> bool:
    """Whether *point* would get past gates 1-3 right now. Never raises.

    For a hook that must do real work to BUILD its state -- walking the skill
    tree, reading frontmatter -- and would otherwise do that work on the default
    configuration and then hand it to a ``decide`` that refuses on its first
    line. Cheap by construction: attribute reads and one hash, no await, no IO.

    It is not a second gate and it grants nothing. ``decide`` re-runs all five
    refusals itself, so a caller that skips this check is merely wasteful, and a
    caller that races a config change between the two only loses or gains one
    row. An exception here reads as NOT armed: the seam declining to observe is
    always preferable to a state-building path that raises into a turn.
    """
    try:
        return _armed_entry(point, session_key, config) is not None
    except Exception:  # pragma: no cover - defensive
        logger.debug("decisions: is_armed(%s) failed", point, exc_info=True)
        return False


async def decide(
    point: str,
    state: dict | str,
    questions: list[Question],
    *,
    session_key: str | None = None,
    baseline: dict | None = None,
    agree: _log.AgreeFn | None = None,
    config: Any | None = None,
) -> Answers | None:
    """Ask *questions* about *state* at *point*, or return ``None``.

    ``None`` is the ONLY failure signal and it is never exceptional: every
    refusal above, every provider error, every timeout, and the whole ``shadow``
    arm all return it. A caller therefore needs no try/except and no arm check
    of its own -- ``answers = await decide(...)`` followed by ``if answers is
    None: <existing behaviour>`` is the complete integration, and it is the same
    line whether the seam is off, shadowing, or live.

    *baseline* is what the EXISTING logic concluded, passed in so the row can
    record whether the two agreed. It never influences the answer.

    *agree* is the point's own agreement rule, ``(answers, baseline) -> bool |
    None``. Left unset, the row falls back to value equality over a shared
    question-id set (:func:`kiro_crew.decisions.log.agree`), which two of the
    three shipped points cannot be judged by -- see that module's
    ``_resolved_agree``. A rule that raises costs the ROW, never the turn.

    *config* injects a config instead of reading the live snapshot; production
    callers leave it unset.
    """
    # ---- Gates 1-3: preview, arm, bucket. No await, no IO, no import of an
    # implementation -- see ``_armed_entry``, which ``is_armed`` shares. ----
    entry = _armed_entry(point, session_key, config)
    if entry is None:
        return None
    decisions = _decisions_config(config)
    arm = str(getattr(entry, "arm", ARM_OFF) or ARM_OFF)

    impl_name = str(getattr(entry, "impl", IMPL_LLM) or IMPL_LLM)
    provider = getattr(decisions, "provider", None)

    async def _write(
        *,
        latency_ms: int,
        answers: Answers | None = None,
        in_tokens: int = 0,
        cost_usd: float = 0.0,
        scrubbed: bool = False,
        error: str | None = None,
    ) -> None:
        # Guarded here as well as inside ``log.append``. ``append`` protects the
        # WRITE; this protects BUILDING the row, which ``append`` never sees.
        # ``build_row`` compares the caller's ``baseline`` against the answers,
        # and a baseline is an arbitrary object supplied by a point file -- one
        # whose ``__eq__`` raises would otherwise escape ``decide`` and break the
        # very turn this seam promises never to affect. The same guard covers a
        # point's own ``agree`` rule, which runs inside ``build_row``.
        try:
            row = _log.build_row(
                point=point,
                arm=arm,
                impl=impl_name,
                session_key=session_key,
                latency_ms=latency_ms,
                answers=answers,
                baseline=baseline,
                in_tokens=in_tokens,
                cost_usd=cost_usd,
                scrubbed=scrubbed,
                agree_fn=agree,
                error=error,
            )
            await asyncio.to_thread(_log.append, row)
        except Exception as exc:
            logger.warning("decisions: could not record %s row: %s", point, exc)

    # ---- Gate 4: credentials and exfiltration-shaped URLs. Refuse before any
    # transport exists, and preserve which scanner fired in the row. ----
    refusal = scrub_reason(state)
    if refusal is not None:
        await _write(latency_ms=0, scrubbed=True, error=refusal)
        return None

    # ---- Gate 5: the call itself. ----
    started = time.monotonic()
    try:
        impl = _resolve_impl(impl_name, provider)
        timeout = _timeout_secs(provider)
        result = await asyncio.wait_for(impl.ask(state, questions), timeout=timeout)
    except asyncio.TimeoutError:
        # Named explicitly rather than folded into the generic branch: "timeout"
        # is the one error a report can act on mechanically (raise timeout_ms, or
        # accept the loss rate), so it must not arrive as a provider-specific
        # message that differs per implementation.
        await _write(latency_ms=_elapsed_ms(started), error="timeout")
        return None
    except asyncio.CancelledError:
        # Cancellation is the CALLER going away, not a decision failure. Writing
        # a row would attribute the caller's shutdown to the provider, and
        # swallowing it would break structured concurrency, so: no row, re-raise.
        raise
    except Exception as exc:
        await _write(latency_ms=_elapsed_ms(started), error=f"{type(exc).__name__}: {exc}"[:300])
        return None

    latency_ms = _elapsed_ms(started)
    answers = result.answers if isinstance(result, OracleResult) else None
    if not answers:
        # An implementation that returns an empty result broke its own contract
        # (oracle.py: raise, never return empty). Record it as an error rather
        # than as a successful decision with nothing in it, so the row cannot be
        # read as agreement.
        await _write(latency_ms=latency_ms, error="empty result")
        return None
    if not _answers_are_valid(answers, questions):
        await _write(latency_ms=latency_ms, error="invalid result")
        return None

    await _write(
        latency_ms=latency_ms,
        answers=answers,
        in_tokens=result.in_tokens,
        cost_usd=result.cost_usd,
    )

    if arm == ARM_LIVE:
        return answers
    # ``shadow`` -- and any arm value the loader let through that is neither
    # ``off`` nor ``live``. Returning None for an unrecognised arm keeps the
    # unknown case on the observe-only side of the switch.
    return None


def _timeout_secs(provider: Any) -> float:
    """``provider.timeout_ms`` in seconds, floored so it is always a real budget.

    A zero or negative ``timeout_ms`` would make ``wait_for`` cancel immediately
    and turn every row into ``error: timeout`` -- a config typo that looks like a
    broken provider. The floor makes it look like a fast timeout instead.
    """
    raw = getattr(provider, "timeout_ms", 1000) if provider is not None else 1000
    try:
        ms = float(raw)
    except (TypeError, ValueError):
        ms = 1000.0
    return max(0.001, ms / 1000.0)


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since *started* (a ``time.monotonic()`` reading)."""
    return int((time.monotonic() - started) * 1000)
