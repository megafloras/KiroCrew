"""``decide``'s five gates, in order, each proven to REFUSE rather than merely to differ.

The load-bearing test in this file is
:class:`TestPreviewOffPerformsNoAwait`: it is the reason this seam is safe to
place in three hot paths, and it is asserted by making the implementation fail
the test if it is entered at all, rather than by timing anything.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from kiro_crew import credential_patterns as _cred
from kiro_crew.config.sections import (
    DecisionPointConfig,
    DecisionProviderConfig,
    DecisionsConfig,
)
from kiro_crew.decisions import gate as gate_mod
from kiro_crew.decisions import log as log_mod
from kiro_crew.decisions.gate import decide, in_bucket
from kiro_crew.decisions.oracle import OracleResult
from kiro_crew.decisions.types import Answer, Choice, Noul

# ---------------------------------------------------------------------------
# Fixtures and doubles
# ---------------------------------------------------------------------------

#: AWS key-id samples, assembled from the prefix list in
#: ``credential_patterns`` rather than written out. A contiguous key-shaped
#: literal is refused by the repo's own secret scanners -- correctly, since
#: neither the content scan nor Semgrep can tell a test vector from a real
#: leak -- and deriving the samples means a prefix added there is exercised
#: here without this list being edited to match.
_AWS_KEY_BODY = "A2B3C4D5E6F7G8H9"  # 16 of [A-Z0-9], per AWS_KEY_ID_BODY
_AWS_KEY_SAMPLES = tuple(prefix + _AWS_KEY_BODY for prefix in _cred.AWS_KEY_ID_PREFIXES.split("|"))

POINT = "skills.select"
QUESTIONS = [Choice(id="verdict", prompt="Duplicate?", options=["NONE", "DUP"])]


def _config(
    *,
    preview: bool = True,
    arm: str = "shadow",
    impl: str = "fake",
    bucket: int = 100,
    timeout_ms: int = 1000,
    point: str = POINT,
):
    """A config object shaped like the one ``decide`` reads off the live snapshot.

    A real ``KiroCrewConfig`` would work too, but constructing one loads the
    whole config module for a two-field read; the gate only ever touches
    ``.decisions``, so a stand-in with that attribute is the honest surface.
    """

    class _Cfg:
        decisions = DecisionsConfig(
            preview=preview,
            provider=DecisionProviderConfig(timeout_ms=timeout_ms),
            points={point: DecisionPointConfig(arm=arm, impl=impl, bucket=bucket)},
        )

    return _Cfg()


class _RecordingOracle:
    """Answers ``verdict`` with a fixed value and records every call."""

    def __init__(self, value: str = "DUP", *, in_tokens: int = 1900) -> None:
        self.value = value
        self.in_tokens = in_tokens
        self.calls: list[tuple] = []

    async def ask(self, state, questions):
        self.calls.append((state, questions))
        return OracleResult(
            answers={
                q.id: Answer(id=q.id, value=self.value, p=0.87, confidence=0.82) for q in questions
            },
            in_tokens=self.in_tokens,
            cost_usd=self.in_tokens * 0.042 / 1e6,
        )


class _ExplodingOracle:
    """Fails the TEST if it is ever entered.

    Not ``raise`` -- a raise would be caught by gate 5 and logged as an error,
    which is a pass-looking outcome. ``pytest.fail`` inside a coroutine that the
    gate awaits surfaces as a test failure.
    """

    def __init__(self) -> None:
        self.entered = False

    async def ask(self, state, questions):  # pragma: no cover - must never run
        self.entered = True
        pytest.fail("implementation was entered on a path that must perform no await")


@pytest.fixture
def install_impl(monkeypatch):
    """Register an implementation under the name ``fake`` for ``_resolve_impl``."""

    def _install(oracle):
        real = gate_mod._resolve_impl

        def _resolve(name, provider):
            if name == "fake":
                return oracle
            return real(name, provider)

        monkeypatch.setattr(gate_mod, "_resolve_impl", _resolve)
        return oracle

    return _install


@pytest.fixture
def log_home(tmp_path, monkeypatch):
    """Point the log at *tmp_path* and return a reader for the rows written."""
    monkeypatch.setattr(log_mod, "log_dir", lambda: tmp_path / "decisions")

    def _rows():
        directory = tmp_path / "decisions"
        if not directory.exists():
            return []
        out = []
        for path in sorted(directory.glob("decisions-*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out

    _rows.dir = directory = tmp_path / "decisions"  # type: ignore[attr-defined]
    assert directory  # keeps the attribute meaningful to a reader
    return _rows


# ---------------------------------------------------------------------------
# Gate 1 -- preview
# ---------------------------------------------------------------------------


class TestPreviewOffPerformsNoAwait:
    """``preview=false`` must cost nothing measurable, not merely return None."""

    def test_refuses_without_entering_the_implementation(self, install_impl, log_home):
        oracle = install_impl(_ExplodingOracle())
        result = asyncio.run(
            decide(POINT, "hello", QUESTIONS, config=_config(preview=False, arm="live"))
        )
        assert result is None
        assert oracle.entered is False
        assert log_home() == [], "a refused decision must not write a log row"

    def test_the_coroutine_completes_on_its_first_step(self, install_impl):
        """No await before the refusal -- proven by driving the coroutine by hand.

        ``send(None)`` runs the body until it either yields to the loop (an
        awaited future) or finishes. A ``StopIteration`` on the very first send
        means the body reached ``return None`` without ever suspending, which is
        the strongest available form of "performs zero awaits": it does not
        depend on how fast anything ran.
        """
        install_impl(_ExplodingOracle())
        coro = decide(POINT, "hello", QUESTIONS, config=_config(preview=False))
        with pytest.raises(StopIteration) as caught:
            coro.send(None)
        assert caught.value.value is None

    def test_absent_section_is_off(self, install_impl, log_home):
        """A config predating the section must read as off, not crash."""
        install_impl(_ExplodingOracle())

        class _Old:
            pass

        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=_Old())) is None
        assert log_home() == []

    def test_no_snapshot_is_off(self, install_impl, log_home, monkeypatch):
        """An unprimed live watcher fails CLOSED rather than reading from disk."""
        install_impl(_ExplodingOracle())
        monkeypatch.setattr(gate_mod, "_snapshot", lambda: None)
        assert asyncio.run(decide(POINT, "hi", QUESTIONS)) is None
        assert log_home() == []


# ---------------------------------------------------------------------------
# Gate 2 -- point configured and armed
# ---------------------------------------------------------------------------


class TestArmAndPointLookup:
    def test_arm_off_refuses_and_logs_nothing(self, install_impl, log_home):
        oracle = install_impl(_ExplodingOracle())
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config(arm="off"))) is None
        assert oracle.entered is False
        assert log_home() == []

    def test_unconfigured_point_refuses(self, install_impl, log_home):
        install_impl(_ExplodingOracle())
        cfg = _config(arm="live", point="some.other.point")
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=cfg)) is None
        assert log_home() == []

    def test_unknown_arm_does_not_return_answers(self, install_impl, log_home):
        """An arm the loader let through must land on the observe-only side.

        Constructed directly rather than loaded, because the loader normalizes an
        unknown arm to ``off``: this pins the gate's OWN behaviour for the case
        where a future loader, or a caller building the dataclass by hand, hands
        it something it does not know.
        """
        oracle = install_impl(_RecordingOracle())
        cfg = _config(arm="experimental")
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=cfg)) is None
        assert len(oracle.calls) == 1, "the call still happened, so it is still logged"
        assert log_home()[0]["arm"] == "experimental"


# ---------------------------------------------------------------------------
# Gate 3 -- bucket
# ---------------------------------------------------------------------------


class TestBucket:
    def test_bucket_zero_admits_nothing(self):
        assert not any(in_bucket(f"session-{i}", 0) for i in range(200))

    def test_bucket_hundred_admits_everything(self):
        assert all(in_bucket(f"session-{i}", 100) for i in range(200))

    def test_a_key_is_consistently_in_or_out(self):
        assert len({in_bucket("stable-key", 50) for _ in range(20)}) == 1

    def test_bucket_is_roughly_the_percentage_it_claims(self):
        """Not a distribution test -- a coherence floor that the modulus is not skewed."""
        admitted = sum(in_bucket(f"session-{i}", 50) for i in range(1000))
        assert 400 < admitted < 600, admitted

    def test_out_of_range_is_clamped_not_treated_as_off(self):
        assert in_bucket("k", 500) is True
        assert in_bucket("k", -5) is False

    def test_bucket_zero_refuses_before_the_implementation(self, install_impl, log_home):
        oracle = install_impl(_ExplodingOracle())
        cfg = _config(arm="live", bucket=0)
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, session_key="s", config=cfg)) is None
        assert oracle.entered is False
        assert log_home() == [], "a sampling exclusion is not a finding, so no row"

    def test_the_digest_matches_the_logged_session_field(self, install_impl, log_home):
        """A row's ``session`` must be enough to re-derive why it was sampled."""
        install_impl(_RecordingOracle())
        asyncio.run(decide(POINT, "hi", QUESTIONS, session_key="abc", config=_config()))
        assert log_home()[0]["session"] == log_mod.session_digest("abc")


# ---------------------------------------------------------------------------
# Gate 4 -- credential scrub
# ---------------------------------------------------------------------------


class TestScrub:
    @pytest.mark.parametrize(
        "payload",
        [
            # Both prefixes in both shapes: embedded in prose, and with no word
            # boundary before the key (``aws_AKIA...`` has no ``\b`` between
            # ``_`` and ``A``, so a bounded pattern would miss it).
            *(f"creds are {key} ok" for key in _AWS_KEY_SAMPLES),
            *(f"aws_{key}" for key in _AWS_KEY_SAMPLES),
            "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghij",
            "sk-ant-abcdefghijklmnopqrstuvwxyz",
            "github_pat_" + "a" * 44,
            "glpat-abcdefghijklmnopqr",
        ],
    )
    def test_credential_in_state_refuses_before_the_network(self, install_impl, log_home, payload):
        oracle = install_impl(_ExplodingOracle())
        cfg = _config(arm="live")
        assert asyncio.run(decide(POINT, payload, QUESTIONS, config=cfg)) is None
        assert oracle.entered is False, "state with a credential must never reach a provider"
        rows = log_home()
        assert len(rows) == 1, "a scrub hit IS a finding and must be visible"
        assert rows[0]["scrubbed"] is True
        assert rows[0]["error"] == "scrubbed: credential"
        assert rows[0]["answers"] is None

    def test_credential_nested_in_a_dict_state_is_found(self, install_impl, log_home):
        oracle = install_impl(_ExplodingOracle())
        state = {"messages": [{"text": f"key {_AWS_KEY_SAMPLES[0]}"}]}
        assert asyncio.run(decide(POINT, state, QUESTIONS, config=_config())) is None
        assert oracle.entered is False
        assert log_home()[0]["scrubbed"] is True

    def test_the_state_itself_is_never_written_to_the_log(self, install_impl, log_home):
        install_impl(_RecordingOracle())
        secret_ish = "the user said something private about their salary"
        asyncio.run(decide(POINT, secret_ish, QUESTIONS, config=_config()))
        assert secret_ish not in json.dumps(log_home())

    def test_ordinary_state_passes(self, install_impl, log_home):
        oracle = install_impl(_RecordingOracle())
        assert asyncio.run(decide(POINT, "just words", QUESTIONS, config=_config())) is None
        assert len(oracle.calls) == 1
        assert log_home()[0]["scrubbed"] is False


# ---------------------------------------------------------------------------
# Gate 5 -- the call, its timeout, and its errors
# ---------------------------------------------------------------------------


class _SlowOracle:
    """Sleeps *secs* before answering."""

    def __init__(self, secs: float) -> None:
        self.secs = secs
        self.finished = False

    async def ask(self, state, questions):
        await asyncio.sleep(self.secs)
        self.finished = True
        return OracleResult(answers={q.id: Answer(q.id, "NONE", 1.0) for q in questions})


class TestTimeout:
    def test_a_slow_provider_returns_none_within_the_budget(self, install_impl, log_home):
        """timeout_ms=100 against a 2s provider: None, fast, and error='timeout'."""
        oracle = install_impl(_SlowOracle(2.0))

        async def _run():
            loop = asyncio.get_running_loop()
            started = loop.time()
            result = await decide(
                POINT, "hi", QUESTIONS, config=_config(arm="live", timeout_ms=100)
            )
            return result, loop.time() - started

        result, elapsed = asyncio.run(_run())
        assert result is None
        assert oracle.finished is False, "the provider must be cancelled, not merely ignored"
        # Generous ceiling: the assertion is that the budget bounds the wait, not
        # that the machine is fast. A regression that dropped the timeout would
        # take the full 2s and fail this by 10x.
        assert elapsed < 1.0, elapsed
        rows = log_home()
        assert len(rows) == 1
        assert rows[0]["error"] == "timeout"
        assert rows[0]["answers"] is None

    def test_a_nonpositive_timeout_is_floored_not_disabled(self, install_impl, log_home):
        """timeout_ms=0 must behave as a very fast timeout, never as 'no timeout'."""
        install_impl(_SlowOracle(2.0))

        async def _run():
            loop = asyncio.get_running_loop()
            started = loop.time()
            await decide(POINT, "hi", QUESTIONS, config=_config(timeout_ms=0))
            return loop.time() - started

        assert asyncio.run(_run()) < 1.0
        assert log_home()[0]["error"] == "timeout"


class _RaisingOracle:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def ask(self, state, questions):
        raise self.exc


class TestErrors:
    def test_a_provider_error_returns_none_and_logs_the_reason(self, install_impl, log_home):
        install_impl(_RaisingOracle(RuntimeError("HTTP 429: slow down")))
        cfg = _config(arm="live")
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=cfg)) is None
        row = log_home()[0]
        assert row["error"] == "RuntimeError: HTTP 429: slow down"
        assert row["answers"] is None

    def test_a_bad_json_error_returns_none_and_logs_the_reason(self, install_impl, log_home):
        install_impl(_RaisingOracle(ValueError("response is not JSON: line 1")))
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config())) is None
        assert "not JSON" in log_home()[0]["error"]

    def test_an_unknown_impl_name_returns_none_and_logs(self, install_impl, log_home):
        """No fallback to another provider -- state must only go where it was told."""
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config(impl="nope"))) is None
        assert "unknown decisions impl" in log_home()[0]["error"]

    def test_an_empty_result_is_recorded_as_an_error_not_as_agreement(self, install_impl, log_home):
        class _Empty:
            async def ask(self, state, questions):
                return OracleResult(answers={})

        install_impl(_Empty())
        assert asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config(arm="live"))) is None
        row = log_home()[0]
        assert row["error"] == "empty result"
        assert row["agree"] is None

    @pytest.mark.parametrize(
        "question,answer",
        [
            (Choice("q", "pick", ["yes", "no"]), Answer("q", "maybe", 0.5)),
            (Choice("q", "pick", ["yes", "no"]), Answer("q", "yes", float("inf"))),
            (Noul("q", "true?"), Answer("q", float("nan"), 0.5)),
        ],
    )
    def test_an_out_of_domain_answer_is_recorded_as_an_error(
        self, install_impl, log_home, question, answer
    ):
        class _Invalid:
            async def ask(self, state, questions):
                return OracleResult(answers={"q": answer})

        install_impl(_Invalid())
        result = asyncio.run(decide(POINT, "hi", [question], config=_config(arm="live")))
        assert result is None
        row = log_home()[0]
        assert row["error"] == "invalid result"
        assert row["answers"] is None
        assert row["agree"] is None

    def test_cancellation_propagates_and_writes_no_row(self, install_impl, log_home):
        """A caller going away is not a provider failure."""
        install_impl(_RaisingOracle(asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config()))
        assert log_home() == []

    def test_a_broken_log_does_not_break_the_decision(self, install_impl, monkeypatch):
        """A read-only home must not turn an observation into a failed turn."""
        install_impl(_RecordingOracle())

        def _boom(row):
            raise OSError("read-only file system")

        monkeypatch.setattr(log_mod, "append", _boom)
        # append() swallows its own errors, so patch it to raise and confirm the
        # gate is not relying on that -- decide must still return normally.
        result = asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config(arm="live")))
        assert result is not None

    def test_an_unbuildable_row_does_not_break_the_decision(self, install_impl, log_home):
        """A baseline whose ``__eq__`` raises must not escape ``decide``.

        ``baseline`` is an arbitrary object a point file supplies, and the row
        builder compares it against the answers. That comparison happens OUTSIDE
        ``log.append``'s own guard, so it needs the gate's -- which is the hole
        this file found.
        """
        install_impl(_RecordingOracle("DUP"))

        class _Hostile(dict):
            def __eq__(self, other):
                raise RuntimeError("comparison exploded")

            __hash__ = None  # type: ignore[assignment]

        baseline = _Hostile({"verdict": "DUP"})
        result = asyncio.run(
            decide(POINT, "hi", QUESTIONS, baseline=baseline, config=_config(arm="live"))
        )
        assert result is not None, "an unloggable row must not cost the caller its answer"


# ---------------------------------------------------------------------------
# Arms: what shadow returns vs what live returns
# ---------------------------------------------------------------------------


class TestArms:
    def test_shadow_logs_but_returns_none(self, install_impl, log_home):
        oracle = install_impl(_RecordingOracle("DUP"))
        result = asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config(arm="shadow")))
        assert result is None, "shadow must be byte-identical to off at the call site"
        assert len(oracle.calls) == 1, "but the call and the row must happen"
        row = log_home()[0]
        assert row["arm"] == "shadow"
        assert row["answers"] == {"verdict": {"value": "DUP", "p": 0.87, "confidence": 0.82}}

    def test_live_returns_the_answers(self, install_impl, log_home):
        install_impl(_RecordingOracle("NONE"))
        result = asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config(arm="live")))
        assert result is not None
        assert result["verdict"].value == "NONE"
        assert result["verdict"].p == pytest.approx(0.87)
        assert log_home()[0]["arm"] == "live"

    def test_usage_reaches_the_row(self, install_impl, log_home):
        install_impl(_RecordingOracle(in_tokens=1900))
        asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config()))
        row = log_home()[0]
        assert row["in_tokens"] == 1900
        assert row["cost_usd"] == pytest.approx(1900 * 0.042 / 1e6)
        assert row["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# baseline / agree
# ---------------------------------------------------------------------------


class TestBaselineAgreement:
    def test_matching_baseline_agrees(self, install_impl, log_home):
        install_impl(_RecordingOracle("DUP"))
        asyncio.run(decide(POINT, "hi", QUESTIONS, baseline={"verdict": "DUP"}, config=_config()))
        row = log_home()[0]
        assert row["baseline"] == {"verdict": "DUP"}
        assert row["agree"] is True

    def test_differing_baseline_disagrees(self, install_impl, log_home):
        install_impl(_RecordingOracle("DUP"))
        asyncio.run(decide(POINT, "hi", QUESTIONS, baseline={"verdict": "NONE"}, config=_config()))
        assert log_home()[0]["agree"] is False

    def test_no_baseline_is_null_not_false(self, install_impl, log_home):
        """A missing baseline must not depress the measured agreement rate."""
        install_impl(_RecordingOracle("DUP"))
        asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config()))
        assert log_home()[0]["agree"] is None

    def test_a_baseline_covering_other_ids_is_null(self, install_impl, log_home):
        install_impl(_RecordingOracle("DUP"))
        asyncio.run(
            decide(POINT, "hi", QUESTIONS, baseline={"something_else": "DUP"}, config=_config())
        )
        assert log_home()[0]["agree"] is None


# ---------------------------------------------------------------------------
# The state the implementation is handed
# ---------------------------------------------------------------------------


class TestOffloadedLoggingAndDefaults:
    def test_append_runs_off_the_event_loop(self, install_impl, monkeypatch):
        install_impl(_RecordingOracle())
        caller_thread = threading.get_ident()
        append_threads = []
        monkeypatch.setattr(
            log_mod, "append", lambda row: append_threads.append(threading.get_ident())
        )

        asyncio.run(decide(POINT, "hi", QUESTIONS, config=_config()))

        assert append_threads
        assert append_threads[0] != caller_thread

    def test_missing_impl_uses_llm_in_the_gate_fallback(self, monkeypatch, log_home):
        names = []
        oracle = _RecordingOracle()

        def _resolve(name, provider):
            names.append(name)
            return oracle

        cfg = SimpleNamespace(
            decisions=SimpleNamespace(
                preview=True,
                provider=DecisionProviderConfig(),
                points={POINT: SimpleNamespace(arm="shadow", bucket=100)},
            )
        )
        monkeypatch.setattr(gate_mod, "_resolve_impl", _resolve)

        asyncio.run(decide(POINT, "hi", QUESTIONS, config=cfg))

        assert names == ["llm"]

    def test_config_defaults_to_llm_and_warns_without_rewriting_live(self, caplog):
        assert DecisionPointConfig().impl == "llm"
        parsed = DecisionsConfig.from_raw({"points": {POINT: {"arm": "live", "impl": "unknown"}}})
        assert parsed.points[POINT].impl == "llm"
        assert parsed.points[POINT].arm == "live"
        assert "no decision point consumes live answers" in caplog.text


class TestStateAndQuestionsReachTheImplementation:
    def test_state_and_questions_pass_through_unchanged(self, install_impl):
        oracle = install_impl(_RecordingOracle())
        questions = [
            Choice(id="c", prompt="pick", options=["a", "b"]),
            Noul(id="n", prompt="true?"),
        ]
        state = {"a": 1}
        asyncio.run(decide(POINT, state, questions, config=_config(arm="live")))
        seen_state, seen_questions = oracle.calls[0]
        assert seen_state is state
        assert seen_questions is questions
