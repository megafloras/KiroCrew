"""``impl_llm``: the sessions registry, the prompt, and strict JSON parsing.

``run_bg_oneliner`` is substituted rather than driven: it needs a real ACP
runtime, and what this file is about is the prompt going in and the reply coming
out. The one thing NOT substituted is the registry -- ``set_bg_sessions`` /
``bg_sessions`` are exercised for real, including the weak-reference behaviour,
because that is the wiring most likely to be silently absent in production.
"""

from __future__ import annotations

import asyncio
import gc
import json

import pytest

from kiro_crew.decisions import impl_llm
from kiro_crew.decisions.impl_llm import (
    LlmOracle,
    bg_sessions,
    build_prompt,
    parse_reply,
    set_bg_sessions,
)
from kiro_crew.decisions.types import Choice, Noul

CHOICE = Choice(id="verdict", prompt="Is this a duplicate?", options=["NONE", "DUP", "UPDATE"])
NOUL = Noul(id="novel", prompt="Is there new information?")


@pytest.fixture(autouse=True)
def clean_registry():
    """Leave the module-level registry as it was found."""
    yield
    set_bg_sessions(None)


class _FakeSessions:
    """Stands in for a SessionManager. Weak-referenceable."""


@pytest.fixture
def registered_sessions():
    """A registered manager the TEST keeps a strong reference to.

    Registering ``_FakeSessions()`` inline does not work, and that is the weak
    reference doing its job: the temporary has no other referent, so it is
    collected before ``ask`` runs and the lane correctly reads as unset. Any test
    that needs the lane live must hold the object, exactly as the dashboard does.
    """
    sessions = _FakeSessions()
    set_bg_sessions(sessions)
    return sessions


@pytest.fixture
def fake_oneliner(monkeypatch):
    """Substitute ``run_bg_oneliner`` and record how it was called."""
    calls: list[dict] = []

    def _install(reply: str | BaseException):
        async def _fake(sessions, prompt, **kwargs):
            calls.append({"sessions": sessions, "prompt": prompt, **kwargs})
            if isinstance(reply, BaseException):
                raise reply
            return reply

        import kiro_crew.llm_helpers as helpers

        monkeypatch.setattr(helpers, "run_bg_oneliner", _fake)
        return calls

    return _install


# ---------------------------------------------------------------------------
# The sessions registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_unset_means_no_lane(self):
        assert bg_sessions() is None

    def test_set_then_read(self):
        sessions = _FakeSessions()
        set_bg_sessions(sessions)
        assert bg_sessions() is sessions

    def test_none_clears(self):
        set_bg_sessions(_FakeSessions())
        set_bg_sessions(None)
        assert bg_sessions() is None

    def test_the_reference_is_weak(self):
        """Holding it here must not keep a torn-down SessionManager alive."""
        sessions = _FakeSessions()
        set_bg_sessions(sessions)
        del sessions
        gc.collect()
        assert bg_sessions() is None, "a collected manager must read as unset"

    def test_an_unweakrefable_object_is_still_registered(self):
        """A test double with __slots__ must not silently disable the lane."""

        class _NoWeakref:
            __slots__ = ()

        sessions = _NoWeakref()
        set_bg_sessions(sessions)
        assert bg_sessions() is sessions

    def test_ask_without_a_registry_raises_with_a_readable_reason(self):
        oracle = LlmOracle()
        with pytest.raises(RuntimeError, match="no background sessions registered"):
            asyncio.run(oracle.ask("hi", [NOUL]))

    def test_the_dashboard_registers_the_manager_at_construction(self):
        """The one edit outside this package: DashboardState.__init__ wires the lane.

        Asserted through the real ``__init__`` (via ``__new__`` + a direct call
        would still run the body) so that removing the registration breaks this
        test rather than only breaking production.
        """
        from kiro_crew.dashboard.state import DashboardState

        sessions = _FakeSessions()
        state = DashboardState.__new__(DashboardState)
        DashboardState.__init__(
            state,
            sessions=sessions,  # type: ignore[arg-type]
            crons=None,  # type: ignore[arg-type]
            lessons=None,  # type: ignore[arg-type]
            start_time=0.0,
        )
        assert bg_sessions() is sessions


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_every_question_appears_once(self):
        prompt = build_prompt("some state", [CHOICE, NOUL])
        for q in (CHOICE, NOUL):
            assert prompt.count(f"- {q.id}:") == 1
            assert q.prompt in prompt

    def test_choice_options_are_listed(self):
        prompt = build_prompt("s", [CHOICE])
        assert "NONE, DUP, UPDATE" in prompt

    def test_the_state_is_included_verbatim_for_a_string(self):
        assert "the user said hello" in build_prompt("the user said hello", [NOUL])

    def test_a_dict_state_is_json_rendered(self):
        prompt = build_prompt({"messages": ["hi"]}, [NOUL])
        assert '{"messages": ["hi"]}' in prompt

    def test_it_asks_for_one_json_object_and_no_fence(self):
        prompt = build_prompt("s", [NOUL])
        assert "ONE JSON object" in prompt
        assert "no code fence" in prompt

    def test_one_prompt_covers_all_questions(self, fake_oneliner, registered_sessions):
        """Same shape as the Jev lane: one turn, so latency describes one unit."""
        calls = fake_oneliner(
            json.dumps(
                {
                    "verdict": {"value": "DUP", "p": 0.7},
                    "novel": {"value": 0.2, "p": 0.2},
                }
            )
        )
        asyncio.run(LlmOracle().ask("s", [CHOICE, NOUL]))
        assert len(calls) == 1

    def test_denials_are_attributed_to_this_feature(self, fake_oneliner, registered_sessions):
        calls = fake_oneliner(json.dumps({"novel": {"value": 0.5, "p": 0.5}}))
        asyncio.run(LlmOracle().ask("s", [NOUL]))
        assert calls[0]["sel_source"] == "decisions"

    def test_the_gates_budget_is_passed_to_the_background_turn(
        self, fake_oneliner, registered_sessions
    ):
        """So a background turn cannot outlive the decision that asked for it."""
        calls = fake_oneliner(json.dumps({"novel": {"value": 0.5, "p": 0.5}}))

        class _Provider:
            timeout_ms = 250

        asyncio.run(LlmOracle(_Provider()).ask("s", [NOUL]))
        assert calls[0]["timeout"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParse:
    def test_a_clean_object_parses(self):
        answers = parse_reply(json.dumps({"verdict": {"value": "DUP", "p": 0.71}}), [CHOICE])
        assert answers["verdict"].value == "DUP"
        assert answers["verdict"].p == pytest.approx(0.71)
        assert answers["verdict"].confidence is None, "a chat model's certainty is not calibrated"

    def test_a_markdown_fence_is_tolerated(self):
        raw = '```json\n{"verdict": {"value": "NONE", "p": 0.9}}\n```'
        assert parse_reply(raw, [CHOICE])["verdict"].value == "NONE"

    def test_a_bare_fence_is_tolerated(self):
        raw = '```\n{"verdict": {"value": "NONE", "p": 0.9}}\n```'
        assert parse_reply(raw, [CHOICE])["verdict"].value == "NONE"

    def test_prose_around_one_object_is_recovered(self):
        raw = 'Sure! Here you go: {"verdict": {"value": "UPDATE", "p": 0.5}} Hope that helps.'
        assert parse_reply(raw, [CHOICE])["verdict"].value == "UPDATE"

    def test_a_noul_is_clamped_to_zero_one(self):
        assert parse_reply(json.dumps({"novel": {"value": 1.7}}), [NOUL])[
            "novel"
        ].value == pytest.approx(1.0)

    def test_a_noul_with_no_p_mirrors_its_own_value(self):
        answer = parse_reply(json.dumps({"novel": {"value": 0.3}}), [NOUL])["novel"]
        assert answer.p == pytest.approx(0.3)

    def test_a_noul_preserves_an_explicit_zero_p(self):
        answer = parse_reply(json.dumps({"novel": {"value": 1.0, "p": 0.0}}), [NOUL])["novel"]
        assert answer.p == 0.0

    def test_a_numeric_string_value_is_accepted(self):
        assert parse_reply(json.dumps({"novel": {"value": "0.4"}}), [NOUL])[
            "novel"
        ].value == pytest.approx(0.4)

    def test_a_missing_p_reads_as_not_reported(self):
        assert parse_reply(json.dumps({"verdict": {"value": "DUP"}}), [CHOICE])["verdict"].p == 0.0

    @pytest.mark.parametrize(
        "raw,questions",
        [
            ("", [CHOICE]),
            ("   ", [CHOICE]),
            ("I cannot answer that.", [CHOICE]),
            ("[1, 2, 3]", [CHOICE]),
            ('"a string"', [CHOICE]),
            # Question absent from the reply.
            ('{"other": {"value": "DUP"}}', [CHOICE]),
            # Answer is not an object.
            ('{"verdict": "DUP"}', [CHOICE]),
            # Choice value is not a string.
            ('{"verdict": {"value": 3}}', [CHOICE]),
            # Noul value is not numeric.
            ('{"novel": {"value": null}}', [NOUL]),
            ('{"novel": {"value": true}}', [NOUL]),
        ],
    )
    def test_anything_else_raises(self, raw, questions):
        """A default would be a fabricated verdict; an error is visible in the report."""
        with pytest.raises(ValueError):
            parse_reply(raw, questions)

    def test_a_choice_outside_its_options_raises_rather_than_snapping(self):
        with pytest.raises(ValueError, match="is not one of"):
            parse_reply(json.dumps({"verdict": {"value": "MAYBE", "p": 0.9}}), [CHOICE])

    def test_a_partial_reply_raises_rather_than_returning_half(self):
        raw = json.dumps({"verdict": {"value": "DUP", "p": 0.9}})
        with pytest.raises(ValueError, match="no answer for 'novel'"):
            parse_reply(raw, [CHOICE, NOUL])


# ---------------------------------------------------------------------------
# ask()
# ---------------------------------------------------------------------------


class TestAsk:
    def test_a_good_reply_becomes_answers_with_no_usage_reported(
        self, fake_oneliner, registered_sessions
    ):
        fake_oneliner(json.dumps({"verdict": {"value": "DUP", "p": 0.8}}))
        result = asyncio.run(LlmOracle().ask("s", [CHOICE]))
        assert result.answers["verdict"].value == "DUP"
        # 0/0.0, not a guess from prompt length: a guessed number in the same
        # column as Jev's billed count would make the two silently incomparable.
        assert result.in_tokens == 0
        assert result.cost_usd == 0.0

    def test_a_helper_failure_propagates_for_the_gate_to_log(
        self, fake_oneliner, registered_sessions
    ):
        fake_oneliner(RuntimeError("acp runtime dead"))
        with pytest.raises(RuntimeError, match="acp runtime dead"):
            asyncio.run(LlmOracle().ask("s", [CHOICE]))

    def test_an_unparseable_reply_propagates(self, fake_oneliner, registered_sessions):
        fake_oneliner("I'd rather not.")
        with pytest.raises(ValueError):
            asyncio.run(LlmOracle().ask("s", [CHOICE]))

    def test_no_questions_raises(self, registered_sessions):
        with pytest.raises(ValueError, match="no questions"):
            asyncio.run(LlmOracle().ask("s", []))

    def test_the_registered_manager_is_the_one_used(self, fake_oneliner, registered_sessions):
        calls = fake_oneliner(json.dumps({"novel": {"value": 0.1}}))
        asyncio.run(LlmOracle().ask("s", [NOUL]))
        assert calls[0]["sessions"] is registered_sessions


# ---------------------------------------------------------------------------
# Reachability from the gate
# ---------------------------------------------------------------------------


class TestGateResolvesThisImplementation:
    def test_impl_llm_is_reachable_by_name(self):
        from kiro_crew.decisions.gate import _resolve_impl

        assert isinstance(_resolve_impl("llm", None), LlmOracle)

    def test_impl_jev_is_reachable_by_name(self):
        from kiro_crew.decisions.gate import _resolve_impl
        from kiro_crew.decisions.impl_jev import JevOracle

        assert isinstance(_resolve_impl("jev", None), JevOracle)

    def test_an_unknown_name_raises(self):
        from kiro_crew.decisions.gate import _resolve_impl

        with pytest.raises(ValueError, match="unknown decisions impl"):
            _resolve_impl("something", None)

    def test_the_package_does_not_import_from_points(self):
        """Points depend on the seam, never the reverse -- the dependency stays acyclic."""
        import ast
        import pathlib

        pkg = pathlib.Path(impl_llm.__file__).parent
        offenders: list[str] = []
        for path in sorted(pkg.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "kiro_crew.decisions.points"
                ):
                    offenders.append(path.name)
                elif isinstance(node, ast.Import) and any(
                    a.name.startswith("kiro_crew.decisions.points") for a in node.names
                ):
                    offenders.append(path.name)
        assert offenders == [], offenders
