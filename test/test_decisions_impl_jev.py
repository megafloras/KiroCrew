"""``impl_jev`` against a real aiohttp server: 200, 429, 5xx, timeout, bad JSON.

A live loopback server rather than a mocked ``ClientSession``: the thing most
likely to be wrong here is the WIRE, and a mock asserts only that the code calls
the mock. The server below records the request body, so the request-shape
assertions are made against what actually went over a socket.

Every field name asserted here is quoted from ``https://docs.typesafe.ai/api``.
These tests are the record of what that page specifies, and
``_to_wire``/``_from_wire`` are the only two functions to edit when it changes.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from kiro_crew.config.sections import DecisionProviderConfig
from kiro_crew.decisions.impl_jev import (
    JevHttpError,
    JevOracle,
    JevProtocolError,
    resolve_api_key,
)
from kiro_crew.decisions.types import Choice, Noul

CHOICE = Choice(id="department", prompt="Which team?", options=["billing", "technical", "sales"])
NOUL = Noul(id="is_urgent", prompt="Does this convey urgency?")


class _Recorder:
    """A tiny aiohttp app that answers with a canned response and records the request."""

    def __init__(self, *, status=200, body=None, raw=None, delay=0.0):
        self.status = status
        self.body = body
        self.raw = raw
        self.delay = delay
        self.requests: list[dict] = []
        self.headers: list[dict] = []

    async def handle(self, request: web.Request) -> web.Response:
        self.headers.append(dict(request.headers))
        try:
            self.requests.append(await request.json())
        except Exception:
            self.requests.append({})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raw is not None:
            return web.Response(status=self.status, text=self.raw, content_type="application/json")
        return web.json_response(self.body or {}, status=self.status)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/systemone", self.handle)
        return app


async def _run(recorder: _Recorder, questions, *, timeout_ms=5000, api_key="test-key", state="hi"):
    """Serve *recorder* on loopback and ask *questions* through a real socket."""
    server = TestServer(recorder.app())
    await server.start_server()
    try:
        provider = DecisionProviderConfig(
            endpoint=str(server.make_url("/v1/systemone")),
            api_key=api_key,
            model="jev-latest",
            timeout_ms=timeout_ms,
        )
        return await JevOracle(provider).ask(state, questions)
    finally:
        await server.close()


def _ok_body(answers, *, input_tokens=312):
    return {
        "model": "jev-latest",
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 48},
    }


# ---------------------------------------------------------------------------
# The request that goes over the wire
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_the_bearer_header_carries_the_key(self):
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.9}}))
        asyncio.run(_run(rec, [NOUL], api_key="sk-live-abc"))
        assert rec.headers[0]["Authorization"] == "Bearer sk-live-abc"

    def test_state_and_model_are_top_level(self):
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.9}}))
        asyncio.run(_run(rec, [NOUL], state={"messages": ["hello"]}))
        sent = rec.requests[0]
        assert sent["model"] == "jev-latest"
        assert sent["state"] == {"messages": ["hello"]}

    def test_choice_criteria_is_a_map_of_option_to_null(self):
        """Per the API a Choice's ``criteria`` is a MAP; a list would be rejected."""
        rec = _Recorder(
            body=_ok_body(
                {
                    "department": {
                        "type": "choice",
                        "choice": "sales",
                        "probabilities": {"sales": 1.0},
                        "confidence": 0.9,
                    }
                }
            )
        )
        asyncio.run(_run(rec, [CHOICE]))
        q = rec.requests[0]["questions"]["department"]
        assert q["type"] == "choice"
        assert q["instructions"] == CHOICE.prompt
        assert q["criteria"] == {"billing": None, "technical": None, "sales": None}

    def test_noul_sends_type_and_instructions_only(self):
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.5}}))
        asyncio.run(_run(rec, [NOUL]))
        assert rec.requests[0]["questions"]["is_urgent"] == {
            "type": "noul",
            "instructions": NOUL.prompt,
        }


# ---------------------------------------------------------------------------
# 200: parsing each answer type back into the question's own units
# ---------------------------------------------------------------------------


class TestSuccessfulParse:
    def test_choice_answer(self):
        rec = _Recorder(
            body=_ok_body(
                {
                    "department": {
                        "type": "choice",
                        "choice": "technical",
                        "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
                        "confidence": 0.82,
                    }
                }
            )
        )
        result = asyncio.run(_run(rec, [CHOICE]))
        answer = result.answers["department"]
        assert answer.value == "technical"
        assert answer.p == pytest.approx(0.85), "p is the mass behind the chosen option"
        assert answer.confidence == pytest.approx(0.82)

    def test_noul_answer_puts_the_probability_in_both_value_and_p(self):
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.92}}))
        answer = asyncio.run(_run(rec, [NOUL])).answers["is_urgent"]
        assert answer.value == pytest.approx(0.92)
        assert answer.p == pytest.approx(0.92)
        assert answer.confidence is None, "the API reports no confidence for a Noul"

    def test_usage_becomes_in_tokens_and_cost(self):
        rec = _Recorder(
            body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.1}}, input_tokens=1900)
        )
        result = asyncio.run(_run(rec, [NOUL]))
        assert result.in_tokens == 1900
        assert result.cost_usd == pytest.approx(1900 * 0.042 / 1e6)

    def test_missing_usage_reads_as_not_reported_rather_than_failing(self):
        rec = _Recorder(body={"model": "jev-latest", "answers": {"is_urgent": {"noul": 0.1}}})
        result = asyncio.run(_run(rec, [NOUL]))
        assert result.in_tokens == 0
        assert result.cost_usd == 0.0

    def test_a_2xx_other_than_200_is_accepted(self):
        rec = _Recorder(status=202, body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.3}}))
        assert asyncio.run(_run(rec, [NOUL])).answers["is_urgent"].value == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Failures: each raises, so the gate can convert it into None + a logged reason
# ---------------------------------------------------------------------------


class TestFailures:
    @pytest.mark.parametrize("status", [429, 500, 502, 529, 401, 422])
    def test_a_non_2xx_raises_with_its_status(self, status):
        rec = _Recorder(status=status, body={"error": "nope"})
        with pytest.raises(JevHttpError) as caught:
            asyncio.run(_run(rec, [NOUL]))
        assert caught.value.status == status
        assert str(status) in str(caught.value)

    def test_the_error_body_reaches_the_message(self):
        """A 422 names the offending field; that text is the whole value of the row."""
        rec = _Recorder(status=422, raw='{"detail":"questions.is_urgent.criteria invalid"}')
        with pytest.raises(JevHttpError) as caught:
            asyncio.run(_run(rec, [NOUL]))
        assert "criteria invalid" in str(caught.value)

    def test_a_body_that_is_not_json_raises(self):
        rec = _Recorder(raw="<html>502 Bad Gateway</html>")
        with pytest.raises(JevProtocolError, match="not JSON"):
            asyncio.run(_run(rec, [NOUL]))

    def test_a_body_with_no_answers_object_raises(self):
        rec = _Recorder(body={"model": "jev-latest", "usage": {"input_tokens": 1}})
        with pytest.raises(JevProtocolError, match="no 'answers' object"):
            asyncio.run(_run(rec, [NOUL]))

    def test_a_json_array_body_raises(self):
        rec = _Recorder(raw="[1, 2, 3]")
        with pytest.raises(JevProtocolError, match="not an object"):
            asyncio.run(_run(rec, [NOUL]))

    def test_an_unanswered_question_raises_rather_than_returning_a_partial(self):
        """All-or-nothing: a point file has no way to ask which half it got."""
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.5}}))
        with pytest.raises(JevProtocolError, match="no answer for question 'department'"):
            asyncio.run(_run(rec, [NOUL, CHOICE]))

    def test_a_choice_answer_without_a_choice_string_raises(self):
        rec = _Recorder(body=_ok_body({"department": {"type": "choice", "probabilities": {}}}))
        with pytest.raises(JevProtocolError, match="no 'choice' string"):
            asyncio.run(_run(rec, [CHOICE]))

    def test_a_boolean_is_not_accepted_as_a_number(self):
        """``isinstance(True, int)`` is True in Python; a bool score is not a score."""
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": True}}))
        with pytest.raises(JevProtocolError, match="no numeric 'noul'"):
            asyncio.run(_run(rec, [NOUL]))

    def test_a_slow_server_times_out(self):
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.1}}), delay=2.0)
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            asyncio.run(_run(rec, [NOUL], timeout_ms=100))

    def test_no_api_key_raises_before_any_request(self):
        """An empty bearer would come back 401 and be indistinguishable from a bad key."""
        rec = _Recorder(body=_ok_body({"is_urgent": {"type": "noul", "noul": 0.1}}))
        with pytest.raises(JevProtocolError, match="no api key"):
            asyncio.run(_run(rec, [NOUL], api_key=""))
        assert rec.requests == [], "nothing may reach the network without a key"

    def test_no_questions_raises(self):
        rec = _Recorder(body=_ok_body({}))
        with pytest.raises(JevProtocolError, match="no questions"):
            asyncio.run(_run(rec, []))


# ---------------------------------------------------------------------------
# api_key resolution
# ---------------------------------------------------------------------------


class TestResponseBound:
    """A provider cannot make the reader allocate an unbounded body."""

    def test_an_oversized_response_is_refused_not_allocated(self):
        from kiro_crew.decisions.impl_jev import _MAX_RESPONSE_BYTES

        # One byte past the cap, and valid JSON so nothing else can be blamed:
        # the padding sits inside a real field, so a reader without the cap would
        # parse this happily and only the allocation would be unbounded.
        padding = "x" * (_MAX_RESPONSE_BYTES + 1)
        recorder = _Recorder(raw='{"model": "jev-latest", "answers": {}, "pad": "' + padding + '"}')

        with pytest.raises(JevProtocolError) as caught:
            asyncio.run(_run(recorder, [Choice(id="verdict", prompt="?", options=["A", "B"])]))

        assert "exceeded" in str(caught.value)

    def test_a_normal_response_is_unaffected_by_the_bound(self):
        """The cap must be generous next to a real answer set, not near it."""
        recorder = _Recorder(body=_ok_body({"verdict": {"choice": "A", "p": 0.9}}))
        result = asyncio.run(_run(recorder, [Choice(id="verdict", prompt="?", options=["A", "B"])]))
        assert result.answers["verdict"].value == "A"


class TestResolveApiKey:
    def test_a_literal_value_passes_through(self):
        assert resolve_api_key("sk-plain-value") == "sk-plain-value"

    def test_whitespace_is_stripped(self):
        assert resolve_api_key("  sk-plain  ") == "sk-plain"

    @pytest.mark.parametrize("raw", ["", "   ", "secret://", "secret://   "])
    def test_nothing_usable_resolves_to_empty(self, raw):
        assert resolve_api_key(raw) == ""

    def test_a_secret_reference_reads_the_vault(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        import kiro_crew.secrets.vault as vault_mod

        class _Value:
            def reveal(self):
                return "sk-from-vault"

        class _Vault:
            def __init__(self, *_a, **_k):
                pass

            def get(self, name):
                assert name == "TYPESAFE_API_KEY"
                return _Value()

        monkeypatch.setattr(vault_mod, "SecretVault", _Vault)
        assert resolve_api_key("secret://TYPESAFE_API_KEY") == "sk-from-vault"

    def test_a_vault_reference_is_not_resolved_for_a_custom_endpoint(self, monkeypatch):
        """A custom endpoint must never reach the vault, not merely fail to use it.

        The vault is observed by whether it is CONSTRUCTED, not by making it
        raise: ``resolve_api_key`` catches ``Exception`` around the vault call
        (a vault error is contractually "no usable key"), and ``AssertionError``
        is an ``Exception`` -- so a raising double returns ``""`` whether the
        guard is present or not, and the test could never fail. A recorded
        construction is not swallowable.
        """
        import kiro_crew.secrets.vault as vault_mod

        constructed: list[int] = []

        class _Vault:
            def __init__(self, *_a, **_k):
                constructed.append(1)

            def get(self, name):  # pragma: no cover - reached only on regression
                return None

        monkeypatch.setattr(vault_mod, "SecretVault", _Vault)
        custom = "https://judge.example.invalid/v1/decide"

        assert resolve_api_key("secret://TYPESAFE_API_KEY", endpoint=custom) == ""
        assert constructed == [], "a custom endpoint must not read the vault"

        # A literal key is the operator's own config, already readable by anyone
        # who can set the endpoint, so it stays allowed and still touches nothing.
        assert resolve_api_key("literal-operator-key", endpoint=custom) == "literal-operator-key"
        assert constructed == []

    def test_a_missing_vault_entry_resolves_to_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        import kiro_crew.secrets.vault as vault_mod

        class _Vault:
            def __init__(self, *_a, **_k):
                pass

            def get(self, name):
                return None

        monkeypatch.setattr(vault_mod, "SecretVault", _Vault)
        assert resolve_api_key("secret://ABSENT") == ""

    def test_a_vault_error_resolves_to_empty_rather_than_raising(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        import kiro_crew.secrets.vault as vault_mod

        class _Vault:
            def __init__(self, *_a, **_k):
                raise OSError("vault key unreadable")

        monkeypatch.setattr(vault_mod, "SecretVault", _Vault)
        assert resolve_api_key("secret://ANY") == ""


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestProviderDefaults:
    def test_an_empty_provider_falls_back_to_the_documented_endpoint_and_model(self):
        class _Bare:
            endpoint = ""
            model = ""
            api_key = "k"
            timeout_ms = 1000

        oracle = JevOracle(_Bare())
        assert oracle._endpoint == "https://api.typesafe.ai/v1/systemone"
        assert oracle._model == "jev-latest"

    def test_the_shipped_config_default_matches_the_documented_endpoint(self):
        """A drift guard: the config default and this module must not diverge."""
        assert DecisionProviderConfig.endpoint == "https://api.typesafe.ai/v1/systemone"
        assert DecisionProviderConfig.model == "jev-latest"

    def test_request_body_is_json_serialisable_as_sent(self):
        """Pins that nothing in ``_to_wire`` needs a custom encoder."""
        from kiro_crew.decisions.impl_jev import _to_wire

        json.dumps(_to_wire("hi", "jev-latest", [CHOICE, NOUL]))
