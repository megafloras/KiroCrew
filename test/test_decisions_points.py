"""The three DecisionOracle shadow points: shape, silence, and agreement.

These hooks are the only part of the seam that touches production paths, so the
two properties that matter here are the ones a reviewer cannot read off the diff:

1. With the core package ABSENT (or any failure inside the hook), every hook is a
   no-op that returns ``None`` and raises nothing. The call sites are
   fire-and-forget, so a raise would surface as an unretrieved task exception
   rather than as a test failure — it has to be pinned here.
2. With the oracle mocked, ``state``/``questions``/``baseline`` carry exactly the
   documented shape. The core hashes ``session_key`` and logs ``baseline``
   against the answers, so a wrong shape silently produces a meaningless
   agreement rate rather than an error.

``agree`` is tested as a plain function per point because it is the one thing a
generic transport cannot define: ``skills.select`` agrees on set membership,
``skills.dedupe`` on exact string equality, ``cron.novelty`` on a boolean.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import kiro_crew
from kiro_crew.decisions.points import cron_novelty, skills_dedupe, skills_select

# ── stand-ins for the core types (owned by the core package, not by us) ──


class _Choice:
    def __init__(self, id, prompt, options):  # noqa: A002 - mirrors the core signature
        self.id = id
        self.prompt = prompt
        self.options = list(options)


class _Noul:
    def __init__(self, id, prompt):  # noqa: A002 - mirrors the core signature
        self.id = id
        self.prompt = prompt


@pytest.fixture
def fake_core(monkeypatch):
    """Install a minimal ``kiro_crew.decisions`` + ``.types`` for the hook's import.

    The hooks import the core INSIDE the function, so a fake registered in
    ``sys.modules`` is enough — no dependency on worker A's tree.
    """
    core = types.ModuleType("kiro_crew.decisions")
    core.decide = AsyncMock(return_value=None)
    core_types = types.ModuleType("kiro_crew.decisions.types")
    core_types.Choice = _Choice
    core_types.Noul = _Noul
    core_types.Answer = object
    monkeypatch.setitem(sys.modules, "kiro_crew.decisions", core)
    monkeypatch.setitem(sys.modules, "kiro_crew.decisions.types", core_types)
    return core


@pytest.fixture
def spy():
    """The ``_decide`` override every point module exposes for exactly this."""
    return AsyncMock(return_value=None)


@pytest.fixture
def armed_select(monkeypatch):
    """Declare ``skills.select`` armed, so its hook reaches the enumeration.

    The hook asks ``is_armed`` before it walks the skill tree, because doing that
    walk on the default (disabled) configuration was a real cost. A unit test
    about the payload therefore has to state the precondition, or it asserts
    about a code path it never entered.
    """
    monkeypatch.setattr(skills_select, "_is_armed", lambda *a, **k: True)


# ── 1. no core, no crash ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hooks_are_no_ops_when_the_core_has_no_decide(monkeypatch):
    """Every hook returns None and raises nothing when the oracle is not there.

    The absence is FORCED by removing the symbol rather than by relying on the
    core being absent from this tree: once the core package lands, an assumption
    of absence would silently stop asserting anything.
    """
    core = types.ModuleType("kiro_crew.decisions")
    core_types = types.ModuleType("kiro_crew.decisions.types")
    core_types.Choice = _Choice
    core_types.Noul = _Noul
    monkeypatch.setitem(sys.modules, "kiro_crew.decisions", core)
    monkeypatch.setitem(sys.modules, "kiro_crew.decisions.types", core_types)
    monkeypatch.setattr(kiro_crew, "decisions", core, raising=False)
    assert not hasattr(core, "decide")

    assert await skills_select.shadow_skills_select("hi", [], [], session_key="s") is None
    assert await skills_dedupe.shadow_skills_dedupe({}, [], "new", None, session_key="s") is None
    assert await cron_novelty.shadow_cron_novelty("j", "t", "a", "b", session_key="s") is None


@pytest.mark.asyncio
async def test_hooks_are_no_ops_when_the_core_import_fails(monkeypatch):
    """An ImportError reaching out for the core types is absorbed, not raised.

    A ``None`` entry in ``sys.modules`` is the documented way to make an import
    of that name fail, so this holds whether or not the core exists here.
    """
    monkeypatch.setitem(sys.modules, "kiro_crew.decisions.types", None)

    assert await skills_select.shadow_skills_select("hi", [], [], session_key="s") is None
    assert await skills_dedupe.shadow_skills_dedupe({}, [], "new", None, session_key="s") is None
    assert await cron_novelty.shadow_cron_novelty("j", "t", "a", "b", session_key="s") is None


@pytest.mark.asyncio
async def test_hooks_swallow_a_failing_oracle(fake_core, monkeypatch):
    """A raising oracle is absorbed: these run as detached tasks."""
    boom = AsyncMock(side_effect=RuntimeError("transport died"))
    monkeypatch.setattr(skills_select, "_decide", boom)
    monkeypatch.setattr(skills_dedupe, "_decide", boom)
    monkeypatch.setattr(cron_novelty, "_decide", boom)

    assert await skills_select.shadow_skills_select("hi", [], [], session_key="s") is None
    assert await skills_dedupe.shadow_skills_dedupe({}, [], "new", None, session_key="s") is None
    assert await cron_novelty.shadow_cron_novelty("j", "t", "a", "b", session_key="s") is None


@pytest.mark.asyncio
async def test_skills_select_swallows_a_failing_candidate_provider(
    fake_core, spy, armed_select, monkeypatch
):
    """Candidate enumeration runs on the task, so its failure must die there."""
    monkeypatch.setattr(skills_select, "_decide", spy)

    def _explode():
        raise OSError("skills tree unreadable")

    assert await skills_select.shadow_skills_select("hi", _explode, [], session_key="s") is None
    spy.assert_not_awaited()


# ── 2. documented payload shape ──────────────────────────────────────


@pytest.mark.asyncio
async def test_skills_select_payload_shape(fake_core, spy, armed_select, monkeypatch):
    monkeypatch.setattr(skills_select, "_decide", spy)
    candidates = [
        {"key": "brazil", "description": "d" * 500},
        {"key": "crux", "description": "code review"},
    ]

    await skills_select.shadow_skills_select("x" * 5000, candidates, ["crux"], session_key="chat-1")

    point, state, questions = spy.await_args.args
    kwargs = spy.await_args.kwargs
    assert point == "skills.select"
    assert len(state["message"]) == skills_select.MAX_MESSAGE_CHARS
    assert state["candidates"] == [
        {"key": "brazil", "description": "d" * skills_select.MAX_DESCRIPTION_CHARS},
        {"key": "crux", "description": "code review"},
    ]
    assert [q.id for q in questions] == ["pick"]
    assert questions[0].options == ["brazil", "crux", skills_select.NONE_OPTION]
    assert kwargs["baseline"] == {"pick": ["crux"]}
    assert kwargs["session_key"] == "chat-1"


@pytest.mark.asyncio
async def test_skills_select_accepts_a_lazy_candidate_provider(
    fake_core, spy, armed_select, monkeypatch
):
    """The callable candidate walk runs off the event-loop thread."""
    monkeypatch.setattr(skills_select, "_decide", spy)
    caller_thread = threading.get_ident()
    calls = []

    def _provider():
        calls.append(threading.get_ident())
        return [{"key": "tst", "description": "tests"}]

    await skills_select.shadow_skills_select("run tests", _provider, [], session_key="s")

    assert len(calls) == 1
    assert calls[0] != caller_thread
    _, state, questions = spy.await_args.args
    assert state["candidates"] == [{"key": "tst", "description": "tests"}]
    assert questions[0].options == ["tst", skills_select.NONE_OPTION]


@pytest.mark.asyncio
async def test_skills_select_caps_the_candidate_menu(fake_core, spy, armed_select, monkeypatch):
    monkeypatch.setattr(skills_select, "_decide", spy)
    many = [{"key": f"s{i}", "description": ""} for i in range(250)]

    await skills_select.shadow_skills_select("hi", many, [], session_key="s")

    _, state, questions = spy.await_args.args
    assert len(state["candidates"]) == skills_select.MAX_CANDIDATES
    assert len(questions[0].options) == skills_select.MAX_CANDIDATES + 1


@pytest.mark.asyncio
async def test_skills_select_skips_an_unrepresentable_capped_baseline(
    fake_core, spy, armed_select, monkeypatch
):
    monkeypatch.setattr(skills_select, "_decide", spy)
    many = [{"key": f"s{i}", "description": ""} for i in range(250)]

    await skills_select.shadow_skills_select("hi", many, ["s249"], session_key="s")

    spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_skills_dedupe_payload_shape(fake_core, spy, monkeypatch):
    monkeypatch.setattr(skills_dedupe, "_decide", spy)
    candidate = {"key": "auto/new-thing", "description": "does a thing", "triggers": "thing"}
    existing = [
        {"key": "auto/old-thing", "description": "does the same thing", "triggers": "thing"},
        {"key": "auto/other", "description": "unrelated"},
    ]

    await skills_dedupe.shadow_skills_dedupe(
        candidate, existing, "dup", "auto/old-thing", session_key="skill_dedupe:new-thing"
    )

    point, state, questions = spy.await_args.args
    kwargs = spy.await_args.kwargs
    assert point == "skills.dedupe"
    assert state["candidate"] == candidate
    # Only identity + description travel for the existing set: the candidate's
    # triggers are what the verdict is about, an existing skill's are not.
    assert state["existing"] == [
        {"key": "auto/old-thing", "description": "does the same thing"},
        {"key": "auto/other", "description": "unrelated"},
    ]
    assert [q.id for q in questions] == ["verdict"]
    assert questions[0].options == [
        "NONE",
        "DUP:auto/old-thing",
        "DUP:auto/other",
        "UPDATE:auto/old-thing",
        "UPDATE:auto/other",
    ]
    assert kwargs["baseline"] == {"verdict": "DUP:auto/old-thing"}
    # The baseline must be answerable, or every log line reads as a disagreement.
    assert kwargs["baseline"]["verdict"] in questions[0].options


@pytest.mark.asyncio
async def test_skills_dedupe_skips_a_baseline_outside_the_offered_options(
    fake_core, spy, monkeypatch
):
    monkeypatch.setattr(skills_dedupe, "_decide", spy)
    existing = [{"key": f"auto/s{i}", "description": ""} for i in range(101)]

    await skills_dedupe.shadow_skills_dedupe(
        {"key": "auto/new", "description": "new"},
        existing,
        "dup",
        "auto/s100",
        session_key="skill_dedupe:new",
    )

    spy.assert_not_awaited()


@pytest.mark.parametrize(
    "verdict,key,expected",
    [
        ("new", None, "NONE"),
        ("new", "auto/x", "NONE"),
        ("dup", "auto/x", "DUP:auto/x"),
        ("update", "auto/x", "UPDATE:auto/x"),
        # A relation with no key names nothing, so it cannot be an option.
        ("dup", None, "NONE"),
        ("", None, "NONE"),
    ],
)
def test_skills_dedupe_baseline_spelling(verdict, key, expected):
    assert skills_dedupe.baseline_verdict(verdict, key) == expected


@pytest.mark.asyncio
async def test_cron_novelty_payload_shape(fake_core, spy, monkeypatch):
    monkeypatch.setattr(cron_novelty, "_decide", spy)

    await cron_novelty.shadow_cron_novelty(
        "job-7", "PR status", "a" * 9000, "b" * 9000, session_key="cron:job-7"
    )

    point, state, questions = spy.await_args.args
    kwargs = spy.await_args.kwargs
    assert point == "cron.novelty"
    assert state["job_id"] == "job-7"
    assert state["title"] == "PR status"
    assert len(state["last_result"]) == cron_novelty.MAX_RESULT_CHARS
    assert len(state["new_result"]) == cron_novelty.MAX_RESULT_CHARS
    assert [q.id for q in questions] == ["has_new_info"]
    # The hook only runs on the delivering path, so the baseline is fixed.
    assert kwargs["baseline"] == {"delivered": True}
    assert kwargs["session_key"] == "cron:job-7"


# ── 3. agreement, per point ──────────────────────────────────────────


@pytest.mark.parametrize(
    "pick,chosen,expected",
    [
        ("crux", ["brazil", "crux"], True),
        ("tst", ["brazil", "crux"], False),
        (skills_select.NONE_OPTION, [], True),
        (skills_select.NONE_OPTION, ["crux"], False),
        (None, ["crux"], False),
        (None, [], False),
    ],
)
def test_skills_select_agreement_is_set_membership(pick, chosen, expected):
    assert skills_select.agree(pick, chosen) is expected


@pytest.mark.parametrize(
    "oracle,baseline,expected",
    [
        ("DUP:auto/x", "DUP:auto/x", True),
        ("NONE", "NONE", True),
        # Same key, different relation: a disagreement, because the two spell
        # different actions (drop it vs stage an update).
        ("DUP:auto/x", "UPDATE:auto/x", False),
        ("NONE", "DUP:auto/x", False),
        (None, "NONE", False),
    ],
)
def test_skills_dedupe_agreement_is_exact(oracle, baseline, expected):
    assert skills_dedupe.agree(oracle, baseline) is expected


@pytest.mark.parametrize(
    "has_new_info,delivered,expected",
    [
        (True, True, True),
        (False, True, False),
        (0.9, True, True),
        (0, True, False),
        (None, True, False),
    ],
)
def test_cron_novelty_agreement_is_boolean(has_new_info, delivered, expected):
    assert cron_novelty.agree(has_new_info, delivered) is expected


# ── candidate enumeration off a real-shaped loader ───────────────────


class _FakeLoader:
    """Shaped like ``SkillsLoader`` at the two methods the enumerator uses."""

    def __init__(self, rows, meta):
        self._rows = rows
        # Keyed through Path the same way the code under test looks them up:
        # `_has_triggers` calls `_cached_frontmatter(Path(row["path"]))`, and on
        # Windows `str(Path("/s/brazil"))` is a backslash path, so a dict keyed on
        # the raw literal missed every lookup and the enumerator saw no skills.
        self._meta = {str(Path(key)): value for key, value in meta.items()}
        self.seen_project = "unset"

    def list_skills(self, project_dir=None):
        self.seen_project = project_dir
        return self._rows

    def _cached_frontmatter(self, path, within=None):
        return self._meta[str(path)]


def _row(key, path, always=False, description=""):
    return {"key": key, "description": description, "path": path, "always": always}


def test_the_no_selection_sentinel_cannot_collide_with_a_skill_key():
    """A skill named like the sentinel must stay distinguishable from a refusal.

    Keys come from directory names and the ``auto/`` prefix, so a key can hold
    neither a space nor a parenthesis -- which is what makes the sentinel safe
    rather than merely unlikely. A bare ``none`` was not: a skill called ``none``
    made "the oracle picked that skill" and "the oracle picked nothing" the same
    answer, and agreement then scored a real selection as a refusal.
    """
    assert " " in skills_select.NONE_OPTION or "(" in skills_select.NONE_OPTION
    # The collision itself: picking a skill whose key is the old spelling must
    # not read as "nothing applies".
    assert skills_select.agree("none", ["none"]) is True
    assert skills_select.agree(skills_select.NONE_OPTION, ["none"]) is False


def test_a_pathological_skill_key_is_bounded_in_state_and_options():
    """The description was bounded and the key was not."""
    from kiro_crew.decisions.points import MAX_KEY_CHARS

    huge = "k" * (MAX_KEY_CHARS * 4)
    state = skills_select.build_state("hi", [{"key": huge, "description": "d"}])
    assert len(state["candidates"][0]["key"]) == MAX_KEY_CHARS

    dedupe_state = skills_dedupe.build_state(
        {"key": huge, "description": "d", "triggers": "t"}, [{"key": huge, "description": "d"}]
    )
    assert len(dedupe_state["candidate"]["key"]) == MAX_KEY_CHARS
    assert len(dedupe_state["existing"][0]["key"]) == MAX_KEY_CHARS
    # The option strings embed the key, so bounding only the state would leak it.
    assert all(
        len(opt) <= MAX_KEY_CHARS + 16
        for opt in skills_dedupe.verdict_options([{"key": huge, "description": "d"}])
    )


def test_candidates_skip_always_on_and_triggerless_skills():
    loader = _FakeLoader(
        rows=[
            _row("brazil", "/s/brazil", description="build"),
            _row("pinned", "/s/pinned", always=True, description="always"),
            _row("manual", "/s/manual", description="no triggers"),
            _row("", "/s/blank", description="no key"),
        ],
        meta={
            "/s/brazil": {"triggers": "brazil, build"},
            "/s/pinned": {"triggers": "anything"},
            "/s/manual": {"triggers": "   "},
            "/s/blank": {"triggers": "x"},
        },
    )

    rows = skills_select.candidates_from_loader(loader, "/proj")

    # always-on skills are injected unconditionally (never selected), and a
    # triggerless skill can never be the baseline's pick — offering either would
    # manufacture disagreements the baseline had no way to avoid.
    assert rows == [{"key": "brazil", "description": "build"}]
    assert loader.seen_project == "/proj"


def test_candidates_drop_a_row_whose_metadata_cannot_be_read():
    class _BadMeta(_FakeLoader):
        def _cached_frontmatter(self, path, within=None):
            raise OSError("gone")

    loader = _BadMeta(rows=[_row("x", "/s/x")], meta={})

    assert skills_select.candidates_from_loader(loader, None) == []


def test_candidates_are_empty_when_the_listing_itself_fails():
    class _BadList(_FakeLoader):
        def list_skills(self, project_dir=None):
            raise RuntimeError("tree walk failed")

    assert skills_select.candidates_from_loader(_BadList([], {}), None) == []


# ── the same enumeration against the REAL loader ─────────────────────


def _write_skill(root, name, *, triggers="zebra", always=None, description="d"):
    d = root / name
    d.mkdir(parents=True)
    fm = f"---\nname: {name}\ndescription: {description}\n"
    if triggers is not None:
        fm += f"triggers: {triggers}\n"
    if always is not None:
        fm += f"always: {always}\n"
    (d / "SKILL.md").write_text(fm + "---\nbody")


def test_candidates_from_a_real_skills_loader(tmp_path):
    """Pins the enumerator to SkillsLoader's actual shape, not to the fake above.

    ``list_skills`` row keys and the frontmatter reader's signature are both
    read here, so a rename in either one fails this test instead of silently
    emptying the candidate menu at runtime — where the hook swallows everything.
    """
    from kiro_crew.skills import SkillsLoader

    root = tmp_path / "skills"
    _write_skill(root, "matcher", triggers="zebra, giraffe", description="animal work")
    _write_skill(root, "pinned", triggers="zebra", always="true")
    _write_skill(root, "manual", triggers=None, description="no triggers at all")
    loader = SkillsLoader(skills_path=root, install_builtins=False)

    rows = skills_select.candidates_from_loader(loader)

    assert rows == [{"key": "matcher", "description": "animal work"}]
    # The one candidate offered is exactly the one trigger matching CAN pick --
    # asserted with the cap raised, because `skills.max_triggered` defaults to 0,
    # so the stock baseline selects nothing at all and a plain call here would
    # pass while measuring nothing.
    loader._max_triggered = 3
    assert loader.get_triggered_skills("zebra please") == ["matcher"]


@pytest.mark.asyncio
async def test_skills_select_does_not_enumerate_when_the_point_is_off(fake_core, spy, monkeypatch):
    """The candidate walk sits BELOW the arm check, not above it.

    This is the whole point of the check: ``candidates`` reads the skill tree, and
    on the default configuration that ran once per eligible message and was then
    thrown away by a ``decide`` that refuses on its first line.
    """
    monkeypatch.setattr(skills_select, "_decide", spy)
    monkeypatch.setattr(skills_select, "_is_armed", lambda *a, **k: False)
    walked = []

    def _provider():
        walked.append(1)
        return [{"key": "tst", "description": "tests"}]

    assert await skills_select.shadow_skills_select("hi", _provider, [], session_key="s") is None
    assert walked == [], "the skill tree was enumerated for a point that is off"
    spy.assert_not_awaited()


def test_cron_novelty_survives_a_non_string_previous_result():
    """A previous result that is not a string costs the row, never the turn.

    The delivering path hashes the PREVIOUS result to exclude the identical
    24-hour reminder, and `_result_hash` calls `.encode()`, so a non-string
    previous result raises. A shadow observation must never fail the turn it
    observes, which means every call that can raise belongs inside the hook's own
    guard rather than in the condition that reaches it.

    This pins the point-side half: the hook absorbs whatever its inputs do. The
    call-site ordering is pinned by the cron tests in test_slack_gateway.py, which
    fail if the hash call moves into the guard condition.
    """
    import asyncio
    from unittest.mock import MagicMock

    # Every shape the call site could hand over, including the one that broke it.
    for previous in (MagicMock(), None, 12345, object()):
        result = asyncio.run(
            cron_novelty.shadow_cron_novelty(
                "job-1", "Daily", previous, "new text", session_key="s"
            )
        )
        assert result is None, f"a {type(previous).__name__} previous result must be absorbed"
