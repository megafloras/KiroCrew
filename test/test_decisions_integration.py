"""The three point hooks, end to end into the real gate, log and agreement rule.

Every other decisions test exercises one half. Worker A built the gate against a
stand-in point, worker B built the points against a stand-in ``decide``, and
worker C built the report against a stand-in row. This file is the only place
where a POINT hook calls the REAL ``decide``, the real gate reads a config the
way production does (off the live snapshot, with no ``config=`` injection the
points are forbidden to pass), the real log writes a real file, and the real
report reads it back.

What it is here to catch
------------------------
Two things the halves cannot see on their own:

1. ``agree``. The gate's generic rule is value equality over a shared question-id
   set. Two of the three points cannot be judged that way -- ``skills.select``
   asks one skill against a baseline LIST and adds a second question, and
   ``cron.novelty`` answers with a probability against a boolean -- so before the
   points handed in their own rules both logged ``agree: null`` on every row and
   dropped out of the one rate this log exists to produce. Each case below
   asserts a real ``True``/``False``, never merely a present key.
2. That the hooks reach ``decide`` at all. The hooks resolve it with
   ``getattr(kiro_crew.decisions, "decide", None)`` so they degrade on a tree
   without the core; that same indirection means a rename in the core would
   degrade them SILENTLY to no-ops in production, with every test still green.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kiro_crew import credential_patterns as _cred
from kiro_crew.config.sections import (
    DecisionPointConfig,
    DecisionProviderConfig,
    DecisionsConfig,
)
from kiro_crew.decisions import gate as gate_mod
from kiro_crew.decisions import log as log_mod
from kiro_crew.decisions.oracle import OracleResult
from kiro_crew.decisions.points import cron_novelty, skills_dedupe, skills_select
from kiro_crew.decisions.types import Answer

IMPL_NAME = "fake"


def _snapshot_with(point: str, *, preview: bool = True, arm: str = "shadow", bucket: int = 100):
    """A stand-in for what ``config.live.snapshot()`` hands the gate."""

    class _Cfg:
        decisions = DecisionsConfig(
            preview=preview,
            provider=DecisionProviderConfig(timeout_ms=1000),
            points={point: DecisionPointConfig(arm=arm, impl=IMPL_NAME, bucket=bucket)},
        )

    return _Cfg()


class _FixedOracle:
    """Answers each question id from a supplied map; records what it was sent."""

    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.states: list[object] = []

    async def ask(self, state, questions):
        self.states.append(state)
        return OracleResult(
            answers={
                q.id: Answer(id=q.id, value=self.values[q.id], p=0.9, confidence=None)
                for q in questions
                if q.id in self.values
            },
            in_tokens=1200,
            cost_usd=0.00005,
        )


@pytest.fixture
def armed(tmp_path, monkeypatch):
    """Arm one point on a fake implementation and a temp log; yield the rows reader.

    The config arrives through ``_snapshot`` rather than through ``decide``'s
    ``config=`` kwarg on purpose: the points are contractually forbidden to pass
    that kwarg, so injecting it here would test a path production never takes.
    """
    log_dir = tmp_path / "decisions"
    monkeypatch.setattr(log_mod, "log_dir", lambda: log_dir)

    def _arm(point: str, oracle: _FixedOracle, **cfg):
        monkeypatch.setattr(gate_mod, "_snapshot", lambda: _snapshot_with(point, **cfg))
        real = gate_mod._resolve_impl
        monkeypatch.setattr(
            gate_mod,
            "_resolve_impl",
            lambda name, provider: oracle if name == IMPL_NAME else real(name, provider),
        )
        return oracle

    def _rows():
        if not log_dir.exists():
            return []
        out = []
        for path in sorted(log_dir.glob("decisions-*.jsonl")):
            out.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        return out

    _arm.rows = _rows  # type: ignore[attr-defined]
    _arm.dir = log_dir  # type: ignore[attr-defined]
    return _arm


# ---------------------------------------------------------------------------
# skills.select
# ---------------------------------------------------------------------------


class TestSkillsSelect:
    def test_the_hook_writes_one_row_with_a_baseline_and_a_real_agreement(self, armed):
        oracle = armed(
            skills_select.POINT,
            _FixedOracle({"pick": "alpha"}),
        )

        result = asyncio.run(
            skills_select.shadow_skills_select(
                "please run alpha",
                [{"key": "alpha", "description": "does alpha"}],
                ["alpha"],
                session_key="s-1",
            )
        )

        assert result is None, "shadow must return None however well the call went"
        rows = armed.rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["point"] == "skills.select"
        assert row["arm"] == "shadow"
        assert row["impl"] == IMPL_NAME
        assert row["baseline"] == {"pick": ["alpha"]}
        assert row["agree"] is True, (
            "set membership, not value equality: the answer is one key and the "
            "baseline is a list, so the generic rule would log null here"
        )
        assert row["error"] is None
        assert row["scrubbed"] is False
        assert set(row["answers"]) == {"pick"}, (
            "one question only: the report buckets a row by the highest p across "
            "its answers, so an ungraded second answer would steer calibration"
        )
        assert oracle.states[0]["message"] == "please run alpha"

    def test_a_pick_outside_the_baseline_disagrees(self, armed):
        armed(skills_select.POINT, _FixedOracle({"pick": "beta"}))
        asyncio.run(
            skills_select.shadow_skills_select(
                "hello",
                [{"key": "alpha", "description": "a"}, {"key": "beta", "description": "b"}],
                ["alpha"],
                session_key="s-2",
            )
        )
        assert armed.rows()[0]["agree"] is False

    def test_the_candidate_callable_runs_on_this_task_not_the_call_site(self, armed):
        """The sync call site passes a callable so the skill walk is off its path."""
        armed(skills_select.POINT, _FixedOracle({"pick": "alpha"}))
        calls = []

        def _candidates():
            calls.append(1)
            return [{"key": "alpha", "description": "a"}]

        asyncio.run(
            skills_select.shadow_skills_select("hi", _candidates, ["alpha"], session_key="s-3")
        )
        assert calls == [1]
        assert armed.rows()[0]["agree"] is True


# ---------------------------------------------------------------------------
# skills.dedupe
# ---------------------------------------------------------------------------


class TestSkillsDedupe:
    def test_matching_verdicts_agree(self, armed):
        armed(skills_dedupe.POINT, _FixedOracle({"verdict": "DUP:auto/x"}))
        asyncio.run(
            skills_dedupe.shadow_skills_dedupe(
                {"key": "auto/y", "description": "d", "triggers": "t"},
                [{"key": "auto/x", "description": "d"}],
                "dup",
                "auto/x",
                session_key="s-4",
            )
        )
        row = armed.rows()[0]
        assert row["point"] == "skills.dedupe"
        assert row["baseline"] == {"verdict": "DUP:auto/x"}
        assert row["agree"] is True

    def test_the_same_key_under_a_different_relation_disagrees(self, armed):
        armed(skills_dedupe.POINT, _FixedOracle({"verdict": "UPDATE:auto/x"}))
        asyncio.run(
            skills_dedupe.shadow_skills_dedupe(
                {"key": "auto/y", "description": "d", "triggers": "t"},
                [{"key": "auto/x", "description": "d"}],
                "dup",
                "auto/x",
                session_key="s-5",
            )
        )
        assert armed.rows()[0]["agree"] is False


# ---------------------------------------------------------------------------
# cron.novelty
# ---------------------------------------------------------------------------


class TestCronNovelty:
    def test_a_confident_yes_agrees_with_the_delivery_that_happened(self, armed):
        armed(cron_novelty.POINT, _FixedOracle({"has_new_info": 0.95}))
        asyncio.run(
            cron_novelty.shadow_cron_novelty(
                "job-1", "Daily", "old text", "new text", session_key="s-6"
            )
        )
        row = armed.rows()[0]
        assert row["point"] == "cron.novelty"
        assert row["baseline"] == {"delivered": True}
        assert row["agree"] is True, "a probability against a boolean needs the point's threshold"

    def test_a_confident_no_disagrees_with_the_delivery_that_happened(self, armed):
        armed(cron_novelty.POINT, _FixedOracle({"has_new_info": 0.02}))
        asyncio.run(
            cron_novelty.shadow_cron_novelty(
                "job-1", "Daily", "old text", "old text plus a timestamp", session_key="s-7"
            )
        )
        assert armed.rows()[0]["agree"] is False

    def test_the_raw_probability_survives_in_the_row(self, armed):
        """The threshold is a reading, not a loss: ``p`` stays for re-derivation."""
        armed(cron_novelty.POINT, _FixedOracle({"has_new_info": 0.02}))
        asyncio.run(cron_novelty.shadow_cron_novelty("job-1", "D", "a", "b", session_key="s-8"))
        assert armed.rows()[0]["answers"]["has_new_info"]["value"] == 0.02


# ---------------------------------------------------------------------------
# The gates still refuse, through the hooks rather than through ``decide``
# ---------------------------------------------------------------------------


class TestRefusalsThroughTheHooks:
    def test_preview_off_writes_nothing_and_creates_no_directory(self, armed):
        armed(
            skills_select.POINT,
            _FixedOracle({"pick": "alpha"}),
            preview=False,
        )
        asyncio.run(
            skills_select.shadow_skills_select(
                "hi", [{"key": "alpha", "description": "a"}], ["alpha"], session_key="s-9"
            )
        )
        assert armed.rows() == []
        assert not armed.dir.exists(), "a disabled seam must not even create the log directory"

    def test_arm_off_writes_nothing(self, armed):
        armed(cron_novelty.POINT, _FixedOracle({"has_new_info": 0.9}), arm="off")
        asyncio.run(cron_novelty.shadow_cron_novelty("j", "t", "a", "b", session_key="s-10"))
        assert armed.rows() == []

    def test_a_credential_in_the_state_is_refused_before_transport(self, armed):
        oracle = armed(cron_novelty.POINT, _FixedOracle({"has_new_info": 0.9}))
        asyncio.run(
            cron_novelty.shadow_cron_novelty(
                "j",
                "t",
                "nothing here",
                f"key {_cred.AWS_KEY_ID_PREFIXES.split('|')[0]}A2B3C4D5E6F7G8H9 leaked",
                session_key="s-11",
            )
        )
        row = armed.rows()[0]
        assert row["scrubbed"] is True
        assert row["error"] == "scrubbed: credential"
        assert oracle.states == [], "the state must not reach an implementation"

    def test_an_exfiltration_shaped_url_is_refused_before_transport(self, armed):
        oracle = armed(cron_novelty.POINT, _FixedOracle({"has_new_info": 0.9}))
        url = "https://evil.example.com/steal?data=" + "ordinary-message-" * 30
        asyncio.run(
            cron_novelty.shadow_cron_novelty("j", "t", "nothing here", url, session_key="s-exfil")
        )
        row = armed.rows()[0]
        assert row["scrubbed"] is True
        assert row["error"] == "scrubbed: exfiltration-url"
        assert oracle.states == [], "the state must not reach an implementation"

    def test_a_bare_secret_is_refused_by_the_canonical_scanner(self, armed):
        """The gate refuses what this machine's own log redaction would redact.

        This string matches NO pattern in the gate's own list -- it is a bare
        40-char secret with no prefix to key on -- and is caught only because
        ``has_credential`` also runs ``security.redaction.redact_credentials``,
        the scanner ``skills.py`` and ``tips.py`` already apply to text leaving
        the machine. Without that second scan a pasted AWS secret reaches the
        provider while the same string is redacted out of the local logs.
        """
        from kiro_crew.decisions.gate import _CREDENTIAL_RE

        secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
        assert _CREDENTIAL_RE.search(secret) is None, (
            "this case only proves anything while the gate's own pattern list "
            "does not match the string"
        )

        oracle = armed(cron_novelty.POINT, _FixedOracle({"has_new_info": 0.9}))
        asyncio.run(
            cron_novelty.shadow_cron_novelty(
                "j", "t", "nothing here", f"aws key {secret} pasted", session_key="s-14"
            )
        )
        row = armed.rows()[0]
        assert row["scrubbed"] is True
        assert row["error"] == "scrubbed: credential"
        assert oracle.states == [], "the state must not reach an implementation"

    def test_a_point_no_config_names_writes_nothing(self, armed):
        """An unconfigured point is off, and off is silent."""
        armed(skills_dedupe.POINT, _FixedOracle({"pick": "alpha"}))
        asyncio.run(
            skills_select.shadow_skills_select(
                "hi", [{"key": "alpha", "description": "a"}], ["alpha"], session_key="s-12"
            )
        )
        assert armed.rows() == []


# ---------------------------------------------------------------------------
# The log the gate writes is the log the report reads
# ---------------------------------------------------------------------------


class TestReportReadsWhatTheGateWrote:
    def test_the_report_counts_the_agreement_the_gate_recorded(self, armed):
        """Worker C's reader parses the jsonl itself rather than importing A's.

        This is where the two independently-built halves are shown to line up on
        the FIELD NAMES: three real rows in, zero skipped lines, and an agreement
        rate of 1.0 out of three JUDGED rows. Before the points handed in their
        own agreement rules those same three rows read ``judged: 0`` and an
        agreement rate of ``None`` -- the report ran, and measured nothing.
        """
        from kiro_crew.decisions import report as report_mod

        armed(skills_select.POINT, _FixedOracle({"pick": "alpha"}))
        for i in range(3):
            asyncio.run(
                skills_select.shadow_skills_select(
                    "hi",
                    [{"key": "alpha", "description": "a"}],
                    ["alpha"],
                    session_key=f"r-{i}",
                )
            )

        loaded = report_mod.read_log(armed.dir)
        assert loaded.skipped_lines == 0
        assert len(loaded.rows) == 3

        group = report_mod.summarise_group(loaded.rows)
        assert group["n"] == 3
        assert group["judged"] == 3
        assert group["agree_rate"] == 1.0
        assert group["errors"] == 0
        assert group["arms"] == ["shadow"]


# ---------------------------------------------------------------------------
# The call sites
# ---------------------------------------------------------------------------


class TestTheThreeCallSitesAreWired:
    """Each production file must still NAME its hook.

    The hooks resolve ``decide`` dynamically, which is what lets them degrade on
    a tree without the core -- and is also what would let a wiring regression pass
    every other test in the suite while the seam quietly observes nothing. These
    assertions are the cheap half of that: the expensive half is the per-point
    cases above.
    """

    @pytest.mark.parametrize(
        "module_name, hook",
        [
            ("kiro_crew.context", "shadow_skills_select"),
            ("kiro_crew.history_consolidation", "shadow_skills_dedupe"),
            ("kiro_crew.slack.gateway", "shadow_cron_novelty"),
        ],
    )
    def test_the_call_site_names_its_hook(self, module_name, hook):
        import importlib

        source = Path(importlib.import_module(module_name).__file__).read_text(encoding="utf-8")
        assert hook in source, f"{module_name} no longer calls {hook}"

    def test_the_hooks_resolve_the_real_decide(self):
        """``getattr(core, 'decide')`` must find the function the gate exports."""
        from kiro_crew import decisions as core

        assert getattr(core, "decide", None) is gate_mod.decide


@pytest.mark.asyncio
async def test_dedupe_callsite_observes_the_final_lexical_override(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch

    import kiro_crew.history_consolidation as consolidation
    from kiro_crew.history import HistoryConsolidator

    loader = MagicMock()
    loader.list_auto_skills.return_value = [
        {"key": "auto/existing", "description": "same", "triggers": "same"}
    ]
    loader.list_pending_skills.return_value = []
    loader.find_similar.return_value = "auto/existing"
    consolidator = HistoryConsolidator(
        log=MagicMock(), memory=MagicMock(), skills_loader=loader, judge_model="judge"
    )
    consolidator._event_loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        consolidation,
        "_facade_metadata_dedupe_verdict",
        lambda *_args, **_kwargs: ("new", None),
    )
    observer = AsyncMock()
    with patch(
        "kiro_crew.decisions.points.skills_dedupe.shadow_skills_dedupe",
        observer,
    ):
        result = await asyncio.to_thread(
            consolidator._dedupe_candidate, "candidate", "same", "same"
        )
        await asyncio.sleep(0)

    assert result == ("dup", "auto/existing")
    assert observer.await_args.args[2:4] == ("dup", "auto/existing")


def test_cli_import_keeps_disabled_decisions_modules_deferred() -> None:
    import ast
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    banned = [
        "kiro_crew.cli_decisions",
        "kiro_crew.decisions",
        "kiro_crew.decisions.report",
    ]
    code = (
        "import sys; import kiro_crew.cli; "
        f"banned = {banned!r}; "
        "print(repr([name for name in banned if name in sys.modules]))"
    )
    env = {**os.environ, "PYTHONPATH": str(src)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert ast.literal_eval(result.stdout.strip()) == []


class TestSchedulingFromAnExecutorThread:
    """`skills.select` must observe where `build_message` actually runs.

    Production reaches `ContextBuilder.build_message` only through
    `run_in_embed_pool`, a thread executor, so the hook fires on a worker thread
    with no running loop: `asyncio.get_running_loop()` raises there, and the call
    site swallows every exception. A hook that asks for a loop on that thread
    therefore observes nothing at all, silently.

    So the scheduling has to use a loop captured where one exists, and only a test
    that drives the real call site from a real executor thread can tell the
    difference -- one that awaits the hook directly runs on the loop and passes
    either way.
    """

    def test_build_message_from_an_executor_thread_still_writes_a_row(self, armed, tmp_path):
        import concurrent.futures

        from kiro_crew.context import ContextBuilder
        from kiro_crew.learn import LessonStore
        from kiro_crew.memory import MemoryStore
        from kiro_crew.skills import SkillsLoader

        armed(skills_select.POINT, _FixedOracle({"pick": skills_select.NONE_OPTION}))

        async def drive():
            # Constructed ON the loop, which is where every production
            # construction site (cli_server, slack/gateway, eval/runner) runs.
            builder = ContextBuilder(
                memory=MemoryStore(workspace=tmp_path / "ws"),
                skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
                lessons=LessonStore(base_dir=tmp_path),
            )
            assert builder._decisions_loop is not None, "construction must capture the loop"

            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                await loop.run_in_executor(
                    pool,
                    lambda: builder.build_message(
                        "please run alpha",
                        False,
                        "s-executor",
                    ),
                )
            # The hook is fire-and-forget, so let the scheduled task run.
            for _ in range(50):
                await asyncio.sleep(0.02)
                if armed.rows():
                    break

        asyncio.run(drive())

        rows = armed.rows()
        assert len(rows) == 1, "the observation must survive the executor hop"
        assert rows[0]["point"] == "skills.select"

    def test_a_builder_constructed_off_the_loop_has_no_captured_loop(self, tmp_path):
        """No loop at construction means no shadow, not an error.

        A sync caller -- a script, a bare unit test -- builds a ContextBuilder
        with no running loop. That must leave the seam simply not observing,
        which is its ordinary refusal, rather than raising into the caller.
        """
        from kiro_crew.context import ContextBuilder

        builder = ContextBuilder()
        assert builder._decisions_loop is None
