"""Autopilot must consume a stage's completion reports before advancing.

A stage turn can dispatch background agents and end while their terminal reports
are still entering the parent conversation.  The active LLM turn remains in
``slot.task`` so completion delivery waits for that turn, while the outer stage
controller waits for every terminal report before advancing.

These tests make that ordering deterministic: the agent is already done, its
terminal report is still pending, and Stage 2 may start only after that report
has queued and its completion turn has finished.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from chat_test_helpers import _make_state

from kiro_crew.acp.client import AcpAuthRequired
from kiro_crew.dashboard.chat_utils import SUBAGENT_COMPLETION_KIND
from kiro_crew.subagent import SubagentManager


@pytest.fixture(autouse=True)
def _isolate_config_dir(tmp_path, monkeypatch):
    """Keep stage result files inside the test's temporary directory."""
    for module in ("state", "chat", "chat_orchestrator"):
        monkeypatch.setattr(f"kiro_crew.dashboard.{module}.config_dir", lambda: tmp_path)


class _StageDeliveryManager:
    """Minimal manager double with one parent report still in flight."""

    def __init__(self) -> None:
        self.report_task: asyncio.Task | None = None

    def running_agents_for(self, _parent: str) -> list[dict]:
        return []

    async def has_pending_work_for_async(self, _parent: str) -> bool:
        return False

    async def wait_for_parent_reports(self, _parent: str) -> None:
        if self.report_task is not None:
            await asyncio.shield(self.report_task)


def _mark_consumed(kwargs: dict) -> None:
    callback = kwargs.get("_on_consumed")
    if callable(callback):
        callback(True)


def _auth_retry_state(tmp_path, name: str):
    """Build a real queued-turn runner whose provider requires login."""
    state = _make_state(tmp_path)
    state.sessions.get_or_create = AsyncMock(return_value=(MagicMock(), False, False))
    state.sessions.release = MagicMock()
    state.sessions.reset = AsyncMock()
    state.sessions.set_approval_policy = MagicMock()
    state.sessions.check_context_usage = MagicMock()
    state.sessions.get_slack_link = MagicMock(return_value=(None, None))
    state.sessions.record_failure = AsyncMock()
    state.broadcast_ws = MagicMock()
    state.push_slots_update = MagicMock()
    state.is_yolo_active = MagicMock(return_value=False)
    state._background_tasks = set()
    slot = state.get_or_create_slot(name)
    client = state.sessions.get_or_create.return_value[0]
    client.shutdown = AsyncMock()

    async def _auth_fails(_message):
        raise AcpAuthRequired("kiro-cli is not logged in.")
        yield  # pragma: no cover - async-generator shape only

    client.stream = _auth_fails
    client.stream_command = _auth_fails
    return state, slot


@pytest.mark.asyncio
async def test_auth_retry_does_not_promote_pasted_completion_text(tmp_path):
    """A transcript role inferred from user text is not retry provenance."""
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

    content = "[Subagent completion event]\nuser pasted an old completion"
    state, slot = _auth_retry_state(tmp_path, "pasted-completion-auth")
    slot.queue_append(content)

    assert await _start_next_queued_turn(state, slot) is True
    task = slot.task
    assert task is not None
    await task

    assert any(
        row.get("role") == "subagent" and row.get("content") == content for row in slot.messages
    ), "the test did not exercise the content-derived transcript role"
    assert not any(
        entry.get("kind") == SUBAGENT_COMPLETION_KIND and entry.get("content") == content
        for entry in slot._queue
    ), "content-derived transcript role forged a structural completion retry"


@pytest.mark.asyncio
async def test_auth_retry_restores_structurally_tagged_completion(tmp_path):
    """A genuine injected completion remains retryable after login."""
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

    content = "[Subagent completion event]\nreal completion"
    state, slot = _auth_retry_state(tmp_path, "tagged-completion-auth")
    slot._in_stage_execution = True
    slot.queue_append(content, kind=SUBAGENT_COMPLETION_KIND)

    assert await _start_next_queued_turn(state, slot) is True
    task = slot.task
    assert task is not None
    await task

    retries = [
        entry
        for entry in slot._queue
        if entry.get("kind") == SUBAGENT_COMPLETION_KIND and entry.get("content") == content
    ]
    assert len(retries) == 1
    assert slot._stage_delivery_retry_queue_id == retries[0]["id"]


@pytest.mark.asyncio
async def test_completion_turn_finishes_before_next_stage_starts(tmp_path, monkeypatch):
    """A done agent with an undelivered report must hold the stage boundary."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    manager = _StageDeliveryManager()
    state.subagents = manager
    slot = state.get_or_create_slot("completion-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True

    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if message.startswith("[Subagent completion event]"):
            order.append("completion")
            _slot.append("assistant", "completion synthesized", "msg msg-a")
            return
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.append("assistant", "agents dispatched", "msg msg-a")

            async def _finish_report() -> None:
                await asyncio.sleep(0)
                _slot.queue_append(
                    "[Subagent completion event]\nall agents complete",
                    kind=SUBAGENT_COMPLETION_KIND,
                )

            manager.report_task = asyncio.create_task(_finish_report())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            _slot.append("assistant", "verified", "msg msg-a")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    async def _start_system_turn(_state, _slot):
        index = next(
            (
                i
                for i, entry in enumerate(_slot._queue)
                if entry.get("kind") == SUBAGENT_COMPLETION_KIND
            ),
            None,
        )
        if index is None:
            return False
        entry = _slot._queue.pop(index)
        task = asyncio.create_task(_mock_run_chat(_state, _slot, entry["content"]))
        _slot.task = task
        return True

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_system_turn,
    )

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)
    if slot.task is not None and slot.task is not asyncio.current_task():
        await asyncio.wait_for(slot.task, timeout=5)

    assert order == ["stage-1", "completion", "stage-2"]


@pytest.mark.asyncio
async def test_manager_waits_for_parent_terminal_reports():
    """The manager's parent-scoped wait follows report tasks, not ``info.done``."""
    manager = object.__new__(SubagentManager)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _report() -> None:
        entered.set()
        await release.wait()

    report = asyncio.create_task(_report())
    owner = SimpleNamespace(parent_session_key="dashboard:parent")
    manager._report_tasks = {report}
    manager._report_owners = {report: owner}
    manager._report_failures = {}

    def _forget(task: asyncio.Task) -> None:
        manager._report_tasks.discard(task)
        manager._report_owners.pop(task, None)

    report.add_done_callback(_forget)
    waiter = asyncio.create_task(manager.wait_for_parent_reports("dashboard:parent"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not waiter.done(), "the parent wait returned before terminal delivery finished"

    release.set()
    await asyncio.wait_for(waiter, timeout=1)


@pytest.mark.asyncio
async def test_live_stage_controller_counts_as_running_between_stages(tmp_path):
    """The slot stays busy while its outer controller owns the stage boundary."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("running-handoff", mode="orchestrator")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _controller() -> None:
        entered.set()
        await release.wait()

    controller = asyncio.create_task(_controller())
    slot.track_stage_controller(controller)
    await asyncio.wait_for(entered.wait(), timeout=1)
    slot.task = None

    try:
        assert slot.running is True
    finally:
        release.set()
        await asyncio.wait_for(controller, timeout=1)


class _KeyedStageDeliveryManager:
    """Record every parent key the stage controller asks about."""

    def __init__(self) -> None:
        self.running_keys: list[str] = []
        self.report_keys: list[str] = []

    def running_agents_for(self, parent: str) -> list[dict]:
        self.running_keys.append(parent)
        return []

    async def has_pending_work_for_async(self, _parent: str) -> bool:
        return False

    async def wait_for_parent_reports(self, parent: str) -> bool:
        self.report_keys.append(parent)
        return False


@pytest.mark.asyncio
async def test_channel_linked_stage_uses_effective_parent_key(tmp_path, monkeypatch):
    """Channel-born children and terminal reports live under the channel key."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    manager = _KeyedStageDeliveryManager()
    state.subagents = manager
    slot = state.get_or_create_slot("linked-handoff", mode="orchestrator")
    slot.linked_session_key = "slack:C123:1700000000.000001"
    slot._stage_titles = ["Only"]
    slot._plan_goal = "Use the linked session"
    slot._auto_run = True

    async def _mock_run_chat(_state, _slot, _message, **kwargs):
        _mark_consumed(kwargs)
        _slot.append("assistant", "done", "msg msg-a")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    expected = "slack:C123:1700000000.000001"
    assert expected in manager.running_keys
    assert manager.report_keys
    assert set(manager.report_keys) == {expected}


@pytest.mark.asyncio
async def test_auth_refusal_restores_completion_and_pauses(tmp_path, monkeypatch):
    """A signed-out completion resumes its stage boundary before Stage 2."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    manager = _StageDeliveryManager()
    state.subagents = manager
    slot = state.get_or_create_slot("auth-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    order: list[str] = []
    refuse_auth = True

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if message.startswith("[Subagent completion event]"):
            order.append("completion")
            _slot.append("assistant", "completion synthesized", "msg msg-a")
            return
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.append("assistant", "agents dispatched", "msg msg-a")

            async def _finish_report() -> None:
                await asyncio.sleep(0)
                _slot.queue_append(
                    "[Subagent completion event]\nall agents complete",
                    kind=SUBAGENT_COMPLETION_KIND,
                )

            manager.report_task = asyncio.create_task(_finish_report())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected direct turn: {message[:80]}")

    async def _start_completion_turn(_state, _slot):
        index = next(
            i
            for i, entry in enumerate(_slot._queue)
            if entry.get("kind") == SUBAGENT_COMPLETION_KIND
        )
        entry = _slot._queue.pop(index)

        async def _completion_turn() -> None:
            if refuse_auth:
                _slot.queue_insert(
                    0,
                    entry["content"],
                    kind=SUBAGENT_COMPLETION_KIND,
                )
                _slot._last_turn_auth_required = True
                return
            await _mock_run_chat(_state, _slot, entry["content"])

        task = asyncio.create_task(_completion_turn())
        _slot.task = task
        return True

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_completion_turn,
    )

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1"]
    assert any(
        entry.get("kind") == SUBAGENT_COMPLETION_KIND for entry in slot._queue
    ), "the completion must remain queued for post-login retry"
    text = "\n".join(
        str(message.get("content", ""))
        for message in slot.messages
        if message.get("role") == "assistant"
    )
    assert "completion event could not be processed" in text

    refuse_auth = False
    slot._last_turn_auth_required = False
    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1", "completion", "stage-2"]
    assert not any(entry.get("kind") == SUBAGENT_COMPLETION_KIND for entry in slot._queue)


@pytest.mark.asyncio
async def test_synthetic_recovery_finishes_before_next_stage(tmp_path, monkeypatch):
    """A recovery turn started by Stage 1 must finish before Stage 2."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("recovery-handoff", mode="orchestrator")
    slot._stage_titles = ["Recover", "Verify"]
    slot._plan_goal = "Recover then verify"
    slot._auto_run = True
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")

            async def _recovery() -> None:
                order.append("recovery-start")
                await asyncio.sleep(0)
                order.append("recovery-end")
                _slot._synthetic_recovery_inflight -= 1

            _slot._synthetic_recovery_inflight += 1
            _slot.task = asyncio.create_task(_recovery())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1", "recovery-start", "recovery-end", "stage-2"]


@pytest.mark.asyncio
async def test_closing_slot_cancels_controller_before_next_stage(tmp_path, monkeypatch):
    """Closing an active Autopilot slot prevents every remaining stage."""
    from kiro_crew.dashboard.chat_handlers import close_slot
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("close-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    stage_one_started = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            stage_one_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # _run_chat owns cancellation cleanup and returns normally.
                return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    async def _retire_nudge(_slot_key):
        return None

    async def _save_slot(*_args, **_kwargs):
        return None

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go all"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._retire_slot_nudge_loop", _retire_nudge)
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.save_slot_off_loop", _save_slot)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot.task
    assert controller is not None
    await asyncio.wait_for(stage_one_started.wait(), timeout=1)

    await close_slot(state, slot, slot.key)
    if not controller.done():
        await asyncio.wait_for(asyncio.shield(controller), timeout=1)

    assert controller.done()
    assert order == ["stage-1"]


@pytest.mark.asyncio
async def test_stop_between_stages_cancels_controller_before_next_stage(tmp_path, monkeypatch):
    """Session Stop cancels the live controller even when no child turn owns the slot."""
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("stop-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    boundary_entered = asyncio.Event()
    release_boundary = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.append("assistant", "collected", "msg msg-a")
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    async def _hold_stage_boundary(_state, _slot, _tracker, stage_num, **_kwargs):
        if stage_num == 1:
            boundary_entered.set()
            await release_boundary.wait()
        return True

    async def _stop_turn(*_args, **_kwargs):
        return "idle"

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go all"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._settle_stage_delivery",
        _hold_stage_boundary,
    )
    monkeypatch.setattr(state.sessions, "stop_turn", _stop_turn)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot._stage_controller_task
    assert controller is not None
    await asyncio.wait_for(boundary_entered.wait(), timeout=1)
    assert slot.task is None

    result = await stop_slot_turn(state, slot, source="session_control")
    release_boundary.set()
    await asyncio.gather(controller, return_exceptions=True)

    assert result["ok"] is True
    assert slot._auto_run is False
    assert controller.done()
    assert order == ["stage-1"]


@pytest.mark.asyncio
async def test_live_completion_turn_holds_stage_boundary(tmp_path, monkeypatch):
    """A live completion turn must finish before the next stage starts."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("live-completion-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    completion_started = asyncio.Event()
    stage_two_started = asyncio.Event()
    release_completion = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")

            async def _completion_turn() -> None:
                order.append("completion-start")
                completion_started.set()
                await release_completion.wait()
                order.append("completion-end")

            _slot.task = asyncio.create_task(_completion_turn())
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            stage_two_started.set()
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    controller = asyncio.create_task(_stage_loop(state, slot, auto_run=True))
    await asyncio.wait_for(completion_started.wait(), timeout=1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stage_two_started.wait(), timeout=0.1)

    release_completion.set()
    await asyncio.wait_for(controller, timeout=5)

    assert order == ["stage-1", "completion-start", "completion-end", "stage-2"]


@pytest.mark.asyncio
async def test_final_controller_await_does_not_strand_a_late_queue_entry(tmp_path, monkeypatch):
    """A message queued during the final payload await must start before release."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("late-queue-handoff", mode="orchestrator")
    slot._stage_titles = ["Only"]
    slot._plan_goal = "Finish once"
    slot._auto_run = True
    started: list[str] = []

    async def _mock_run_chat(_state, _slot, _message, **kwargs):
        _mark_consumed(kwargs)
        _slot.append("assistant", "stage complete", "msg msg-a")

    async def _start_queued(_state, _slot):
        if not _slot._queue:
            return False
        entry = _slot.queue_pop(0)
        started.append(entry["content"])
        _slot.task = asyncio.create_task(asyncio.sleep(0))
        return True

    async def _late_done_payload(_state, _slot, **_kwargs):
        await asyncio.sleep(0)
        _slot.queue_append("queued during done payload")
        return {"slot": _slot.key, "continuing": True, "needs_input": False}

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_queued,
    )
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator.chat_done_payload",
        _late_done_payload,
    )

    controller = asyncio.create_task(_stage_loop(state, slot, auto_run=True))
    slot.track_stage_controller(controller)
    slot.task = controller
    await asyncio.wait_for(controller, timeout=5)
    if slot.task is not None and slot.task is not controller:
        await asyncio.wait_for(slot.task, timeout=1)

    assert started == ["queued during done payload"]
    assert slot._queue == []


class _ReboundStageDeliveryManager:
    """Hold the terminal report under the stage turn's original parent key."""

    def __init__(self, original_key: str) -> None:
        self.original_key = original_key
        self.release = asyncio.Event()
        self.wait_started = asyncio.Event()
        self.waited = False

    def running_agents_for(self, _parent: str) -> list[dict]:
        return []

    async def has_pending_work_for_async(self, _parent: str) -> bool:
        return False

    async def wait_for_parent_reports(self, parent: str) -> bool:
        if parent != self.original_key or self.waited:
            return False
        self.waited = True
        self.wait_started.set()
        await self.release.wait()
        return True


@pytest.mark.asyncio
async def test_stage_rebind_still_waits_for_the_original_parent_reports(tmp_path, monkeypatch):
    """A mid-stage link cannot move the boundary away from existing children."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("rebind-handoff", mode="orchestrator")
    original_key = f"dashboard:{slot.key}"
    manager = _ReboundStageDeliveryManager(original_key)
    state.subagents = manager
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    stage_two_started = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            _slot.linked_session_key = "cron:rebound-parent"
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            stage_two_started.set()
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    controller = asyncio.create_task(_stage_loop(state, slot, auto_run=True))
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stage_two_started.wait(), timeout=0.1)
    finally:
        manager.release.set()
    await asyncio.wait_for(controller, timeout=5)

    assert manager.wait_started.is_set()
    assert order == ["stage-1", "stage-2"]


@pytest.mark.asyncio
async def test_failed_completion_turn_requeues_its_unconsumed_entry(tmp_path, monkeypatch):
    """A pre-consumption turn failure must leave the exact completion retryable."""
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("failed-completion-handoff", mode="orchestrator")
    slot._in_stage_execution = True
    content = "[Subagent completion event]\nresult still owed"
    consumption: list[bool] = []
    slot.queue_insert(
        0,
        content,
        kind=SUBAGENT_COMPLETION_KIND,
        on_consumed=consumption.append,
    )

    async def _failing_turn(*_args, **_kwargs):
        raise RuntimeError("completion turn failed before consumption")

    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _failing_turn)

    assert await _start_next_queued_turn(state, slot) is True
    task = slot.task
    assert task is not None
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    retries = [
        entry
        for entry in slot._queue
        if entry.get("kind") == SUBAGENT_COMPLETION_KIND and entry.get("content") == content
    ]
    assert len(retries) == 1
    assert callable(retries[0].get("_on_consumed"))
    assert consumption == []


@pytest.mark.asyncio
async def test_queued_stage_agent_holds_boundary_until_registered(tmp_path, monkeypatch):
    """A spawn waiting behind admission is still unfinished stage work."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("queued-agent-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    queue_checked = asyncio.Event()
    release_queue = asyncio.Event()
    stage_two_started = asyncio.Event()
    order: list[str] = []

    class _QueuedManager(_StageDeliveryManager):
        async def has_pending_work_for_async(self, _parent: str) -> bool:
            queue_checked.set()
            await release_queue.wait()
            return False

    state.subagents = _QueuedManager()

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            stage_two_started.set()
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    controller = asyncio.create_task(_stage_loop(state, slot, auto_run=True))
    await asyncio.wait_for(queue_checked.wait(), timeout=1)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stage_two_started.wait(), timeout=0.1)
    finally:
        release_queue.set()
    await asyncio.wait_for(controller, timeout=5)

    assert order == ["stage-1", "stage-2"]


@pytest.mark.asyncio
async def test_unconsumed_auth_refusal_retries_the_same_stage(tmp_path, monkeypatch):
    """Signing in after a pre-consumption refusal must not skip the stage."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("stage-auth-retry", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    refuse_auth = True
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        nonlocal refuse_auth
        if "Execute Stage 1 of 2 now" in message:
            if refuse_auth:
                order.append("stage-1-auth")
                _slot._last_turn_auth_required = True
                return
            order.append("stage-1")
            callback = kwargs.get("_on_consumed")
            if callback is not None:
                callback(True)
            _slot._last_turn_auth_required = False
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)
    assert order == ["stage-1-auth"]

    refuse_auth = False
    slot._last_turn_auth_required = False
    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1-auth", "stage-1", "stage-2"]


@pytest.mark.asyncio
async def test_handled_unconsumed_completion_turn_requeues_its_entry(tmp_path, monkeypatch):
    """A handled pre-consumption failure still owes the exact completion."""
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("handled-completion-handoff", mode="orchestrator")
    slot._in_stage_execution = True
    content = "[Subagent completion event]\nhandled failure still owes this result"
    slot.queue_insert(0, content, kind=SUBAGENT_COMPLETION_KIND)

    async def _handled_failure(*_args, **_kwargs):
        return None

    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _handled_failure)

    assert await _start_next_queued_turn(state, slot) is True
    task = slot.task
    assert task is not None
    await task
    await asyncio.sleep(0)

    retries = [
        entry
        for entry in slot._queue
        if entry.get("kind") == SUBAGENT_COMPLETION_KIND and entry.get("content") == content
    ]
    assert len(retries) == 1, "normal task return hid an unconsumed handled failure"


@pytest.mark.asyncio
async def test_auth_after_tool_consumption_does_not_requeue_completion(tmp_path):
    """An auth failure after irreversible tool work must not replay the turn."""
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn
    from kiro_crew.providers.base import EVENT_TOOL_CALL, LLMEvent

    content = "[Subagent completion event]\nrun one tool before auth fails"
    state, slot = _auth_retry_state(tmp_path, "consumed-completion-auth")
    client = state.sessions.get_or_create.return_value[0]

    async def _tool_then_auth(_message):
        yield LLMEvent(
            kind=EVENT_TOOL_CALL,
            tool_call_id="tool-before-auth",
            title="read",
            tool_kind="read",
        )
        raise AcpAuthRequired("kiro-cli is not logged in.")

    client.stream = _tool_then_auth
    client.stream_command = _tool_then_auth
    slot._in_stage_execution = True
    slot.queue_append(content, kind=SUBAGENT_COMPLETION_KIND)

    assert await _start_next_queued_turn(state, slot) is True
    task = slot.task
    assert task is not None
    await task

    assert not any(
        entry.get("kind") == SUBAGENT_COMPLETION_KIND and entry.get("content") == content
        for entry in slot._queue
    ), "auth retry replayed a completion after irreversible tool consumption"


@pytest.mark.asyncio
async def test_pending_delivery_exit_does_not_start_completion_outside_stage_guard(
    tmp_path, monkeypatch
):
    """A failed settlement leaves its completion queued for the next guarded Go."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("pending-exit-handoff", mode="orchestrator")
    slot._stage_titles = ["Collect"]
    slot._plan_goal = "Collect once"
    slot._stage_delivery_pending = 1
    slot._stage_delivery_consumed = True
    content = "[Subagent completion event]\nretry under stage protection"
    slot.queue_append(content, kind=SUBAGENT_COMPLETION_KIND)
    started: list[str] = []

    async def _load_budgets(_slot, _tracker):
        return True

    async def _settlement_fails(*_args, **_kwargs):
        return False

    async def _start_queued(_state, _slot):
        entry = _slot.queue_pop(0)
        started.append(entry["content"])
        return True

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._load_plan_budgets", _load_budgets)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._settle_stage_delivery", _settlement_fails
    )
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn", _start_queued
    )

    await _stage_loop(state, slot, auto_run=True)

    assert started == []
    assert slot._stage_delivery_pending == 1
    assert [entry.get("content") for entry in slot._queue] == [content]


@pytest.mark.asyncio
async def test_go_resumes_consumed_auth_interrupted_stage_before_advancing(tmp_path, monkeypatch):
    """After sign-in, Go must finish Stage N before Stage N+1 starts."""
    from kiro_crew.dashboard.chat_orchestrator import _stage_loop, api_chat_plan_action
    from kiro_crew.dashboard.chat_utils import _MANUAL_RESUME_MSG

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("consumed-stage-auth-resume", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    resume_started = asyncio.Event()
    release_resume = asyncio.Event()
    stage_two_started = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        callback = kwargs.get("_on_consumed")
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1-auth")
            if callback is not None:
                callback(True)
            _slot._last_turn_auth_required = True
            return
        if message == _MANUAL_RESUME_MSG:
            order.append("stage-1-resume-start")
            resume_started.set()
            await release_resume.wait()
            order.append("stage-1-resume-end")
            if callback is not None:
                callback(True)
            _slot._last_turn_auth_required = False
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            stage_two_started.set()
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)
    assert order == ["stage-1-auth"]
    assert slot._stage_delivery_pending == 1
    assert slot._stage_delivery_consumed is True

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot._stage_controller_task
    assert controller is not None
    await asyncio.wait_for(resume_started.wait(), timeout=1)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stage_two_started.wait(), timeout=0.1)
    finally:
        release_resume.set()
    await asyncio.wait_for(controller, timeout=5)

    assert order == [
        "stage-1-auth",
        "stage-1-resume-start",
        "stage-1-resume-end",
        "stage-2",
    ]


@pytest.mark.asyncio
async def test_unrelated_completion_does_not_suppress_consumed_stage_continuation(
    tmp_path, monkeypatch
):
    """Only this auth-failed turn's exact retry can replace its continuation."""
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action
    from kiro_crew.dashboard.chat_utils import _MANUAL_RESUME_MSG

    unrelated = "[Subagent completion event]\nunrelated terminal report"
    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("unrelated-completion-auth", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._stage_delivery_pending = 1
    slot._stage_delivery_consumed = True
    slot._last_turn_auth_required = True
    slot.queue_append(unrelated, kind=SUBAGENT_COMPLETION_KIND)
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        callback = kwargs.get("_on_consumed")
        if message == _MANUAL_RESUME_MSG:
            order.append("stage-continuation")
        elif message == unrelated:
            order.append("unrelated-completion")
        elif "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
        else:
            raise AssertionError(f"unexpected turn: {message[:80]}")
        if callable(callback):
            callback(True)
        _slot._last_turn_auth_required = False

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _mock_run_chat)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot._stage_controller_task
    assert controller is not None
    await asyncio.wait_for(controller, timeout=5)

    assert order == ["stage-continuation", "unrelated-completion", "stage-2"]
    assert slot._stage_delivery_retry_queue_id == ""


@pytest.mark.asyncio
async def test_go_retries_unconsumed_completion_without_stage_continuation(tmp_path, monkeypatch):
    """An exact unconsumed completion retry outranks a stage continuation."""
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action
    from kiro_crew.dashboard.chat_utils import _MANUAL_RESUME_MSG

    content = "[Subagent completion event]\nunconsumed completion retry"
    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("unconsumed-completion-auth", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._stage_delivery_pending = 1
    slot._stage_delivery_consumed = True
    slot._last_turn_auth_required = True
    retry_queue_id = slot.queue_append(content, kind=SUBAGENT_COMPLETION_KIND)
    slot._stage_delivery_retry_queue_id = retry_queue_id
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        callback = kwargs.get("_on_consumed")
        if message == _MANUAL_RESUME_MSG:
            order.append("stage-continuation")
        elif message == content:
            order.append("completion-retry")
        elif "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
        else:
            raise AssertionError(f"unexpected turn: {message[:80]}")
        if callback is not None:
            callback(True)
        _slot._last_turn_auth_required = False

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _mock_run_chat)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot._stage_controller_task
    assert controller is not None
    await asyncio.wait_for(controller, timeout=5)

    assert order == ["completion-retry", "stage-2"]
    assert slot._stage_delivery_retry_queue_id == ""


@pytest.mark.asyncio
async def test_consumed_completion_auth_resumes_without_replaying_exact_input(
    tmp_path, monkeypatch
):
    """A consumed completion continues after login; its input never replays."""
    from kiro_crew.dashboard.chat_orchestrator import _stage_loop, api_chat_plan_action
    from kiro_crew.dashboard.chat_utils import _MANUAL_RESUME_MSG

    content = "[Subagent completion event]\nconsumed completion input"
    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("consumed-completion-auth", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._stage_delivery_pending = 1
    slot._stage_delivery_consumed = True
    slot.queue_append(content, kind=SUBAGENT_COMPLETION_KIND)
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        callback = kwargs.get("_on_consumed")
        if message == content:
            order.append("completion-auth")
            if callback is not None:
                callback(True)
            _slot._last_turn_auth_required = True
            return
        if message == _MANUAL_RESUME_MSG:
            order.append("completion-continuation")
            if callback is not None:
                callback(True)
            _slot._last_turn_auth_required = False
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)
    assert order == ["completion-auth"]
    assert slot._stage_delivery_pending == 1
    assert not any(entry.get("content") == content for entry in slot._queue)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot._stage_controller_task
    assert controller is not None
    await asyncio.wait_for(controller, timeout=5)

    assert order == ["completion-auth", "completion-continuation", "stage-2"]


@pytest.mark.asyncio
async def test_soft_stop_restores_unconsumed_stage_delivery(tmp_path, monkeypatch):
    """Soft Stop preserves a popped stage input so a later resume can run it."""
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("soft-stop-stage-delivery", mode="orchestrator")
    slot._in_stage_execution = True
    content = "[Subagent completion event]\npreserve me across soft Stop"
    slot.queue_append(content, kind=SUBAGENT_COMPLETION_KIND)
    turn_started = asyncio.Event()

    async def _blocking_turn(*_args, **_kwargs):
        turn_started.set()
        await asyncio.Event().wait()

    async def _controller() -> None:
        assert await _start_next_queued_turn(state, slot) is True
        child = slot.task
        assert child is not None
        await child

    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _blocking_turn)

    controller = asyncio.create_task(_controller())
    slot.track_stage_controller(controller)
    await asyncio.wait_for(turn_started.wait(), timeout=1)

    async def _soft_stop(_key, *, force=False, preserve_queue=False, on_soft=None, **_kwargs):
        assert force is False
        assert preserve_queue is True
        child = slot.task
        assert child is not None and child is not controller
        child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        if on_soft is not None:
            await on_soft()
        return "soft"

    state.sessions.stop_turn = _soft_stop

    result = await stop_slot_turn(state, slot)
    await asyncio.gather(controller, return_exceptions=True)
    await asyncio.sleep(0)

    assert result == {"ok": True}
    retries = [
        entry
        for entry in slot._queue
        if entry.get("kind") == SUBAGENT_COMPLETION_KIND and entry.get("content") == content
    ]
    assert len(retries) == 1, "soft Stop discarded the unconsumed stage delivery"


@pytest.mark.asyncio
async def test_soft_stop_generation_blocks_advance_after_swallowed_cancel(tmp_path, monkeypatch):
    """A turn swallowing soft cancellation cannot advance to Stage 2."""
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("soft-stop-generation", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    stage_one_started = asyncio.Event()
    stage_two_started = asyncio.Event()

    async def _swallowing_stage(_state, _slot, message, **_kwargs):
        if "Execute Stage 1 of 2 now" in message:
            stage_one_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return
        if "Execute Stage 2 of 2 now" in message:
            stage_two_started.set()

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go all"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _swallowing_stage)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    controller = slot._stage_controller_task
    assert controller is not None
    await asyncio.wait_for(stage_one_started.wait(), timeout=1)

    async def _soft_stop(_key, *, force=False, preserve_queue=False, on_soft=None, **_kwargs):
        assert force is False
        assert preserve_queue is True
        child = slot.task
        assert child is not None and child is not controller
        child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        if on_soft is not None:
            await on_soft()
        return "soft"

    state.sessions.stop_turn = _soft_stop
    result = await stop_slot_turn(state, slot)
    await asyncio.gather(controller, return_exceptions=True)

    assert result == {"ok": True}
    assert not stage_two_started.is_set()
    assert slot._stage_delivery_pending == 1
    assert slot._stage_delivery_consumed is False


@pytest.mark.asyncio
async def test_hard_stop_generation_blocks_restore_after_swallowed_cancel(tmp_path, monkeypatch):
    """A swallowed hard kill cannot restore the stage input it discarded."""
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("hard-stop-stage-delivery", mode="orchestrator")
    slot._in_stage_execution = True
    content = "[Subagent completion event]\ndiscard me on hard Stop"
    slot.queue_append(content, kind=SUBAGENT_COMPLETION_KIND)
    turn_started = asyncio.Event()

    async def _swallowing_turn(*_args, **_kwargs):
        turn_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    async def _controller() -> None:
        assert await _start_next_queued_turn(state, slot) is True
        child = slot.task
        assert child is not None
        await child

    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _swallowing_turn)

    controller = asyncio.create_task(_controller())
    slot.track_stage_controller(controller)
    await asyncio.wait_for(turn_started.wait(), timeout=1)
    slot._stop_state = "soft_pending"

    async def _hard_stop(_key, *, force=False, on_hard=None, **_kwargs):
        assert force is True
        child = slot.task
        assert child is not None and child is not controller
        child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        if on_hard is not None:
            await on_hard()
        return "hard"

    state.sessions.stop_turn = _hard_stop
    result = await stop_slot_turn(state, slot)
    await asyncio.gather(controller, return_exceptions=True)
    await asyncio.sleep(0)

    assert result == {"ok": True}
    assert not any(entry.get("content") == content for entry in slot._queue)


@pytest.mark.asyncio
async def test_failed_parent_report_task_halts_before_next_stage(tmp_path, monkeypatch):
    """A terminal-report exception cannot count as delivered stage output."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("failed-parent-report", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    parent_key = f"dashboard:{slot.key}"
    release = asyncio.Event()
    order: list[str] = []

    manager = object.__new__(SubagentManager)
    manager._report_owners = {}
    manager._report_failures = {}
    manager.running_agents_for = MagicMock(return_value=[])

    async def _has_pending_work(_parent: str) -> bool:
        release.set()
        return False

    manager.has_pending_work_for_async = _has_pending_work
    state.subagents = manager

    async def _failed_report() -> None:
        await release.wait()
        raise RuntimeError("terminal report delivery failed")

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        _mark_consumed(kwargs)
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            report = asyncio.create_task(_failed_report())
            manager._report_owners[report] = SimpleNamespace(parent_session_key=parent_key)
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1"]
    assert slot._auto_run is False
    assert any(
        "completion event could not be processed" in str(row.get("content", ""))
        for row in slot.messages
    )


@pytest.mark.asyncio
async def test_done_parent_run_stays_pending_until_report_registration():
    """A done record remains pending while its outer run task can register a report."""
    from kiro_crew.subagent_manager.run import RunEventCoordinator

    release = asyncio.Event()
    run_task = asyncio.create_task(release.wait())
    info = SimpleNamespace(
        id="done-before-report",
        parent_session_key="dashboard:parent",
        done=True,
    )
    manager = SimpleNamespace(
        _agents={info.id: info},
        _tasks={info.id: run_task},
        _queued_depth_async=AsyncMock(return_value=0),
        running=[],
    )
    coordinator = RunEventCoordinator(manager)

    assert await coordinator.has_pending_work_for_async_impl("dashboard:parent") is True
    release.set()
    await run_task
    assert await coordinator.has_pending_work_for_async_impl("dashboard:parent") is False


@pytest.mark.asyncio
async def test_synthesis_auth_retry_keeps_inject_provenance(tmp_path, monkeypatch):
    """A signed-out synthesis retry cannot become user-authored transcript text."""
    from kiro_crew.dashboard.chat_runner import _run_chat, _start_next_queued_turn
    from kiro_crew.dashboard.chat_utils import SYNTHETIC_RECOVERY_KIND
    from kiro_crew.dashboard.state import SUBAGENT_SYNTHESIS_PROMPT

    state, slot = _auth_retry_state(tmp_path, "synthesis-auth-retry")

    await _run_chat(
        state,
        slot,
        SUBAGENT_SYNTHESIS_PROMPT,
        _synthetic_payload=True,
        _turn_actor="subagent",
    )

    retry = next(
        entry for entry in slot._queue if entry.get("content") == SUBAGENT_SYNTHESIS_PROMPT
    )
    assert retry.get("kind") == SYNTHETIC_RECOVERY_KIND

    async def _successful_retry(*_args, **_kwargs):
        return None

    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _successful_retry)
    slot._last_turn_auth_required = False
    assert await _start_next_queued_turn(state, slot) is True
    task = slot.task
    assert task is not None
    await task

    retry_rows = [row for row in slot.messages if row.get("content") == SUBAGENT_SYNTHESIS_PROMPT]
    assert retry_rows[-1].get("role") == "inject"
    assert (retry_rows[-1].get("meta") or {}).get("injectKind") == "recovery"


@pytest.mark.asyncio
async def test_already_failed_parent_report_is_not_lost_before_settlement():
    """A completed failed report remains observable until its parent barrier runs."""
    from kiro_crew.subagent import SubagentReportDeliveryError

    async def _fail() -> None:
        raise RuntimeError("report failed before settlement")

    report = asyncio.create_task(_fail())
    await asyncio.gather(report, return_exceptions=True)
    manager = object.__new__(SubagentManager)
    manager._report_owners = {
        report: SimpleNamespace(parent_session_key="dashboard:parent"),
    }
    manager._report_failures = {}

    with pytest.raises(SubagentReportDeliveryError):
        await manager.wait_for_parent_reports("dashboard:parent")


@pytest.mark.asyncio
async def test_preconsumption_stage_exit_pauses_before_advance(tmp_path, monkeypatch):
    """A normal early return before provider consumption cannot complete a stage."""
    from kiro_crew.dashboard.chat import _stage_loop

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("preconsumption-stage-exit", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    order: list[str] = []

    async def _returns_before_consumption(_state, _slot, message, **_kwargs):
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._run_chat",
        _returns_before_consumption,
    )

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1"]
    assert slot._stage_delivery_pending == 1
    assert slot._stage_delivery_consumed is False
    assert slot._auto_run is False


def test_pending_stage_boundary_counts_as_busy_until_plan_cancel(tmp_path):
    """Destructive/background entry points see a pending stage as busy."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("pending-stage-busy", mode="orchestrator")
    assert slot.task is None and slot._stage_controller_task is None

    slot._stage_delivery_pending = 1
    assert slot.turn_running is False
    assert slot.running is True

    slot._plan_cancelled = True
    assert slot.running is False


@pytest.mark.asyncio
async def test_exact_preconsumption_recovery_finishes_before_stage_advance(tmp_path, monkeypatch):
    """A queued exact retry completes under the same guarded stage controller."""
    from kiro_crew.dashboard.chat import _stage_loop
    from kiro_crew.dashboard.chat_utils import SYNTHETIC_RECOVERY_KIND

    recovery = "[SYSTEM] retry the unconsumed stage input"
    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("preconsumption-recovery", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._auto_run = True
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        callback = kwargs.get("_on_consumed")
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1-failed")
            retry_id = _slot.queue_insert(
                0,
                recovery,
                kind=SYNTHETIC_RECOVERY_KIND,
                on_consumed=callback,
            )
            _slot._stage_delivery_retry_queue_id = retry_id
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            if callable(callback):
                callback(True)
            return
        raise AssertionError(f"unexpected direct turn: {message[:80]}")

    async def _start_retry(_state, _slot):
        entry = _slot.queue_pop(0)
        assert entry["content"] == recovery
        order.append("stage-1-recovery")
        callback = entry.get("_on_consumed")
        if callable(callback):
            callback(True)
        _slot.task = asyncio.create_task(asyncio.sleep(0))
        return True

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_retry,
    )

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert order == ["stage-1-failed", "stage-1-recovery", "stage-2"]


@pytest.mark.asyncio
async def test_consumed_soft_stop_continues_stage_before_capture(tmp_path, monkeypatch):
    """Go after a consumed soft Stop continues Stage N before Stage N+1."""
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.chat_orchestrator import api_chat_plan_action
    from kiro_crew.dashboard.chat_utils import _MANUAL_RESUME_MSG

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("consumed-soft-stop", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    stage_started = asyncio.Event()
    order: list[str] = []

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        callback = kwargs.get("_on_consumed")
        if "Execute Stage 1 of 2 now" in message:
            order.append("stage-1")
            if callable(callback):
                callback(True)
            stage_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return
        if message == _MANUAL_RESUME_MSG:
            order.append("stage-1-continuation")
            if callable(callback):
                callback(True)
            return
        if "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
            if callable(callback):
                callback(True)
            return
        raise AssertionError(f"unexpected turn: {message[:80]}")

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        def __init__(self, action: str) -> None:
            self.action = action

        async def json(self):
            return {"action": self.action}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _mock_run_chat)

    response = await api_chat_plan_action(_PlanRequest("go all"))
    assert response.status == 200
    first_controller = slot._stage_controller_task
    assert first_controller is not None
    await asyncio.wait_for(stage_started.wait(), timeout=1)

    async def _soft_stop(_key, *, force=False, preserve_queue=False, on_soft=None, **_kwargs):
        assert force is False
        assert preserve_queue is True
        child = slot.task
        assert child is not None and child is not first_controller
        child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        if on_soft is not None:
            await on_soft()
        return "soft"

    state.sessions.stop_turn = _soft_stop
    assert await stop_slot_turn(state, slot) == {"ok": True}
    await asyncio.gather(first_controller, return_exceptions=True)
    assert order == ["stage-1"]
    assert slot._stage_delivery_pending == 1
    assert slot._stage_delivery_consumed is True

    response = await api_chat_plan_action(_PlanRequest("go"))
    assert response.status == 200
    resumed_controller = slot._stage_controller_task
    assert resumed_controller is not None
    await asyncio.wait_for(resumed_controller, timeout=5)

    assert order == ["stage-1", "stage-1-continuation", "stage-2"]


@pytest.mark.asyncio
async def test_hard_stop_requeues_consumed_stage_continuation_before_advance(tmp_path, monkeypatch):
    """Go after a hard-stopped recovery continues Stage N before Stage N+1."""
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn
    from kiro_crew.dashboard.chat_orchestrator import _stage_loop, api_chat_plan_action
    from kiro_crew.dashboard.chat_utils import _MANUAL_RESUME_MSG, SYNTHETIC_RECOVERY_KIND

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("consumed-hard-stop", mode="orchestrator")
    slot._stage_titles = ["Collect", "Verify"]
    slot._plan_goal = "Collect then verify"
    slot._stage_delivery_pending = 1
    slot._stage_delivery_consumed = True
    slot._stage_continuation_required = True
    first_controller_started = asyncio.Event()
    order: list[str] = []

    async def _hold_first_controller(*_args, **_kwargs):
        first_controller_started.set()
        await asyncio.Event().wait()

    async def _mock_run_chat(_state, _slot, message, **kwargs):
        if message == _MANUAL_RESUME_MSG:
            order.append("stage-1-continuation")
        elif "Execute Stage 2 of 2 now" in message:
            order.append("stage-2")
        else:
            raise AssertionError(f"unexpected turn: {message[:80]}")
        _mark_consumed(kwargs)

    class _PlanRequest:
        app = {"state": state}
        match_info = {"slot": slot.key}

        async def json(self):
            return {"action": "go"}

        def get(self, _key, default=""):
            return default

    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._stage_loop",
        _hold_first_controller,
    )

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    await asyncio.wait_for(first_controller_started.wait(), timeout=1)
    assert any(
        entry.get("kind") == SYNTHETIC_RECOVERY_KIND and entry.get("content") == _MANUAL_RESUME_MSG
        for entry in slot._queue
    )

    slot._stop_state = "soft_pending"

    async def _hard_stop(_key, *, force=False, on_hard=None, **_kwargs):
        assert force is True
        if on_hard is not None:
            await on_hard()
        return "hard"

    state.sessions.stop_turn = _hard_stop
    assert await stop_slot_turn(state, slot) == {"ok": True}
    assert slot._queue == []

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._stage_loop", _stage_loop)
    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_chat", _mock_run_chat)

    response = await api_chat_plan_action(_PlanRequest())
    assert response.status == 200
    resumed_controller = slot._stage_controller_task
    assert resumed_controller is not None
    await asyncio.wait_for(resumed_controller, timeout=5)

    assert order == ["stage-1-continuation", "stage-2"]


def test_stage_delivery_quiescence_has_one_complete_predicate():
    """Every stage-delivery kind and counter participates in one settle loop."""
    from kiro_crew.dashboard.chat_orchestrator import _settle_stage_delivery, _stage_loop
    from kiro_crew.dashboard.chat_utils import (
        STAGE_DELIVERY_KINDS,
        SUBAGENT_COMPLETION_KIND,
        SYNTHETIC_RECOVERY_KIND,
    )

    assert STAGE_DELIVERY_KINDS == frozenset({SUBAGENT_COMPLETION_KIND, SYNTHETIC_RECOVERY_KIND})
    settle_source = inspect.getsource(_settle_stage_delivery)
    assert 'entry.get("kind") in STAGE_DELIVERY_KINDS' in settle_source
    for counter in (
        "slot._subagent_deliveries_inflight",
        "slot._synthetic_recovery_inflight",
    ):
        assert counter in settle_source
    assert "_sa_rounds" not in inspect.getsource(_stage_loop)


def test_running_guard_never_dereferences_slot_task():
    """A busy boundary may have no live task, so running guards cannot use one."""
    dashboard_root = Path(__file__).parents[1] / "src" / "kiro_crew" / "dashboard"
    readers: list[str] = []
    offenders: list[str] = []

    def _is_slot_attr(node: ast.AST, attr: str, *, load_only: bool = False) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "slot"
            and node.attr == attr
            and (not load_only or isinstance(node.ctx, ast.Load))
        )

    for path in sorted(dashboard_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _is_slot_attr(node, "running", load_only=True):
                readers.append(f"{path.relative_to(dashboard_root)}:{node.lineno}")
            if not isinstance(node, ast.If) or not any(
                _is_slot_attr(part, "running", load_only=True) for part in ast.walk(node.test)
            ):
                continue
            body = ast.Module(body=node.body, type_ignores=[])
            task_reads = [
                part for part in ast.walk(body) if _is_slot_attr(part, "task", load_only=True)
            ]
            offenders.extend(
                f"{path.relative_to(dashboard_root)}:{part.lineno}" for part in task_reads
            )

    assert readers, "the audit did not inspect any slot.running readers"
    assert offenders == [], (
        "slot.running includes pending boundaries with no live task; guard task "
        f"dereferences with slot.turn_running instead: {offenders}"
    )


def test_consumed_stage_resume_obligation_survives_prompt_consumption(tmp_path):
    """A consumed prompt still owes completion before the stage can advance."""
    from kiro_crew.dashboard.chat_orchestrator import _queue_consumed_stage_resume

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("resume-consumption", mode="orchestrator")
    slot._stage_delivery_pending = 1
    slot._stage_delivery_consumed = True
    slot._stage_continuation_required = True

    assert _queue_consumed_stage_resume(
        state,
        slot,
        directive_user_origin=True,
    )
    retry_id = slot._queue[0]["id"]
    assert slot._stage_continuation_required is True
    assert slot._stage_delivery_retry_queue_id == retry_id

    on_consumed = slot._queue[0].get("_on_consumed")
    if callable(on_consumed):
        on_consumed(True)

    assert slot._stage_continuation_required is True
    assert slot._stage_delivery_retry_queue_id == retry_id


@pytest.mark.asyncio
async def test_cancelled_plan_discards_retained_stage_retry(tmp_path, monkeypatch):
    """Cancel revokes a queued stage retry instead of dispatching it as chat."""
    from kiro_crew.dashboard.chat_orchestrator import _stage_loop
    from kiro_crew.dashboard.chat_utils import SYNTHETIC_RECOVERY_KIND

    state = _make_state(tmp_path)
    state.subagents = _StageDeliveryManager()
    slot = state.get_or_create_slot("cancelled-stage-retry", mode="orchestrator")
    slot._stage_titles = ["Collect"]
    slot._plan_goal = "Collect once"
    slot._auto_run = True
    retry = "[SYSTEM] retry the canceled stage"
    started: list[str] = []

    async def _cancel_with_retry(_state, _slot, _message, **kwargs):
        retry_id = _slot.queue_insert(
            0,
            retry,
            kind=SYNTHETIC_RECOVERY_KIND,
            on_consumed=kwargs.get("_on_consumed"),
        )
        _slot._stage_delivery_retry_queue_id = retry_id
        _slot._plan_cancelled = True
        assert _slot._orch_tracker is not None
        _slot._orch_tracker.stop()

    async def _start_queued(_state, _slot):
        entry = _slot.queue_pop(0)
        started.append(entry["content"])
        return True

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _cancel_with_retry)
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_orchestrator._start_next_queued_turn",
        _start_queued,
    )

    await asyncio.wait_for(_stage_loop(state, slot, auto_run=True), timeout=5)

    assert started == []
    assert not any(entry.get("kind") == SYNTHETIC_RECOVERY_KIND for entry in slot._queue)
