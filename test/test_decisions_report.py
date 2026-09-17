"""Tests for the decision-preview report reader and its CLI surface.

The reader's whole job is to be trustworthy about a log it does not control: the
writer appends from a fire-and-forget task, so a torn tail, a line from an older
schema and a whole missing day are all normal inputs. Each test below therefore
pins one of two things — a number the report must compute correctly, or a
malformed input it must survive while *saying* that it skipped something. A
tolerant reader that silently drops rows is worse than a strict one, because the
rates it prints look complete.

Fixtures write real JSONL files under ``tmp_path`` and the reader is pointed at
them with ``home=``, so nothing here touches the developer's own log.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from conftest import requires_symlinks

from kiro_crew.decisions import report as mod

NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


def _row(
    *,
    ts: datetime = NOW,
    point: str = "skills.select",
    arm: str = "shadow",
    impl: str = "jev",
    latency_ms: int = 200,
    cost_usd: float = 0.0001,
    scrubbed: bool = False,
    answers: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    agree: bool | None = True,
    error: str | None = None,
) -> dict[str, Any]:
    """One log line in the writer's shape, with the defaults a healthy call has."""
    return {
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "point": point,
        "arm": arm,
        "impl": impl,
        "session": "0123456789ab",
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "in_tokens": 1900,
        "scrubbed": scrubbed,
        "answers": (
            {"pick": {"value": "a", "p": 0.8, "confidence": "high"}} if answers is None else answers
        ),
        "baseline": baseline,
        "agree": agree,
        "error": error,
    }


def _write_log(home: Path, day: str, lines: list[Any]) -> Path:
    """Write ``decisions-<day>.jsonl``; a plain string is written verbatim."""
    directory = home / mod.LOG_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{mod.LOG_STEM_PREFIX}{day}{mod.LOG_SUFFIX}"
    body = "".join(
        line + "\n" if isinstance(line, str) else json.dumps(line) + "\n" for line in lines
    )
    path.write_text(body, encoding="utf-8")
    return path


class TestSinceWindow:
    """``--since`` is the only user input that can silently change every number."""

    @pytest.mark.parametrize(
        "spec,delta",
        [
            ("30m", timedelta(minutes=30)),
            ("12h", timedelta(hours=12)),
            ("1d", timedelta(days=1)),
            ("7d", timedelta(days=7)),
            ("2w", timedelta(weeks=2)),
            ("45s", timedelta(seconds=45)),
        ],
    )
    def test_relative_windows_subtract_from_now(self, spec: str, delta: timedelta) -> None:
        assert mod.parse_since(spec, now=NOW) == NOW - delta

    def test_iso_timestamp_is_taken_as_the_cutoff(self) -> None:
        assert mod.parse_since("2026-09-16T00:00:00Z", now=NOW) == datetime(
            2026, 9, 16, tzinfo=timezone.utc
        )

    def test_a_naive_timestamp_reads_as_utc(self) -> None:
        """Every ``ts`` the writer emits is UTC, so a local reading would shift the window."""
        assert mod.parse_since("2026-09-16T00:00:00", now=NOW).tzinfo == timezone.utc

    def test_an_offset_timestamp_is_normalised_to_utc(self) -> None:
        assert mod.parse_since("2026-09-16T02:00:00+02:00", now=NOW) == datetime(
            2026, 9, 16, tzinfo=timezone.utc
        )

    @pytest.mark.parametrize("spec", ["", "  ", "yesterday", "7", "d7", "1y", "1 d"])
    def test_an_unreadable_window_is_refused_not_guessed(self, spec: str) -> None:
        with pytest.raises(mod.SinceError):
            mod.parse_since(spec, now=NOW)


class TestEmptyLog:
    """A preview nobody enabled is the normal state, not a failure."""

    def test_no_directory_reports_no_rows(self, tmp_path: Path) -> None:
        payload = mod.report(home=tmp_path)
        assert payload["rows"] == 0
        assert payload["groups"] == []
        assert payload["files"] == 0

    def test_the_empty_line_says_how_to_produce_rows(self, tmp_path: Path) -> None:
        text = mod.render_text(mod.report(home=tmp_path))
        assert "\n" not in text.strip()
        assert "decisions.preview" in text and "shadow" in text

    def test_a_point_filter_that_matches_nothing_says_so(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row()])
        text = mod.render_text(mod.report(home=tmp_path, point="cron.novelty"))
        assert "cron.novelty" in text
        assert "--point" in text


class TestGrouping:
    def test_rows_group_by_point_and_impl(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [
                _row(point="skills.select", impl="jev"),
                _row(point="skills.select", impl="llm"),
                _row(point="cron.novelty", impl="jev"),
                _row(point="cron.novelty", impl="jev"),
            ],
        )
        groups = mod.report(home=tmp_path)["groups"]
        assert [(g["point"], g["impl"], g["n"]) for g in groups] == [
            ("cron.novelty", "jev", 2),
            ("skills.select", "jev", 1),
            ("skills.select", "llm", 1),
        ]

    def test_point_filter_keeps_only_that_point(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [_row(point="skills.select"), _row(point="skills.dedupe")],
        )
        groups = mod.report(home=tmp_path, point="skills.dedupe")["groups"]
        assert [g["point"] for g in groups] == ["skills.dedupe"]

    def test_a_row_without_an_impl_still_forms_a_group(self, tmp_path: Path) -> None:
        """An older writer, or a call that failed before choosing, must not vanish."""
        line = _row()
        del line["impl"]
        _write_log(tmp_path, "20260917", [line])
        groups = mod.report(home=tmp_path)["groups"]
        assert [(g["impl"], g["n"]) for g in groups] == [("-", 1)]


class TestRates:
    def test_agree_rate_is_over_judged_rows_only(self, tmp_path: Path) -> None:
        """A null ``agree`` means the point could not compare, which is not a disagreement."""
        _write_log(
            tmp_path,
            "20260917",
            [
                _row(agree=True),
                _row(agree=True),
                _row(agree=False),
                _row(agree=None),
                _row(agree=None),
            ],
        )
        group = mod.report(home=tmp_path)["groups"][0]
        assert group["n"] == 5
        assert group["judged"] == 3
        assert group["agree_rate"] == pytest.approx(2 / 3)

    def test_agree_rate_is_none_when_nothing_was_judged(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row(agree=None), _row(agree=None)])
        group = mod.report(home=tmp_path)["groups"][0]
        assert group["agree_rate"] is None
        assert mod._fmt_rate(None) == "-"

    def test_an_error_row_is_an_error_not_an_abstain(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [
                _row(error="timeout", answers={}, agree=None),
                _row(error=None, answers={}, agree=None),
                _row(),
                _row(),
            ],
        )
        group = mod.report(home=tmp_path)["groups"][0]
        assert (group["errors"], group["abstains"]) == (1, 1)
        assert group["error_rate"] == pytest.approx(0.25)
        assert group["abstain_rate"] == pytest.approx(0.25)

    def test_an_empty_error_string_is_not_an_error(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row(error="")])
        assert mod.report(home=tmp_path)["groups"][0]["errors"] == 0

    def test_scrubbed_rows_are_counted(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [_row(scrubbed=True), _row(scrubbed=True), _row(scrubbed=False)],
        )
        assert mod.report(home=tmp_path)["groups"][0]["scrubbed"] == 2

    def test_cost_is_summed_across_the_group(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [_row(cost_usd=0.00008), _row(cost_usd=0.00012), _row(cost_usd=0.0)],
        )
        assert mod.report(home=tmp_path)["groups"][0]["cost_usd"] == pytest.approx(0.0002)


class TestLatency:
    @pytest.mark.parametrize(
        "values,p50,p95",
        [
            ([100], 100.0, 100.0),
            ([100, 200], 100.0, 200.0),
            (list(range(1, 101)), 50.0, 95.0),
        ],
    )
    def test_nearest_rank_percentiles(self, values: list[int], p50: float, p95: float) -> None:
        assert mod.percentile(values, 0.50) == p50
        assert mod.percentile(values, 0.95) == p95

    def test_an_empty_sample_has_no_percentile(self) -> None:
        assert mod.percentile([], 0.5) is None

    def test_a_row_without_a_latency_drops_out_of_the_sample(self, tmp_path: Path) -> None:
        """A timed-out call may carry no latency; it must not be read as 0 ms."""
        missing = _row(error="timeout")
        del missing["latency_ms"]
        _write_log(tmp_path, "20260917", [_row(latency_ms=100), missing])
        group = mod.report(home=tmp_path)["groups"][0]
        assert group["n"] == 2
        assert group["latency_p50_ms"] == 100.0

    def test_a_group_with_no_latencies_reports_none(self, tmp_path: Path) -> None:
        line = _row()
        del line["latency_ms"]
        _write_log(tmp_path, "20260917", [line])
        group = mod.report(home=tmp_path)["groups"][0]
        assert group["latency_p50_ms"] is None
        assert group["latency_p95_ms"] is None


class TestCalibration:
    @pytest.mark.parametrize(
        "p,label",
        [
            (0.5, "0.5-0.6"),
            (0.59, "0.5-0.6"),
            (0.6, "0.6-0.7"),
            (0.75, "0.7-0.8"),
            (0.89, "0.8-0.9"),
            (0.9, "0.9-1.0"),
            (1.0, "0.9-1.0"),
            (0.2, "0.5-0.6"),
        ],
    )
    def test_bucket_edges(self, p: float, label: str) -> None:
        assert mod.bucket_label(p) == label

    def test_every_bucket_is_present_even_when_empty(self, tmp_path: Path) -> None:
        """The curve is read as a shape, so a gap must show as 0 rows, not as an absent row."""
        _write_log(tmp_path, "20260917", [_row()])
        curve = mod.report(home=tmp_path)["groups"][0]["calibration"]
        assert [b["bucket"] for b in curve] == [
            "0.5-0.6",
            "0.6-0.7",
            "0.7-0.8",
            "0.8-0.9",
            "0.9-1.0",
        ]
        assert [b["n"] for b in curve] == [0, 0, 0, 1, 0]

    def test_agree_rate_is_computed_per_bucket(self, tmp_path: Path) -> None:
        def line(p: float, agree: bool) -> dict[str, Any]:
            return _row(answers={"pick": {"value": "a", "p": p, "confidence": "x"}}, agree=agree)

        _write_log(
            tmp_path,
            "20260917",
            [
                line(0.95, True),
                line(0.95, True),
                line(0.95, False),
                line(0.55, False),
            ],
        )
        curve = {b["bucket"]: b for b in mod.report(home=tmp_path)["groups"][0]["calibration"]}
        assert curve["0.9-1.0"]["agree_rate"] == pytest.approx(2 / 3)
        assert curve["0.5-0.6"]["agree_rate"] == 0.0
        assert curve["0.7-0.8"]["agree_rate"] is None

    def test_the_bucket_follows_the_most_confident_answer(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [
                _row(
                    answers={
                        "a": {"value": "x", "p": 0.55, "confidence": "low"},
                        "b": {"value": "y", "p": 0.93, "confidence": "high"},
                    }
                )
            ],
        )
        curve = {b["bucket"]: b["n"] for b in mod.report(home=tmp_path)["groups"][0]["calibration"]}
        assert curve["0.9-1.0"] == 1
        assert curve["0.5-0.6"] == 0

    def test_a_row_without_a_p_is_absent_from_the_curve(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [_row(answers={"pick": {"value": "a", "confidence": "high"}})],
        )
        group = mod.report(home=tmp_path)["groups"][0]
        assert group["n"] == 1
        assert sum(b["n"] for b in group["calibration"]) == 0

    def test_an_unjudged_row_is_absent_from_the_curve(self, tmp_path: Path) -> None:
        """Calibration needs both a claim and a verdict; a claim alone says nothing."""
        _write_log(tmp_path, "20260917", [_row(agree=None)])
        group = mod.report(home=tmp_path)["groups"][0]
        assert group["n"] == 1
        assert sum(b["n"] for b in group["calibration"]) == 0


class TestTolerantReader:
    """Every unusable line is skipped AND counted, so a bad log cannot look complete."""

    def test_bad_lines_are_skipped_and_counted(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [
                _row(),
                "{not json",
                "",
                "   ",
                '{"ts": "2026-09-17T12:00:00Z"}',
                '"a bare string"',
                "[1, 2, 3]",
                _row(),
            ],
        )
        payload = mod.report(home=tmp_path)
        assert payload["rows"] == 2
        assert payload["skipped_lines"] == 4

    def test_a_row_with_an_unreadable_ts_is_skipped(self, tmp_path: Path) -> None:
        bad = _row()
        bad["ts"] = "the day before yesterday"
        _write_log(tmp_path, "20260917", [bad, _row()])
        payload = mod.report(home=tmp_path)
        assert (payload["rows"], payload["skipped_lines"]) == (1, 1)

    def test_a_torn_final_line_costs_only_that_line(self, tmp_path: Path) -> None:
        directory = tmp_path / mod.LOG_DIR_NAME
        directory.mkdir(parents=True)
        path = directory / "decisions-20260917.jsonl"
        path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row())[:40], encoding="utf-8")
        payload = mod.report(home=tmp_path)
        assert (payload["rows"], payload["skipped_lines"]) == (1, 1)

    def test_an_oversized_line_is_read_and_drained_in_bounded_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_log(
            tmp_path,
            "20260917",
            ["x" * (mod._MAX_LINE_CHARS + 17), _row()],
        )
        body = path.read_text(encoding="utf-8")

        class _BoundedReader(io.StringIO):
            def __iter__(self):
                raise AssertionError("the report must not iterate unbounded lines")

            def readline(self, size: int = -1) -> str:
                assert size == mod._MAX_LINE_CHARS + 1
                return super().readline(size)

        original_open = Path.open

        def _open(candidate: Path, *args: Any, **kwargs: Any):
            if candidate == path:
                return _BoundedReader(body)
            return original_open(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "open", _open)
        payload = mod.report(home=tmp_path)
        assert (payload["rows"], payload["skipped_lines"]) == (1, 1)

    def test_the_skip_count_reaches_the_rendered_footer(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row(), "{oops"])
        assert "1 unreadable line(s) skipped" in mod.render_text(mod.report(home=tmp_path))

    def test_a_file_that_is_not_a_day_log_is_ignored(self, tmp_path: Path) -> None:
        directory = tmp_path / mod.LOG_DIR_NAME
        directory.mkdir(parents=True)
        (directory / "notes.txt").write_text("hello\n", encoding="utf-8")
        (directory / "decisions-latest.jsonl").write_text("{oops\n", encoding="utf-8")
        _write_log(tmp_path, "20260917", [_row()])
        payload = mod.report(home=tmp_path)
        assert (payload["rows"], payload["files"], payload["skipped_lines"]) == (1, 1, 0)

    def test_a_subdirectory_is_not_read_as_a_log(self, tmp_path: Path) -> None:
        directory = tmp_path / mod.LOG_DIR_NAME
        (directory / "decisions-20260916.jsonl").mkdir(parents=True)
        _write_log(tmp_path, "20260917", [_row()])
        payload = mod.report(home=tmp_path)
        assert payload["rows"] == 1
        assert payload["unreadable_files"] == ["decisions-20260916.jsonl"]

    @requires_symlinks
    def test_a_symlink_under_a_log_name_is_not_followed(self, tmp_path: Path) -> None:
        planted = tmp_path / "planted.jsonl"
        planted.write_text(json.dumps(_row(point="planted")) + "\n", encoding="utf-8")
        _write_log(tmp_path, "20260917", [_row()])
        directory = tmp_path / mod.LOG_DIR_NAME
        os.symlink(planted, directory / "decisions-20260916.jsonl")
        payload = mod.report(home=tmp_path)
        assert payload["rows"] == 1
        assert payload["unreadable_files"] == ["decisions-20260916.jsonl"]
        assert [group["point"] for group in payload["groups"]] == ["skills.select"]

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this platform")
    def test_a_fifo_under_a_log_name_is_refused_not_waited_on(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row()])
        directory = tmp_path / mod.LOG_DIR_NAME
        os.mkfifo(directory / "decisions-20260916.jsonl")
        payload = mod.report(home=tmp_path)
        assert payload["rows"] == 1
        assert payload["unreadable_files"] == ["decisions-20260916.jsonl"]

    @requires_symlinks
    def test_a_symlink_is_refused_where_the_platform_has_no_o_nofollow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows has neither flag, so the ``lstat`` must carry the refusal alone."""
        monkeypatch.setattr(mod, "_O_NOFOLLOW", 0)
        monkeypatch.setattr(mod, "_O_NONBLOCK", 0)
        planted = tmp_path / "planted.jsonl"
        planted.write_text(json.dumps(_row(point="planted")) + "\n", encoding="utf-8")
        _write_log(tmp_path, "20260917", [_row()])
        directory = tmp_path / mod.LOG_DIR_NAME
        os.symlink(planted, directory / "decisions-20260916.jsonl")
        payload = mod.report(home=tmp_path)
        assert payload["rows"] == 1
        assert payload["unreadable_files"] == ["decisions-20260916.jsonl"]

    def test_a_regular_day_file_is_still_read(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row(), _row()])
        payload = mod.report(home=tmp_path)
        assert (payload["rows"], payload["files"], payload["unreadable_files"]) == (2, 1, [])

    def test_the_opener_closes_the_descriptor_it_refuses(self, tmp_path: Path) -> None:
        directory = tmp_path / mod.LOG_DIR_NAME
        directory.mkdir(parents=True)
        path = directory / "decisions-20260916.jsonl"
        path.mkdir()
        before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None
        for _ in range(50):
            with pytest.raises(OSError):
                mod._open_day_file(path)
        if before is not None:
            assert len(os.listdir("/proc/self/fd")) <= before + 2

    def test_an_unreadable_file_is_named_in_the_footer(self, tmp_path: Path) -> None:
        directory = tmp_path / mod.LOG_DIR_NAME
        (directory / "decisions-20260916.jsonl").mkdir(parents=True)
        text = mod.render_text(mod.report(home=tmp_path))
        assert "decisions-20260916.jsonl" in text


class TestMultipleDays:
    def test_rows_from_several_days_are_merged_in_time_order(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260915", [_row(ts=NOW - timedelta(days=2))])
        _write_log(tmp_path, "20260916", [_row(ts=NOW - timedelta(days=1))])
        _write_log(tmp_path, "20260917", [_row(ts=NOW)])
        result = mod.read_log(mod.decisions_dir(tmp_path))
        assert [row.ts for row in result.rows] == sorted(row.ts for row in result.rows)
        assert (len(result.rows), result.files) == (3, 3)

    def test_a_day_entirely_before_the_cutoff_is_never_opened(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260915", [_row(ts=NOW - timedelta(days=2))])
        _write_log(tmp_path, "20260917", [_row(ts=NOW)])
        since = NOW - timedelta(hours=6)
        result = mod.read_log(mod.decisions_dir(tmp_path), since=since)
        assert (len(result.rows), result.files) == (1, 1)

    def test_a_row_before_the_cutoff_inside_a_kept_day_is_dropped(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [_row(ts=NOW - timedelta(hours=10)), _row(ts=NOW - timedelta(hours=1))],
        )
        since = NOW - timedelta(hours=6)
        result = mod.read_log(mod.decisions_dir(tmp_path), since=since)
        assert len(result.rows) == 1
        assert result.rows[0].ts == NOW - timedelta(hours=1)

    def test_a_dropped_row_is_not_counted_as_unreadable(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row(ts=NOW - timedelta(hours=10))])
        result = mod.read_log(mod.decisions_dir(tmp_path), since=NOW - timedelta(hours=1))
        assert (len(result.rows), result.skipped_lines) == (0, 0)


class TestPathResolution:
    def test_the_log_lives_under_the_data_home(self, tmp_path: Path, monkeypatch) -> None:
        """``KIROCREW_HOME`` is honoured the same way every other path honours it."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config import paths

        monkeypatch.setattr(paths, "_config_dir_memo", None)
        assert mod.decisions_dir() == tmp_path / "decisions"

    def test_an_explicit_home_wins_over_the_environment(self, tmp_path: Path) -> None:
        assert mod.decisions_dir(tmp_path) == tmp_path / mod.LOG_DIR_NAME


class TestRendering:
    def test_the_table_carries_every_reported_number(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [_row(latency_ms=210, scrubbed=True), _row(latency_ms=190, agree=False)],
        )
        text = mod.render_text(mod.report(home=tmp_path))
        for token in ("POINT", "AGREE", "SCRUBBED", "P95", "COST", "skills.select", "jev"):
            assert token in text
        assert "Calibration" in text
        assert "0.9-1.0" in text

    def test_a_non_default_endpoint_is_prominent_in_text_and_json(self, tmp_path: Path) -> None:
        endpoint = "https://judge.example.test/v1/decide"
        _write_log(tmp_path, "20260917", [_row()])
        payload = mod.report(home=tmp_path, provider_endpoint=endpoint)
        assert payload["provider_endpoint"] == endpoint
        assert mod.render_text(payload).splitlines()[0] == f"NON-DEFAULT ENDPOINT: {endpoint}"

    def test_the_header_rule_matches_the_column_count(self) -> None:
        lines = mod.render_table([("A", "BB"), ("cccc", "d")])
        assert lines[1] == "----  --"

    def test_json_payload_is_serialisable_and_flat(self, tmp_path: Path) -> None:
        _write_log(tmp_path, "20260917", [_row()])
        payload = mod.report(home=tmp_path)
        assert json.loads(json.dumps(payload))["groups"][0]["point"] == "skills.select"

    def test_the_window_bounds_are_the_first_and_last_row(self, tmp_path: Path) -> None:
        _write_log(
            tmp_path,
            "20260917",
            [_row(ts=NOW - timedelta(hours=3)), _row(ts=NOW)],
        )
        window = mod.report(home=tmp_path)["window"]
        assert window["first_ts"].startswith("2026-09-17T09:00")
        assert window["last_ts"].startswith("2026-09-17T12:00")


class TestCli:
    """The CLI adds argument handling and exit codes; those are what is pinned here."""

    def _args(self, **overrides: Any):
        import argparse

        defaults = {"decisions_action": "report", "since": "1d", "point": "", "json": False}
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config import paths

        monkeypatch.setattr(paths, "_config_dir_memo", None)
        return tmp_path

    def test_an_empty_log_exits_zero(self, capsys) -> None:
        from kiro_crew import cli_decisions

        assert cli_decisions.decisions_cmd(self._args()) == 0
        assert "decisions.preview" in capsys.readouterr().out

    def test_a_bad_since_is_refused_with_exit_two(self, capsys) -> None:
        from kiro_crew import cli_decisions

        assert cli_decisions.decisions_cmd(self._args(since="yesterday")) == 2
        assert "yesterday" in capsys.readouterr().err

    def test_json_mode_prints_the_payload(self, _home: Path, capsys, monkeypatch) -> None:
        from kiro_crew import cli_decisions

        endpoint = "https://judge.example.test/v1/decide"
        monkeypatch.setattr(cli_decisions, "_non_default_endpoint", lambda: endpoint)
        _write_log(
            _home,
            datetime.now(timezone.utc).strftime("%Y%m%d"),
            [_row(ts=datetime.now(timezone.utc))],
        )
        assert cli_decisions.decisions_cmd(self._args(json=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["groups"][0]["point"] == "skills.select"
        assert payload["provider_endpoint"] == endpoint

    def test_an_unknown_subcommand_is_a_usage_error(self, capsys) -> None:
        from kiro_crew import cli_decisions

        assert cli_decisions.decisions_cmd(self._args(decisions_action=None)) == 2
        assert "Usage:" in capsys.readouterr().err

    def test_the_command_is_registered_and_listed(self) -> None:
        """``add_command`` refuses an ungrouped name, so this also pins the help entry."""
        import argparse

        from kiro_crew import cli, cli_help

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        cli.register_decisions_parser(sub)
        assert "decisions" in cli_help.SUMMARIES
        args = parser.parse_args(["decisions", "report", "--since", "7d", "--json"])
        assert (args.decisions_action, args.since, args.json) == ("report", "7d", True)


class TestSinceOutOfRange:
    """A window that matches the pattern but cannot be built is still exit 2."""

    @pytest.mark.parametrize(
        "spec, why",
        [
            ("1000000000d", "past timedelta's 999999999-day ceiling -> OverflowError"),
            ("9" * 4400 + "d", "past CPython's 4300-digit int limit -> ValueError"),
        ],
    )
    def test_an_out_of_range_window_is_a_since_error_not_a_traceback(self, spec, why):
        """The regex bounds the SHAPE of a window, not its magnitude.

        ``--since`` is documented to exit 2 on a value it cannot read, and the CLI
        reaches that code by catching ``SinceError`` alone. Both of these MATCH the
        relative-window pattern and then raise inside it, so before this they
        escaped as a raw traceback: an operator typo read as a crash.
        """
        with pytest.raises(mod.SinceError) as caught:
            mod.parse_since(spec)
        assert "out of range" in str(caught.value), why

    def test_a_rejected_window_does_not_echo_thousands_of_digits(self):
        """Name what was typed without reprinting all of it."""
        with pytest.raises(mod.SinceError) as caught:
            mod.parse_since("9" * 4400 + "d")
        message = str(caught.value)
        assert len(message) < 200, "a pathological amount must not flood the message"
        assert "..." in message

    def test_a_normal_window_is_unaffected(self):
        reference = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
        assert mod.parse_since("7d", now=reference) == reference - timedelta(days=7)
