"""Summarise the decision-preview shadow log.

The oracle seam appends one JSON object per call to
``<data home>/decisions/decisions-YYYYMMDD.jsonl`` (see
:mod:`kiro_crew.decisions.log`, which owns the writer). This module is the
reader behind ``kirocrew decisions report``: it turns those lines into per
``(point, impl)`` rows -- agreement with the shipped baseline, abstain and error
share, latency percentiles, spend -- plus a calibration curve that buckets each
call by the confidence the oracle claimed and shows how often that confidence
was right.

Two properties are deliberate:

* **The reader is tolerant.** A shadow log is written from a fire-and-forget
  task, so a torn tail or a line from an older schema is expected rather than
  exceptional. Every unusable line is skipped and counted, and the count is
  reported, so a run whose log is half garbage says so instead of either
  crashing or quietly averaging fewer rows than the caller thinks.
* **It parses the files itself.** ``report`` is the one consumer that must keep
  working against logs written by an earlier version of the writer, so it reads
  the JSONL directly and depends on nothing the writer exports.

The report never decides anything. It is the evidence a human reads before
promoting a point off the ``shadow`` arm.
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import stat
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any

# One file per UTC day, so a report over a window opens only the days it needs.
LOG_DIR_NAME = "decisions"
LOG_STEM_PREFIX = "decisions-"
LOG_SUFFIX = ".jsonl"
_LOG_NAME_RE = re.compile(r"^decisions-(\d{8})\.jsonl$")

# Lower edges of the calibration buckets. Below 0.5 a two-way answer carries no
# claim at all, so anything under the first edge lands in the first bucket
# rather than earning one of its own.
CALIBRATION_EDGES: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)

_RELATIVE_SINCE_RE = re.compile(r"^(\d+)([smhdw])$")
_RELATIVE_UNITS = {
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
}

# A single line is bounded so a corrupt or hostile log cannot be read whole into
# memory. The writer's rows are a few hundred bytes; a megabyte is not one.
_MAX_LINE_CHARS = 1_000_000

# Open flags for a day-file. ``O_NOFOLLOW`` refuses a symlink and ``O_NONBLOCK``
# keeps a FIFO from blocking the open, both only where the platform has them --
# Windows has neither, and ``getattr`` leaves an absent flag out of the mask.
# That is why the check does not rest on them: ``_open_day_file`` calls ``lstat``
# first, which sees a link or a pipe on every platform.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)


class SinceError(ValueError):
    """A ``--since`` value that is neither a relative window nor a timestamp."""


def decisions_dir(home: Path | None = None) -> Path:
    """Directory holding the JSONL logs, honouring ``KIROCREW_HOME``.

    ``home`` is for tests and for a caller that already resolved the data root;
    the default goes through :func:`kiro_crew.config.paths.config_dir`, which is
    where the ``KIROCREW_HOME`` override is applied for every other path.
    """
    if home is not None:
        return home / LOG_DIR_NAME
    from kiro_crew.config.paths import config_dir

    return config_dir() / LOG_DIR_NAME


#: Characters of a rejected ``--since`` echoed back in the error. Long enough to
#: identify what was typed, short enough that a pathological value cannot flood
#: the terminal with the very digits that made it invalid.
_SPEC_ECHO_CHARS = 32


def parse_since(spec: str, *, now: datetime | None = None) -> datetime:
    """Resolve a ``--since`` value to an aware UTC cutoff.

    Accepts a relative window (``30m``, ``12h``, ``7d``, ``2w``) or an ISO 8601
    timestamp. A naive timestamp is read as UTC, because every ``ts`` the writer
    emits is UTC and a local-time reading would silently shift the window.
    """
    text = spec.strip()
    if not text:
        raise SinceError("--since needs a value like 1d, 12h or an ISO timestamp")
    reference = now or datetime.now(timezone.utc)
    match = _RELATIVE_SINCE_RE.match(text.lower())
    if match:
        # The regex bounds the SHAPE, not the magnitude: `\d+` accepts any digit
        # run, so both steps below can raise on a value that matched. `int()`
        # raises ValueError past CPython's 4300-digit conversion limit, and the
        # multiply raises OverflowError past timedelta's 999999999-day ceiling.
        # Both mean the same thing to a caller -- this is not a window -- so both
        # become SinceError here rather than at the CLI, which keeps every caller
        # of parse_since on one contract instead of only the one that has a
        # handler.
        try:
            amount = int(match.group(1))
            return reference - _RELATIVE_UNITS[match.group(2)] * amount
        except (OverflowError, ValueError) as exc:
            # Clipped: this branch exists for absurdly long amounts, so echoing
            # the value whole would put thousands of digits on the terminal and
            # into anything capturing stderr.
            shown = text if len(text) <= _SPEC_ECHO_CHARS else text[:_SPEC_ECHO_CHARS] + "..."
            raise SinceError(
                f"cannot read {shown!r} as a window: the amount is out of range"
            ) from exc
    parsed = _parse_ts(text)
    if parsed is None:
        raise SinceError(
            f"cannot read {spec!r} as a window: use 1d / 12h / 30m / 2w or an ISO timestamp"
        )
    return parsed


def _parse_ts(text: str) -> datetime | None:
    """Parse an ISO 8601 timestamp to aware UTC, or ``None`` if unreadable."""
    candidate = text.strip()
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Row:
    """One usable log line, with the fields the report actually reads."""

    ts: datetime
    point: str
    arm: str
    impl: str
    latency_ms: int | None
    cost_usd: float
    scrubbed: bool
    agree: bool | None
    error: str | None
    max_p: float | None
    answered: bool


@dataclass
class ReadResult:
    """Rows the reader could use, plus what it had to throw away."""

    rows: list[Row] = field(default_factory=list)
    files: int = 0
    skipped_lines: int = 0
    unreadable_files: list[str] = field(default_factory=list)


def _log_day(path: Path) -> str | None:
    match = _LOG_NAME_RE.match(path.name)
    return match.group(1) if match else None


def log_files(directory: Path, *, since: datetime | None = None) -> list[Path]:
    """The day-files that can hold rows at or after ``since``, oldest first.

    The filename day is a *coarse* filter only: a file is kept when its UTC day
    could still contain a row inside the window, and every kept row is then
    filtered on its own ``ts``. Skipping whole days this way is what keeps a
    ``--since 1d`` report from reading a year of logs.
    """
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    cutoff_day = since.astimezone(timezone.utc).strftime("%Y%m%d") if since else None
    kept: list[Path] = []
    for entry in entries:
        day = _log_day(entry)
        if day is None:
            continue
        if cutoff_day is not None and day < cutoff_day:
            continue
        kept.append(entry)
    return kept


def _coerce_row(obj: Any) -> Row | None:
    """Build a :class:`Row` from a decoded line, or ``None`` if unusable.

    A line is unusable when it is not an object, has no ``ts`` the reader can
    parse, or names no ``point`` -- without those three the row cannot be placed
    in a window or a group. Everything else degrades: a missing ``latency_ms``
    drops out of the percentiles instead of failing the line.
    """
    if not isinstance(obj, dict):
        return None
    point = obj.get("point")
    if not isinstance(point, str) or not point:
        return None
    raw_ts = obj.get("ts")
    ts = _parse_ts(raw_ts) if isinstance(raw_ts, str) else None
    if ts is None:
        return None
    answers = obj.get("answers")
    answers_map = answers if isinstance(answers, dict) else {}
    error = obj.get("error")
    arm = obj.get("arm")
    impl = obj.get("impl")
    agree = obj.get("agree")
    return Row(
        ts=ts,
        point=point,
        arm=arm if isinstance(arm, str) else "",
        impl=impl if isinstance(impl, str) else "",
        latency_ms=_as_int(obj.get("latency_ms")),
        cost_usd=_as_float(obj.get("cost_usd")) or 0.0,
        scrubbed=bool(obj.get("scrubbed")),
        agree=agree if isinstance(agree, bool) else None,
        error=error if isinstance(error, str) and error else None,
        max_p=_max_p(answers_map),
        answered=bool(answers_map),
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _max_p(answers: dict[str, Any]) -> float | None:
    """Highest ``p`` across a row's answers, or ``None`` when none carries one.

    The calibration curve asks "when the oracle was this sure, how often was it
    right", and a call is only as confident as its most confident answer, so the
    max is the value that indexes the bucket.
    """
    best: float | None = None
    for answer in answers.values():
        if not isinstance(answer, dict):
            continue
        p = _as_float(answer.get("p"))
        if p is None:
            continue
        if best is None or p > best:
            best = p
    return best


def _open_day_file(path: Path) -> IO[str]:
    """Open *path* for reading, refusing an entry that is not a regular file.

    The writer in ``log.py`` already refuses to follow a symlink when it appends,
    and this makes the reader keep the same promise about the same files: a
    day-file is a regular file, so a symlink, a FIFO or a device planted under a
    log name is refused instead of read. Raises ``OSError`` like ``open`` does,
    which is what the caller already names rather than raises.

    Three checks, because no one of them holds on every platform:

    * ``lstat`` before the open is the only check that sees a symlink where
      ``O_NOFOLLOW`` does not exist -- Windows has no such flag, so the open
      there would follow the link silently. It is also what makes a FIFO a
      refusal on those platforms, before anything can block on it.
    * ``O_NOFOLLOW`` and ``O_NONBLOCK`` close the gap between that ``lstat`` and
      the open, where the name could be swapped for a link or a pipe.
    * ``fstat`` on the descriptor rules on the file that was actually opened,
      which is the one the reader is about to read.
    """
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise OSError(errno.EINVAL, "not a regular file", str(path))
    fd = os.open(str(path), os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_BINARY)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        return os.fdopen(fd, "r", encoding="utf-8", errors="replace")
    except BaseException:
        os.close(fd)
        raise


def read_log(
    directory: Path,
    *,
    since: datetime | None = None,
    point: str | None = None,
) -> ReadResult:
    """Read every usable row in the window, counting what was skipped.

    A file that cannot be opened or read is named rather than raised: one bad
    day must not cost the report the days around it. The handle is opened inside
    the guard (not by a lazy generator the loop drives later), so a path that is
    a directory, a dangling link or any other non-regular entry is caught here
    and not at first read.
    """
    result = ReadResult()
    for path in log_files(directory, since=since):
        try:
            handle = _open_day_file(path)
        except OSError:
            result.unreadable_files.append(path.name)
            continue
        result.files += 1
        try:
            with handle:
                while True:
                    line = handle.readline(_MAX_LINE_CHARS + 1)
                    if not line:
                        break
                    if len(line) > _MAX_LINE_CHARS:
                        result.skipped_lines += 1
                        while line and not line.endswith("\n"):
                            line = handle.readline(_MAX_LINE_CHARS + 1)
                        continue
                    _consume_line(line, result, since=since, point=point)
        except OSError:
            result.unreadable_files.append(path.name)
    result.rows.sort(key=lambda row: row.ts)
    return result


def _consume_line(
    line: str,
    result: ReadResult,
    *,
    since: datetime | None,
    point: str | None,
) -> None:
    """Decode one line into ``result``, or count it as skipped.

    A row filtered out by the window or the point filter is NOT counted as
    skipped: "outside what you asked for" and "unreadable" are different facts
    and the footer reports the second one.
    """
    text = line.strip()
    if not text:
        return
    try:
        decoded = json.loads(text)
    except ValueError:
        result.skipped_lines += 1
        return
    row = _coerce_row(decoded)
    if row is None:
        result.skipped_lines += 1
        return
    if since is not None and row.ts < since:
        return
    if point is not None and row.point != point:
        return
    result.rows.append(row)


def percentile(values: Sequence[int | float], fraction: float) -> float | None:
    """Nearest-rank percentile, or ``None`` for an empty sample.

    Nearest-rank rather than interpolating: every value is a latency that was
    actually observed, and a p95 a caller can find in the log is easier to act
    on than one that sits between two measurements.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    index = min(max(rank, 1), len(ordered)) - 1
    return float(ordered[index])


def bucket_label(p: float) -> str:
    """Name the calibration bucket ``p`` falls in."""
    for edge in reversed(CALIBRATION_EDGES):
        if p >= edge:
            return f"{edge:.1f}-{min(edge + 0.1, 1.0):.1f}"
    first = CALIBRATION_EDGES[0]
    return f"{first:.1f}-{first + 0.1:.1f}"


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def summarise_group(rows: Sequence[Row]) -> dict[str, Any]:
    """Aggregate one ``(point, impl)`` group into the reported numbers."""
    judged = [row for row in rows if row.agree is not None]
    agreed = [row for row in judged if row.agree]
    errors = [row for row in rows if row.error is not None]
    # An abstain is a call that came back with no error and no answer: the gate
    # or the implementation declined, which is a different event from a failure
    # and has to stay separable from it.
    abstains = [row for row in rows if row.error is None and not row.answered]
    latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
    return {
        "n": len(rows),
        "judged": len(judged),
        "agree_rate": _rate(len(agreed), len(judged)),
        "abstain_rate": _rate(len(abstains), len(rows)),
        "error_rate": _rate(len(errors), len(rows)),
        "abstains": len(abstains),
        "errors": len(errors),
        "scrubbed": sum(1 for row in rows if row.scrubbed),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "cost_usd": round(sum(row.cost_usd for row in rows), 6),
        "arms": sorted({row.arm for row in rows if row.arm}),
        "calibration": _calibration(rows),
    }


def _calibration(rows: Iterable[Row]) -> list[dict[str, Any]]:
    """Agreement per confidence bucket: the calibration curve, as data.

    Only rows carrying both a ``p`` and a non-null ``agree`` can say anything
    about calibration, so ``n`` here is that intersection and is normally
    smaller than the group's ``n``.
    """
    members_by_bucket: dict[str, list[Row]] = {}
    for row in rows:
        if row.max_p is None or row.agree is None:
            continue
        members_by_bucket.setdefault(bucket_label(row.max_p), []).append(row)
    buckets: list[dict[str, Any]] = []
    for edge in CALIBRATION_EDGES:
        label = bucket_label(edge)
        members = members_by_bucket.get(label, [])
        buckets.append(
            {
                "bucket": label,
                "n": len(members),
                "agree_rate": _rate(sum(1 for row in members if row.agree), len(members)),
            }
        )
    return buckets


def build_report(
    result: ReadResult,
    *,
    since: datetime | None,
    point: str | None,
    provider_endpoint: str | None,
) -> dict[str, Any]:
    """The whole report as plain data -- the payload behind ``--json``."""
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in result.rows:
        groups.setdefault((row.point, row.impl or "-"), []).append(row)
    summaries = [
        {"point": key[0], "impl": key[1], **summarise_group(rows)}
        for key, rows in sorted(groups.items())
    ]
    window = {
        "since": since.isoformat() if since else None,
        "first_ts": result.rows[0].ts.isoformat() if result.rows else None,
        "last_ts": result.rows[-1].ts.isoformat() if result.rows else None,
    }
    return {
        "rows": len(result.rows),
        "files": result.files,
        "skipped_lines": result.skipped_lines,
        "unreadable_files": result.unreadable_files,
        "point_filter": point,
        "provider_endpoint": provider_endpoint,
        "window": window,
        "groups": summaries,
    }


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def render_table(rows: Sequence[Sequence[str]]) -> list[str]:
    """Left-aligned fixed-width table with a dashed rule under the header."""
    if not rows:
        return []
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    return [line(rows[0]), "  ".join("-" * width for width in widths)] + [
        line(row) for row in rows[1:]
    ]


_HEADER: tuple[str, ...] = (
    "POINT",
    "IMPL",
    "N",
    "AGREE",
    "JUDGED",
    "ABSTAIN",
    "ERROR",
    "SCRUBBED",
    "P50",
    "P95",
    "COST",
)


def render_text(report_payload: dict[str, Any]) -> str:
    """Render the report for a terminal.

    The empty case is one friendly line rather than an empty table: a log with
    no rows is the normal state of a preview nobody has turned on yet, so it is
    not an error and should not read like one.
    """
    groups = report_payload["groups"]
    endpoint = report_payload.get("provider_endpoint")
    endpoint_header = f"NON-DEFAULT ENDPOINT: {endpoint}" if endpoint else ""
    if not groups:
        empty = _render_empty(report_payload)
        return f"{endpoint_header}\n\n{empty}" if endpoint_header else empty
    table: list[Sequence[str]] = [_HEADER]
    for group in groups:
        table.append(
            (
                group["point"],
                group["impl"],
                str(group["n"]),
                _fmt_rate(group["agree_rate"]),
                str(group["judged"]),
                _fmt_rate(group["abstain_rate"]),
                _fmt_rate(group["error_rate"]),
                str(group["scrubbed"]),
                _fmt_ms(group["latency_p50_ms"]),
                _fmt_ms(group["latency_p95_ms"]),
                f"{group['cost_usd']:.5f}",
            )
        )
    lines = render_table(table)
    if endpoint_header:
        lines[:0] = [endpoint_header, ""]
    for group in groups:
        lines.append("")
        lines.append(f"Calibration - {group['point']} / {group['impl']} (confidence vs agreement)")
        curve: list[Sequence[str]] = [("CONFIDENCE", "N", "AGREE")]
        for bucket in group["calibration"]:
            curve.append((bucket["bucket"], str(bucket["n"]), _fmt_rate(bucket["agree_rate"])))
        lines.extend(render_table(curve))
    lines.append("")
    lines.append(_footer(report_payload))
    return "\n".join(lines)


def _render_empty(report_payload: dict[str, Any]) -> str:
    """One line saying why there is nothing, and what would produce something."""
    if report_payload["point_filter"]:
        return (
            f"No decision-preview rows for point {report_payload['point_filter']!r} in this "
            "window. Drop --point or widen --since to see the rest."
        )
    if report_payload["files"] == 0 and not report_payload["unreadable_files"]:
        return (
            "No decision-preview log yet. Set decisions.preview true and give a point an "
            "arm of shadow, then run this again."
        )
    return f"No decision-preview rows in this window. {_footer(report_payload)}"


def _footer(report_payload: dict[str, Any]) -> str:
    parts = [f"{report_payload['rows']} rows from {report_payload['files']} day-file(s)"]
    window = report_payload["window"]
    if window["first_ts"] and window["last_ts"]:
        parts.append(f"{window['first_ts']} .. {window['last_ts']}")
    if report_payload["skipped_lines"]:
        parts.append(f"{report_payload['skipped_lines']} unreadable line(s) skipped")
    if report_payload["unreadable_files"]:
        parts.append(f"unreadable file(s): {', '.join(report_payload['unreadable_files'])}")
    return "; ".join(parts) + "."


def report(
    *,
    since: datetime | None = None,
    point: str | None = None,
    home: Path | None = None,
    provider_endpoint: str | None = None,
) -> dict[str, Any]:
    """Read the log and return the report payload. The one entry point."""
    result = read_log(decisions_dir(home), since=since, point=point)
    return build_report(result, since=since, point=point, provider_endpoint=provider_endpoint)
