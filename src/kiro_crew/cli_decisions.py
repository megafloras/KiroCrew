"""``kirocrew decisions`` -- read the decision-preview shadow log.

Thin CLI layer: argument handling and output. The reading and the arithmetic
live in :mod:`kiro_crew.decisions.report`, so the same numbers are available to
anything else that wants them without going through argv.

One subcommand today. ``report`` answers the only question a shadow arm exists
to answer: over this window, how often did the oracle agree with the logic that
actually shipped, how confident was it when it did, and what did asking cost.
"""

from __future__ import annotations

import argparse
import json
import sys

from kiro_crew.decisions import report as decisions_report

# Default window. A day is the unit a shadow run is judged in -- long enough to
# cover an overnight soak, short enough that the reader is looking at current
# behavior rather than an average over a config that has since changed.
DEFAULT_SINCE = "1d"


def _non_default_endpoint() -> str | None:
    from kiro_crew.config.loader import KiroCrewConfig
    from kiro_crew.config.sections import DECISION_PROVIDER_ENDPOINT_DEFAULT

    endpoint = str(
        KiroCrewConfig.load().decisions.provider.endpoint or DECISION_PROVIDER_ENDPOINT_DEFAULT
    )
    return endpoint if endpoint != DECISION_PROVIDER_ENDPOINT_DEFAULT else None


def _decisions_report(args: argparse.Namespace) -> int:
    try:
        since = decisions_report.parse_since(args.since)
    except decisions_report.SinceError as exc:
        print(f"kirocrew decisions report: {exc}", file=sys.stderr)
        return 2
    payload = decisions_report.report(
        since=since,
        point=args.point or None,
        provider_endpoint=_non_default_endpoint(),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(decisions_report.render_text(payload))
    # An empty log is the normal state of a preview nobody enabled, so it exits
    # 0 with a friendly line -- a non-zero code here would make "not turned on"
    # indistinguishable from "the reader broke".
    return 0


def decisions_cmd(args: argparse.Namespace) -> int:
    """Dispatch a ``decisions`` subcommand."""
    if args.decisions_action == "report":
        return _decisions_report(args)
    print("Usage: kirocrew decisions report [--since 1d] [--point NAME] [--json]", file=sys.stderr)
    return 2
