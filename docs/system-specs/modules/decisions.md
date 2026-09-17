# Decision Oracle Seam

Status: implemented, shadow only. No point in this release consumes the `live` arm; see §6.
Owners: `src/kiro_crew/decisions/` (the seam: `gate`, `oracle`, `impl_jev`, `impl_llm`, `log`, `report`), `src/kiro_crew/decisions/points/` (one module per decision point), `src/kiro_crew/cli_decisions.py` (the `kirocrew decisions report` surface), and `src/kiro_crew/config/sections.py` (`DecisionsConfig`).

## 1. Purpose

A decision point is a named place in the product where a judgement already has an answer, and where a second opinion can be requested without that answer changing. `decisions.decide` is the one seam for asking: a caller hands it a point name, the state to judge, and the questions, and receives either answers or `None`.

The seam exists to make a comparison measurable before it is trusted. Each point supplies the baseline its shipped logic produced and its own definition of agreement, so a call yields evidence — agreement, claimed confidence, latency, cost — rather than a behavior change. `test/test_decisions_report.py` pins the reader that turns that evidence into a report.

## 2. The seam contract

**Default off.** `decide` returns `None` when `decisions.preview` is false, and this is the outermost condition: no point configuration is consulted, no state is scrubbed, and no implementation is constructed. A point's own `arm` of `off` returns `None` next, and a session outside the point's `bucket` returns `None` after that. Each of the three is independently sufficient, so no single misconfigured value can turn a point on.

**Shadow returns `None`.** On the `shadow` arm the seam writes its log line and then returns `None` regardless of what the implementation answered. A caller therefore cannot distinguish shadow from off by its return value, which is what makes the arm safe to enable: a call site that already handles `None` handles shadow with no further change. This is the load-bearing invariant of the whole feature — a shadow arm that returned answers would let a point act on an unvalidated judge by accident. The loader warns on `arm: live` because no point consumes a live answer in this release; it preserves the value so a future release cannot silently rewrite operator intent.

**LLM by default.** A point with no `impl` uses the exercised `llm` comparison lane. `jev` remains an explicit opt-in until its real endpoint mapping can be exercised with an API key. The question vocabulary is closed to `Choice` and `Noul`, the two shapes with shipped callers; an unused wire shape is not part of the seam contract.

**Bounded by `timeout_ms`.** The implementation call is wrapped in the configured timeout. Expiry, a transport failure, a refusal, and an unparseable response are all the same outcome to the caller: `None`, with the reason recorded in the log line's `error`. The seam never retries, because a retry would multiply the latency the timeout exists to bound and would record one call as several.

**Scrub before send.** State is scanned before any request is constructed, by the local `credential_patterns` regex plus both canonical external-surface scanners: `security.redaction.redact_credentials` and `security.redact_exfiltration_urls`. Any warning refuses the whole state. The credential scanners are complementary — only the canonical one matches a bare 40-char secret, the `aws_secret_access_key=` forms, a PEM block and base64-encoded credentials; only the local list matches the generic `sk-` vendor spelling — while the URL scanner catches exfiltration-shaped destinations in LLM- or user-influenced text. A scanner that itself fails also refuses. Nothing is sent, `None` is returned, and the log row distinguishes `scrubbed: credential` from `scrubbed: exfiltration-url` (and scanner-failure variants). The refusal is whole-request, not per-field redaction — a judge's answer over redacted state would be logged as if it had judged the real thing.

**Agreement is the point's own rule.** A point passes `baseline` (what its shipped logic concluded) and, where value equality over a shared question-id set is the wrong test, an `agree` callable that the log applies instead. Two of the three shipped points need one: `skills.select` answers with one skill against a baseline that is a list, so the test is set membership, and `cron.novelty` answers with a probability against a boolean, so the test is a threshold. A point emits only after its baseline is final and representable in the offered answer domain: dedupe waits for the lexical override, cron waits for confirmed delivery, and skill selection skips a chosen skill truncated from its capped menu. Without the custom rules both comparable points would log `agree: null` on every row and be excluded from `agree_rate` by §4's own rule — the report would run and measure nothing. A rule that raises costs the log row, never the turn: it is applied inside the same guard that already protects row construction. `test/test_decisions_integration.py` pins a real `True` and a real `False` for each of the three points, and pins the count reaching `agree_rate`.

**One log line per call.** Every `decide` that passes the gate appends exactly one line, including the calls that timed out, errored, or were scrubbed. Silence in the log therefore means the gate refused, never that a call failed quietly, which is what lets the report's error and abstain rates be read as shares of real traffic.

**A point may ask whether it is armed.** `is_armed(point, session_key=...)` runs gates 1-3 and nothing else — no scrub, no transport, no await. It exists for a hook whose STATE is expensive to build: `skills.select` walks the skill tree and reads each skill's frontmatter, and doing that above the gate meant the default configuration paid for it once per eligible message and then handed the result to a `decide` that refuses on its first line. It is a cost check and not a second gate: `decide` re-runs all five refusals, so skipping it is merely wasteful and it grants nothing. Gates 1-3 have one implementation that both functions call, so the two cannot drift. `test_decisions_points.py::test_skills_select_does_not_enumerate_when_the_point_is_off` pins that the walk sits below the check.

**One question per point where the answers are graded.** The report places a row in a calibration bucket by the highest `p` across its answers (§4), so an answer that no baseline covers and no agreement rule reads would still decide which bucket the row lands in. A point therefore does not ask a question it does not grade.

**Fire and forget at the call site.** A synchronous call site schedules `decide` as a task rather than awaiting it, so the shipped path never waits on the judge even by `timeout_ms`. The consequence the log reader must live with is a torn tail: a process that exits mid-append leaves a partial line.

## 3. Decision boundary

This seam does not decide anything in this release. No point module consumes the return value of `decide`, so an answer cannot reach a skill list, a dedupe verdict, or a cron delivery. A log line is not evidence that the seam influenced the run it appears in.

The seam is also not an authorization boundary. `decisions.preview` and the per-point arms are ordinary configuration, so any principal that can write `config.json` can arm a point, and arming a point is what causes state to leave the machine. The privacy property the feature rests on is the narrowness of what each point puts in `state`, plus the scrub, not a permission check inside `decide`. A `secret://` API-key reference is resolved only for the default provider endpoint; a custom endpoint must use a literal key already present in the same config, so changing the endpoint cannot disclose a vault entry.

## 4. Report reader

`decisions.report` reads the JSONL day-files directly and imports nothing from `decisions.log`. This is deliberate: the reader is the one consumer that must keep working against lines written by an earlier version of the writer, so it depends on the file format rather than on the writer's exports.

**Tolerance is bounded and counted.** A line that is not a JSON object, or that carries no parseable `ts` or no `point`, is skipped and counted in `skipped_lines`; the count reaches the rendered footer. Every other field degrades instead of failing the line — a row with no `latency_ms` drops out of the percentile sample rather than being read as zero. `TestTolerantReader` pins the skip-and-count behavior on each shape, including a torn final line; `TestLatency.test_a_row_without_a_latency_drops_out_of_the_sample` pins the degradation.

A file that cannot be opened or read is named in `unreadable_files` rather than raised, so one bad day-file does not cost the report the days around it. `TestTolerantReader.test_a_subdirectory_is_not_read_as_a_log` pins that a path shaped like a day-file but not readable as one is reported, not fatal.

**The reader and the writer promise the same thing about the files.** A day-file is a regular file this install wrote, so the reader refuses anything else under a log name — a symlink, a FIFO, a device — mirroring the writer's own `O_NOFOLLOW` append in `decisions.log`. Without that pairing the writer refuses to follow a symlink and the reader follows it, which is the kind of split that survives because each half looks reasonable alone.

**No single check holds on every platform, so `_open_day_file` makes three.** `lstat` runs before the open and is the only one that sees a link where `O_NOFOLLOW` does not exist — Windows has neither that flag nor `O_NONBLOCK`, and the open there would follow the link silently; it is also what keeps a FIFO from blocking that open. The two flags, where the platform has them, close the gap between the `lstat` and the open. `fstat` on the descriptor rules on the file actually opened. `TestTolerantReader` pins the symlink and FIFO refusals, and `test_a_symlink_is_refused_where_the_platform_has_no_o_nofollow` sets both flags to `0` so the Windows path is exercised on every host: it fails if the `lstat` is removed.

**The window is filtered twice.** `log_files` drops whole day-files whose filename day falls entirely before the cutoff, and every kept row is then filtered on its own `ts`. The filename is a coarse index only; correctness comes from the per-row check. A row dropped by either filter is not counted as skipped, because "outside what you asked for" and "unreadable" are different facts. `TestMultipleDays` pins both filters and the distinction.

**Grouping and rates.** Rows group by `(point, impl)`. `agree_rate` is computed over rows whose `agree` is not null and is `None` when that set is empty, so a point that could never compare reads as absent rather than as zero agreement. An abstain (no error, no answers) is counted separately from an error, because a judge declining and a judge failing are different findings. `TestRates` pins each.

**Calibration.** Each row is placed in a bucket by the highest `p` across its answers — a call is only as confident as its most confident answer — and the bucket's agree rate is computed over rows carrying both a `p` and a non-null `agree`. Every bucket appears in the output even when empty, because the curve is read as a shape and an omitted row would read as a gap in the data rather than a gap in the traffic. `TestCalibration` pins the edges, the max-p selection, and the empty-bucket presence.

Percentiles are nearest-rank, so every reported latency is one that was actually observed and can be found in the log.

## 5. CLI surface

`kirocrew decisions report` is argument handling and output only; the reading and the arithmetic are `decisions.report` functions callable without argv. An empty log exits 0 with one line naming the two switches that would produce rows — a preview nobody enabled is the normal state, and a non-zero code there would make "not turned on" indistinguishable from "the reader broke". An unreadable `--since` exits 2 rather than defaulting to a window the caller did not ask for. `TestCli` pins both codes.

The command is registered through `cli_help.add_command`, which refuses a name absent from the help taxonomy, so it cannot be reachable while unlisted.

## 6. Deferred live arm

No point consumes `live` in this release, and the value is accepted by the schema only so that enabling it later is a configuration change rather than a schema migration.

Promotion is deferred per point, not feature-wide, because the blocker differs. `skills.select` is reached from a synchronous assembly path, so consuming an answer there requires that path to acquire a latency budget first; a fire-and-forget task cannot return one. `skills.dedupe` and `cron.novelty` are already asynchronous, so their blocker is evidential rather than structural: agreement alone does not justify promotion, since a judge can agree often while its confidence carries no information, and a point acting on an uninformative confidence has no safe threshold to act at. The calibration curve in §4 is the artifact that answers that question, and it needs traffic that only a shadow arm produces.
