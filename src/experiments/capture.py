"""Adapter: turn a finished hook-based run into the standard record.

Nothing is parsed here. ``build_metrics`` already merges the hook events with
the agent's transcript into turns, matched tool calls, tokens and the model;
this module only reshapes that into ``record.json`` and notes, in
``provenance``, which parts the source could not show.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .harness import TRANSCRIPT_SNAPSHOT, ToolCall, build_metrics, load_events, load_manifest
from .record import Experiment, Record, Source, ToolCallRecord, TurnRecord, save_record

PLATFORMS = {"claude": "claude-code", "codex": "codex-cli"}


def _tool_call(call: ToolCall) -> ToolCallRecord:
    # the harness uses "pending" for a call whose end event never arrived
    status = call.status if call.status in ("ok", "error") else "unknown"
    return ToolCallRecord(
        name=call.name, input_preview=call.preview, status=status, duration_ms=call.duration_ms
    )


def record_from_run(run_dir: str | Path) -> Record:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    if manifest.source_kind != "hooks":
        # an imported run has no hook events; rebuilding would wipe its record
        raise ValueError(f"{manifest.run_id} was imported; its record.json was written by `harness import`")
    events = load_events(run_dir)
    metrics = build_metrics(run_dir)

    hooked = any(e.source != "human" for e in events)
    has_text = any(t.response for t in metrics.turn_records)
    has_tokens = metrics.tokens.total() > 0
    # Hooks see every tool call with its outcome; a parsed transcript lists every
    # call too (status "unknown"). With neither, the calls are genuinely unseen.
    calls_known = hooked or has_text or has_tokens

    turns: list[TurnRecord] = []

    def add(role: str, text: str, timestamp: str | None, **extra) -> None:
        turns.append(TurnRecord(n=len(turns) + 1, role=role, text=text, timestamp=timestamp, **extra))

    for t in metrics.turn_records:
        if t.prompt:
            add("user", t.prompt, t.started_at)
        add(
            "assistant",
            t.response,
            t.ended_at,
            tool_calls=[_tool_call(c) for c in t.calls] if calls_known else None,
            tokens=t.tokens.as_dict() if has_tokens else None,
        )

    # Keep a copy of the transcript: the agent's own copy can be rotated or deleted.
    snapshot_file = run_dir / TRANSCRIPT_SNAPSHOT
    source_file = Path(metrics.transcript_path) if metrics.transcript_path else None
    if source_file and source_file.is_file() and source_file.resolve() != snapshot_file.resolve():
        shutil.copy2(source_file, snapshot_file)

    return Record(
        run_id=manifest.run_id,
        experiment=Experiment(**manifest.experiment),
        source=Source(
            kind="hooks",
            platform=PLATFORMS.get(manifest.agent, manifest.agent),
            model=metrics.model,
            snapshot_path=TRANSCRIPT_SNAPSHOT if snapshot_file.is_file() else None,
        ),
        turns=turns,
        provenance={
            "assistant_text": "auto" if has_text else "unavailable",
            "tool_calls": "auto" if calls_known else "unavailable",
            "tokens": "auto" if has_tokens else "unavailable",
            "model": "auto" if metrics.model else "unavailable",
        },
    )


def write_record(run_dir: str | Path) -> Record:
    record = record_from_run(run_dir)
    save_record(run_dir, record)
    return record
