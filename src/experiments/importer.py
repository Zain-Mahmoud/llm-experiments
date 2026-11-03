"""Adapter: turn a transcript from anywhere (a web chat's text, a pasted
conversation, a Markdown file) into the standard record.

The source's own text is saved as a snapshot next to the record, so the record
never depends on a share link staying alive. A web page cannot show tokens or
(usually) tool calls, so those are marked ``unavailable``, not left looking like zero.
"""

from __future__ import annotations

import re
from pathlib import Path

from .harness import RunManifest, iso, make_run_dir, save_manifest
from .record import Experiment, Record, Source, TurnRecord, save_record

SNAPSHOT_NAME = "source.txt"

# Speaker labels, normalised to a role. "You said:" / "ChatGPT said:" are what
# copying a ChatGPT page gives; "User:" / "Assistant:" is the usual Markdown form.
USER_LABELS = {"user", "you", "human"}
ASSISTANT_LABELS = {"assistant", "claude", "chatgpt", "gemini"}
TURN_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\**(" + "|".join(sorted(USER_LABELS | ASSISTANT_LABELS)) + r")(?:\s+said)?\**\s*:\s*\**\s*(.*)$",
    re.IGNORECASE,
)


def parse_transcript_text(text: str) -> list[tuple[str, str]]:
    """Return [(role, text), ...]. Lines before the first speaker label are ignored.
    A body line that itself starts with a label such as "You:" is taken as a new turn."""
    turns: list[tuple[str, list[str]]] = []  # (role, lines); joined once at the end
    # a byte-order mark (Windows editors, some copy tools) would hide the first label
    for line in text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = TURN_RE.match(line)
        if m:
            role = "user" if m.group(1).lower() in USER_LABELS else "assistant"
            turns.append((role, [m.group(2)]))
        elif turns:
            turns[-1][1].append(line)
    return [(role, "\n".join(lines).strip()) for role, lines in turns]


def turn_warnings(record: Record) -> list[str]:
    """Things that usually mean the text was split wrongly. The parser cannot tell a
    real speaker label from a quoted one, so these are shown for the person to check."""
    warnings = []
    for prev, cur in zip(record.turns, record.turns[1:]):
        if prev.role == cur.role:
            warnings.append(f"turns {prev.n} and {cur.n} are both {cur.role} (check the text was split correctly)")
    warnings += [f"turn {t.n} ({t.role}) has no visible text" for t in record.turns if not t.text]
    return warnings


def import_transcript(
    text: str,
    *,
    experiment: dict[str, str] | None = None,
    platform: str | None = None,
    model: str | None = None,
    url: str | None = None,
) -> tuple[Record, str]:
    """Build a record (without a run id's directory yet) from transcript text.
    Returns the record and the run name used to build the run id."""
    parsed = parse_transcript_text(text)
    if not parsed:
        raise ValueError(
            "no turns found: expected lines starting with 'User:' / 'Assistant:' "
            "(or 'You said:' / 'ChatGPT said:'). Labels need a colon; bare headings are not recognised."
        )

    exp = Experiment(**(experiment or {}))

    kind = "share_link" if url else "paste"
    # text from a captured page is "scraped"; text a person pasted in is "manual"
    how = "scraped" if url else "manual"
    record = Record(
        run_id="",  # filled in once the run directory exists
        experiment=exp,
        source=Source(kind=kind, platform=platform, model=model, url=url, snapshot_path=SNAPSHOT_NAME),
        turns=[TurnRecord(n=i + 1, role=role, text=body) for i, (role, body) in enumerate(parsed)],
        provenance={
            "assistant_text": how,
            "tool_calls": "unavailable",
            "tokens": "unavailable",
            "model": how if model else "unavailable",
        },
    )
    return record, "-".join(p for p in (platform, exp.run or exp.scenario) if p) or "import"


def create_imported_run(text: str, **kwargs) -> tuple[Path, Record]:
    """Create a run from transcript text; the text itself is the snapshot."""
    record, name = import_transcript(text, **kwargs)
    return write_imported_run(record, name, text), record


def write_imported_run(record: Record, name: str, snapshot: str) -> Path:
    """Create ``runs/<id>/`` holding the snapshot, a manifest and ``record.json``.

    ``record.source.snapshot_path`` says which file name the snapshot is saved under.
    """
    run_dir = make_run_dir(name)
    record.run_id = run_dir.name

    (run_dir / (record.source.snapshot_path or SNAPSHOT_NAME)).write_text(snapshot, encoding="utf-8")
    save_manifest(
        RunManifest(
            run_id=run_dir.name,
            name=name,
            agent=record.source.platform or "web",
            access="n/a",
            workspace=str(Path.cwd()),
            run_dir=str(run_dir),
            experiment=record.experiment.model_dump(exclude_none=True),
            source_kind="import",
            status="finished",
            ended_at=iso(),
        )
    )
    save_record(run_dir, record)
    return run_dir
