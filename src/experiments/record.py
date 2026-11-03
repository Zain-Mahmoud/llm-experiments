"""The standard experiment record: facts only.

Every experiment, whatever tool or platform it came from, ends up as one
``record.json``. Adapters fill in what their source can see; everything else is
left ``None`` and marked in ``provenance`` so a missing value is never mistaken
for zero.

Judgments (comments, labels, outcome codes) do NOT live here. They are written
by the annotation app to a separate ``annotations.json``; see ``annotations.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import __version__
from .harness import iso, read_json, write_json

SCHEMA_VERSION = 1

# How a field got its value. "unavailable" means the source cannot show it,
# which is different from "none happened".
Provenance = Literal["auto", "scraped", "manual", "unavailable"]


class Experiment(BaseModel):
    scenario: str | None = None  # never guessed: a wrong scenario would pick the wrong codebook
    framing: str | None = None
    condition: str | None = None
    run: str | None = None
    researcher: str | None = None


class Source(BaseModel):
    """How the data was captured, which tells you how complete it is."""

    kind: Literal["hooks", "share_link", "paste"]
    platform: str | None = None  # e.g. claude-code, codex-cli, chatgpt, gemini
    model: str | None = None
    url: str | None = None
    captured_at: str = Field(default_factory=iso)
    snapshot_path: str | None = None  # raw page/transcript saved next to the record


class ToolCallRecord(BaseModel):
    name: str
    input_preview: str = ""
    status: Literal["ok", "error", "unknown"] = "unknown"
    duration_ms: float | None = None


class TurnRecord(BaseModel):
    n: int = Field(ge=1)
    role: Literal["user", "assistant", "tool", "system"]
    text: str
    timestamp: str | None = None
    # None = the source cannot show tool calls; [] = it showed that none were used.
    tool_calls: list[ToolCallRecord] | None = None
    tokens: dict[str, int] | None = None


class Record(BaseModel):
    schema_version: int = SCHEMA_VERSION
    harness_version: str = __version__
    run_id: str
    experiment: Experiment
    source: Source
    turns: list[TurnRecord] = Field(default_factory=list)
    provenance: dict[str, Provenance] = Field(default_factory=dict)


def record_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "record.json"


def save_record(run_dir: str | Path, record: Record) -> None:
    write_json(record_path(run_dir), record.model_dump(mode="json"))


def load_record(run_dir: str | Path) -> Record:
    return Record.model_validate(read_json(record_path(run_dir)))
