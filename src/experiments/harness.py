from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import __version__
from .agent import hook_config, hook_settings_path, launch_command
from .transcript import TokenUsage, parse_transcript, summarize

RUNS_DIRNAME = "runs"
TRANSCRIPT_SNAPSHOT = "transcript.jsonl"  # copy of the agent's transcript kept in the run dir
FILE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "str_replace_editor"}
RunStatus = Literal["initialized", "running", "finished", "aborted"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime | None = None) -> str:
    return (ts or utcnow()).astimezone(timezone.utc).isoformat()


def runs_dir() -> Path:
    return Path.cwd() / RUNS_DIRNAME


def slugify(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text.strip()]
    slug = "".join(keep).strip("-").lower()
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def new_run_id(name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slugify(name)}"


def make_run_dir(name: str) -> Path:
    """Create a fresh ``runs/<id>/``. Ids only have one-second resolution, so a second
    run started in the same second gets a ``-2``, ``-3`` ... suffix instead of failing."""
    base = new_run_id(name)
    for attempt in range(1, 1000):
        run_dir = runs_dir() / (base if attempt == 1 else f"{base}-{attempt}")
        try:
            run_dir.mkdir(parents=True)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a unique run directory for {base}")


def resolve_run_id(identifier: str) -> str:
    root = runs_dir()
    if identifier == "latest":
        candidates = sorted(p.name for p in root.iterdir() if (p / "run.json").exists())
        return candidates[-1]
    return identifier


def resolve_run(identifier: str) -> Path:
    return runs_dir() / resolve_run_id(identifier)



def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, obj: Any) -> None:
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.write(fd, line.encode("utf-8"))
    os.close(fd)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RunManifest(BaseModel):
    """Persisted to ``runs/<id>/run.json``."""

    run_id: str
    name: str
    agent: str
    access: str
    workspace: str
    run_dir: str
    # Experiment design fields given to `init` (scenario, framing, ...); the
    # allowed keys are the fields of record.Experiment.
    experiment: dict[str, str] = Field(default_factory=dict)
    # "hooks": captured live from an agent; "import": built from a transcript by `harness import`.
    source_kind: Literal["hooks", "import"] = "hooks"

    hook_settings_path: str | None = None
    hook_backup_path: str | None = None

    launch_command: str | None = None
    launch_script: str | None = None
    transcript_path: str | None = None

    status: RunStatus = "initialized"
    created_at: str = Field(default_factory=iso)
    ended_at: str | None = None
    harness_version: str = __version__


class Event(BaseModel):
    """One line in ``events.jsonl``."""

    received_at: str = Field(default_factory=iso)
    run_id: str
    source: str = "hook"
    kind: str
    session_id: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    text: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    index: int
    name: str
    status: str = "pending"  # pending | ok | error | unknown
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: float | None = None
    file_path: str | None = None
    preview: str = ""


class Turn(BaseModel):
    index: int
    prompt: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: float | None = None
    tool_calls: int = 0
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    # In memory only (kept out of metrics.json); capture.py builds the record from these.
    response: str = Field(default="", exclude=True)
    calls: list[ToolCall] = Field(default_factory=list, exclude=True)


class DiffStats(BaseModel):
    is_git: bool = False
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    untracked: int = 0
    patch_path: str | None = None
    stat_path: str | None = None


class RubricScore(BaseModel):
    item: str
    value: str | None = None
    note: str | None = None


class Metrics(BaseModel):
    run_id: str
    name: str
    agent: str
    model: str | None = None
    status: str = "initialized"

    started_at: str | None = None
    ended_at: str | None = None
    wall_clock_s: float | None = None
    first_action_latency_ms: float | None = None

    turns: int = 0
    human_prompts: int = 0
    assistant_messages: int = 0

    tokens: TokenUsage = Field(default_factory=TokenUsage)
    tokens_by_model: dict[str, TokenUsage] = Field(default_factory=dict)

    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    tool_failures: int = 0
    permission_requests: int = 0
    human_interventions: int = 0

    files_touched: list[str] = Field(default_factory=list)
    diff: DiffStats | None = None

    turn_records: list[Turn] = Field(default_factory=list)
    rubric: list[RubricScore] = Field(default_factory=list)
    notes: list[dict[str, Any]] = Field(default_factory=list)

    transcript_path: str | None = None
    generated_at: str = Field(default_factory=iso)



def load_events(run_dir: str | Path) -> list[Event]:
    return [Event.model_validate(r) for r in read_jsonl(Path(run_dir) / "events.jsonl")]


def append_note(run_dir: str | Path, kind: str, text: str, data: dict | None = None) -> None:
    """Append a human-authored record to a run's event log."""
    run_dir = Path(run_dir)
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "run_id": run_dir.name,
            "source": "human",
            "kind": kind,
            "text": text,
            "data": data or {},
        },
    )


def _parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _ms(start: str | None, end: str | None) -> float | None:
    a, b = _parse_ts(start), _parse_ts(end)
    return (b - a).total_seconds() * 1000.0 if a and b else None


def _preview(text: Any, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _file_path(name: str, tool_input: dict) -> str | None:
    if name in FILE_TOOLS or name in {"Read", "NotebookRead"}:
        return tool_input.get("file_path") or tool_input.get("notebook_path")
    return None


def _tool_preview(name: str, tool_input: dict) -> str:
    file_path = _file_path(name, tool_input)
    if file_path:
        return _preview(f"{name} {file_path}")
    if name in {"Bash", "shell", "local_shell_call"}:
        return _preview(tool_input.get("command", ""))
    for key in ("pattern", "url", "query", "description", "prompt"):
        if key in tool_input:
            return _preview(tool_input[key])
    return _preview(json.dumps(tool_input, sort_keys=True))


def _build_tool_calls(events: list[Event], entries) -> list[ToolCall]:
    records: list[ToolCall] = []
    by_id: dict[str, ToolCall] = {}

    for ev in events:
        tool_input = ev.payload.get("tool_input") or {}
        if ev.kind == "pre_tool_use":
            call = ToolCall(
                index=len(records) + 1,
                name=ev.tool_name or "unknown",
                started_at=ev.received_at,
                file_path=_file_path(ev.tool_name or "", tool_input),
                preview=_tool_preview(ev.tool_name or "", tool_input),
            )
            records.append(call)
            if ev.tool_use_id:
                by_id[ev.tool_use_id] = call
        elif ev.kind in {"post_tool_use", "tool_failure"}:
            call = by_id.get(ev.tool_use_id or "")
            if call:
                call.status = "ok" if ev.kind == "post_tool_use" else "error"
                call.ended_at = ev.received_at
                call.duration_ms = _ms(call.started_at, call.ended_at)

    # Fall back to transcript tool_use blocks when no hook events were captured.
    if not records:
        for entry in entries:
            for tu in entry.tool_uses:
                records.append(
                    ToolCall(
                        index=len(records) + 1,
                        name=tu.name,
                        status="unknown",
                        started_at=entry.timestamp,
                        file_path=_file_path(tu.name, tu.input),
                        preview=_tool_preview(tu.name, tu.input),
                    )
                )
    return records


def _event_turns(events: list[Event]) -> list[dict]:
    turns: list[dict] = []
    current: dict | None = None
    for ev in events:
        if ev.kind == "user_prompt":
            current = {
                "prompt": str(ev.payload.get("prompt", "")).strip(),
                "started": ev.received_at,
                "ended": None,
            }
            turns.append(current)
        elif ev.kind == "stop" and current is not None:
            current["ended"] = ev.received_at
    return turns


def _transcript_turns(entries) -> list[dict]:
    turns: list[dict] = []
    current: dict | None = None
    for entry in entries:
        if entry.is_prompt:
            current = {
                "prompt": entry.text.strip(),
                "started": entry.timestamp,
                "ended": None,
                "tokens": TokenUsage(),
                "response": [],
            }
            turns.append(current)
        elif entry.kind == "assistant":
            if current is None:
                current = {
                    "prompt": "",
                    "started": entry.timestamp,
                    "ended": None,
                    "tokens": TokenUsage(),
                    "response": [],
                }
                turns.append(current)
            current["tokens"] = current["tokens"] + entry.usage
            current["ended"] = entry.timestamp or current["ended"]
            if entry.text.strip():
                current["response"].append(entry.text.strip())
    return turns


def _calls_in_range(tool_calls: list[ToolCall], start: str | None, end: str | None) -> list[ToolCall]:
    s = _parse_ts(start)
    if s is None:
        return []
    e = _parse_ts(end)
    calls = []
    for call in tool_calls:
        t = _parse_ts(call.started_at)
        if t and t >= s and (e is None or t <= e):
            calls.append(call)
    return calls


def _build_turns(events: list[Event], entries, tool_calls: list[ToolCall]) -> list[Turn]:
    # Event log gives structure and timing; the transcript gives per-turn token
    # counts, matched by turn order.
    base = _event_turns(events) or _transcript_turns(entries)
    transcript_turns = _transcript_turns(entries)
    records: list[Turn] = []
    for i, turn in enumerate(base):
        matched = transcript_turns[i] if i < len(transcript_turns) else None
        tokens = turn.get("tokens")
        if tokens is None:
            tokens = matched["tokens"] if matched else TokenUsage()
        response = turn.get("response")
        if response is None:
            response = matched["response"] if matched else []
        # A turn with no end (interrupted, or followed straight away by another
        # prompt) stops where the next turn starts, so calls are never counted twice.
        end = turn["ended"] or (base[i + 1]["started"] if i + 1 < len(base) else None)
        calls = _calls_in_range(tool_calls, turn["started"], end)
        records.append(
            Turn(
                index=i + 1,
                prompt=turn["prompt"],
                started_at=turn["started"],
                ended_at=turn["ended"],
                duration_ms=_ms(turn["started"], turn["ended"]),
                tool_calls=len(calls),
                tokens=tokens,
                response="\n\n".join(response),
                calls=calls,
            )
        )
    return records


def _rubric_from_events(events: list[Event]) -> list[RubricScore]:
    return [
        RubricScore(
            item=str(ev.data.get("item") or ev.text or ""),
            value=ev.data.get("value"),
            note=ev.data.get("note"),
        )
        for ev in events
        if ev.source == "human" and ev.kind == "score"
    ]


def _first_action(events: list[Event], tool_calls: list[ToolCall]) -> float | None:
    prompts = [t for t in (_parse_ts(e.received_at) for e in events if e.kind == "user_prompt") if t]
    calls = [t for t in (_parse_ts(c.started_at) for c in tool_calls) if t]
    if prompts and calls:
        return max(0.0, (min(calls) - min(prompts)).total_seconds() * 1000.0)
    return None


def compute_metrics(
    manifest: RunManifest,
    events: list[Event],
    entries,
    diff: DiffStats | None = None,
    transcript_path: str | None = None,
) -> Metrics:
    tool_calls = _build_tool_calls(events, entries)
    turn_records = _build_turns(events, entries, tool_calls)

    summary = summarize(entries)
    tokens = summary["tokens"]
    if tokens.total() == 0 and turn_records:
        acc = TokenUsage()
        for turn in turn_records:
            acc = acc + turn.tokens
        tokens = acc

    tokens_by_model = summary["tokens_by_model"]
    models = [m for m in tokens_by_model if m != "unknown"]
    model = models[0] if len(models) == 1 else None

    counts: dict[str, int] = {}
    for call in tool_calls:
        counts[call.name] = counts.get(call.name, 0) + 1

    timestamps = [t for t in (_parse_ts(e.received_at) for e in events) if t]
    started, ended = (timestamps[0], timestamps[-1]) if timestamps else (None, None)

    return Metrics(
        run_id=manifest.run_id,
        name=manifest.name,
        agent=manifest.agent,
        model=model,
        status=manifest.status,
        started_at=iso(started) if started else None,
        ended_at=iso(ended) if ended else None,
        wall_clock_s=(ended - started).total_seconds() if started and ended else None,
        first_action_latency_ms=_first_action(events, tool_calls),
        turns=len(turn_records),
        human_prompts=sum(1 for e in events if e.kind == "user_prompt"),
        assistant_messages=summary["assistant_messages"],
        tokens=tokens,
        tokens_by_model=tokens_by_model,
        tool_calls=tool_calls,
        tool_call_counts=counts,
        tool_failures=sum(1 for c in tool_calls if c.status == "error"),
        permission_requests=sum(1 for e in events if e.kind == "permission_request"),
        human_interventions=sum(
            1
            for e in events
            if e.source == "human" and e.kind in {"intervention", "approve", "deny"}
        ),
        files_touched=sorted({c.file_path for c in tool_calls if c.file_path}),
        diff=diff,
        turn_records=turn_records,
        rubric=_rubric_from_events(events),
        notes=[
            {"at": e.received_at, "kind": e.kind, "text": e.text, "data": e.data}
            for e in events
            if e.source == "human"
        ],
        transcript_path=transcript_path,
    )


def summary_markdown(m: Metrics) -> str:
    lines = [
        f"# Run {m.run_id}",
        "",
        f"**Experiment:** {m.name}  ",
        f"**Agent:** {m.agent}" + (f" (`{m.model}`)" if m.model else "") + "  ",
        f"**Status:** {m.status}  ",
    ]
    if m.wall_clock_s is not None:
        lines.append(f"**Wall clock:** {m.wall_clock_s:.1f}s  ")
    if m.first_action_latency_ms is not None:
        lines.append(f"**Time to first action:** {m.first_action_latency_ms / 1000:.1f}s  ")

    lines += [
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Turns | {m.turns} |",
        f"| Human prompts | {m.human_prompts} |",
        f"| Tool calls | {len(m.tool_calls)} |",
        f"| Tool failures | {m.tool_failures} |",
        f"| Permission prompts | {m.permission_requests} |",
        f"| Human interventions | {m.human_interventions} |",
        f"| Input tokens | {m.tokens.input_tokens:,} |",
        f"| Output tokens | {m.tokens.output_tokens:,} |",
        f"| Cache read tokens | {m.tokens.cache_read_input_tokens:,} |",
        f"| Cache write tokens | {m.tokens.cache_creation_input_tokens:,} |",
        f"| Total tokens | {m.tokens.total():,} |",
        "",
    ]

    if m.tokens_by_model:
        lines += ["## Tokens by model", "", "| Model | Input | Output |", "| --- | ---: | ---: |"]
        for model, t in m.tokens_by_model.items():
            lines.append(f"| {model} | {t.input_tokens:,} | {t.output_tokens:,} |")
        lines.append("")

    if m.tool_call_counts:
        lines += ["## Tool calls", "", "| Tool | Count |", "| --- | ---: |"]
        for name, count in sorted(m.tool_call_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {count} |")
        lines.append("")

    if m.turn_records:
        lines += [
            "## Turns",
            "",
            "| # | Prompt | Duration | Tool calls | Output tokens |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
        for turn in m.turn_records:
            prompt = _preview(turn.prompt, 60).replace("|", "\\|") or "(no prompt text)"
            dur = f"{turn.duration_ms / 1000:.1f}s" if turn.duration_ms is not None else "—"
            lines.append(
                f"| {turn.index} | {prompt} | {dur} | {turn.tool_calls} | "
                f"{turn.tokens.output_tokens:,} |"
            )
        lines.append("")

    if m.files_touched:
        lines += ["## Files touched", ""] + [f"- `{p}`" for p in m.files_touched] + [""]

    if m.diff and m.diff.is_git:
        lines += [
            "## Diff",
            "",
            f"{m.diff.files_changed} files changed, +{m.diff.insertions} / -{m.diff.deletions} "
            f"({m.diff.untracked} untracked)",
            "",
        ]

    if m.rubric:
        lines += ["## Rubric", "", "| Item | Score | Note |", "| --- | --- | --- |"]
        for score in m.rubric:
            note = (score.note or "").replace("|", "\\|")
            lines.append(f"| {score.item} | {score.value if score.value is not None else '—'} | {note} |")
        lines.append("")

    if m.notes:
        lines += ["## Human log", ""]
        for note in m.notes:
            value = f" — {note['text']}" if note.get("text") else ""
            lines.append(f"- `{note['at']}` **{note['kind']}**{value}")
        lines.append("")

    return "\n".join(lines)


def write_metrics(run_dir: str | Path, metrics: Metrics) -> None:
    run_dir = Path(run_dir)
    write_json(run_dir / "metrics.json", metrics.model_dump(mode="json"))
    (run_dir / "summary.md").write_text(summary_markdown(metrics), encoding="utf-8")


# --------------------------------------------------------------------------- #
# run lifecycle
# --------------------------------------------------------------------------- #
def load_manifest(run_dir: str | Path) -> RunManifest:
    return RunManifest.model_validate(read_json(Path(run_dir) / "run.json"))


def save_manifest(manifest: RunManifest) -> None:
    write_json(Path(manifest.run_dir) / "run.json", manifest.model_dump(mode="json"))


def create_run(
    *,
    agent: str = "claude",
    access: str = "medium",
    name: str | None = None,
    experiment: dict[str, str] | None = None,
) -> RunManifest:
    """Prepare a run in the current directory: hooks and a launch command.

    The agent works in the directory the harness was launched from; the harness
    never launches anything itself.
    """
    workspace = Path.cwd()
    run_name = name or workspace.name or agent
    run_dir = make_run_dir(run_name)

    manifest = RunManifest(
        run_id=run_dir.name,
        name=run_name,
        agent=agent,
        access=access,
        workspace=str(workspace),
        run_dir=str(run_dir),
        experiment=experiment or {},
    )
    _install_hooks(manifest)

    manifest.launch_command = launch_command(manifest, workspace)
    script = run_dir / "launch.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(workspace))}\nexec {manifest.launch_command}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    manifest.launch_script = str(script)

    save_manifest(manifest)
    return manifest


def _is_git_repo(path: Path) -> bool:
    if not path.is_dir():
        return False
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def _install_hooks(manifest: RunManifest) -> None:
    settings = hook_settings_path(manifest.agent, manifest.workspace)
    settings.parent.mkdir(parents=True, exist_ok=True)
    existing = read_json(settings) if settings.exists() else {}
    if settings.exists():
        backup = Path(manifest.run_dir) / "hooks.original.json"
        shutil.copy2(settings, backup)
        manifest.hook_backup_path = str(backup)
    hooks = existing.get("hooks") or {}
    hooks.update(hook_config(manifest.agent, manifest.run_dir)["hooks"])
    existing["hooks"] = hooks
    write_json(settings, existing)
    manifest.hook_settings_path = str(settings)


def _restore_hooks(manifest: RunManifest) -> None:
    if not manifest.hook_settings_path:
        return
    settings = Path(manifest.hook_settings_path)
    if manifest.hook_backup_path:
        shutil.copy2(manifest.hook_backup_path, settings)
    elif settings.exists():
        settings.unlink()


def git_diff(workspace: str | Path, run_dir: str | Path) -> DiffStats:
    workspace, run_dir = Path(workspace), Path(run_dir)
    if not _is_git_repo(workspace):
        return DiffStats(is_git=False)

    stats = DiffStats(is_git=True)
    pathspec = ["--", ".", f":(exclude){RUNS_DIRNAME}"]
    numstat = _run(["git", "-C", str(workspace), "diff", "--numstat", *pathspec]).stdout
    for line in numstat.splitlines():
        insertions, deletions, *_ = line.split("\t")
        stats.files_changed += 1
        stats.insertions += int(insertions)
        stats.deletions += int(deletions)

    status = _run(["git", "-C", str(workspace), "status", "--porcelain", *pathspec]).stdout
    stats.untracked = sum(1 for line in status.splitlines() if line.startswith("??"))

    patch = _run(["git", "-C", str(workspace), "diff", "--no-color", *pathspec]).stdout
    if patch:
        (run_dir / "diff.patch").write_text(patch, encoding="utf-8")
        stats.patch_path = str(run_dir / "diff.patch")

    stat_text = _run(["git", "-C", str(workspace), "diff", "--stat", *pathspec]).stdout
    if stat_text:
        (run_dir / "diff.stat.txt").write_text(stat_text, encoding="utf-8")
        stats.stat_path = str(run_dir / "diff.stat.txt")
    return stats


def _transcript_path(manifest: RunManifest, events: list[Event]) -> str | None:
    path = manifest.transcript_path or next(
        (str(ev.payload["transcript_path"]) for ev in events if ev.payload.get("transcript_path")),
        None,
    )
    # The agent's own copy can be rotated or deleted; fall back to the snapshot kept in the run.
    snapshot = Path(manifest.run_dir) / TRANSCRIPT_SNAPSHOT
    if (not path or not Path(path).is_file()) and snapshot.is_file():
        return str(snapshot)
    return path


def build_metrics(run_dir: str | Path, diff: DiffStats | None = None) -> Metrics:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    events = load_events(run_dir)
    transcript = _transcript_path(manifest, events)
    entries = parse_transcript(transcript) if transcript and Path(transcript).is_file() else []
    return compute_metrics(manifest, events, entries, diff=diff, transcript_path=transcript)


def finalize_run(identifier: str, *, capture_diff: bool = True) -> Metrics:
    run_dir = resolve_run(identifier)
    manifest = load_manifest(run_dir)

    # Restore hooks first so harness artifacts do not show up in the diff.
    _restore_hooks(manifest)

    diff = git_diff(manifest.workspace, run_dir) if capture_diff else None
    metrics = build_metrics(run_dir, diff=diff)

    manifest.status = "finished"
    manifest.ended_at = metrics.ended_at or iso()
    manifest.transcript_path = metrics.transcript_path
    save_manifest(manifest)

    metrics.status = "finished"
    metrics.ended_at = manifest.ended_at
    write_metrics(run_dir, metrics)
    return metrics


# --------------------------------------------------------------------------- #
# background watcher
# --------------------------------------------------------------------------- #
def spawn_watcher(run_dir: str | Path) -> int:
    run_dir = Path(run_dir)
    with open(run_dir / "watcher.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "experiments.cli", "watch", str(run_dir), "--quiet"],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (run_dir / "watcher.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def stop_watcher(run_dir: str | Path) -> None:
    pid_file = Path(run_dir) / "watcher.pid"
    if pid_file.exists():
        os.kill(int(pid_file.read_text(encoding="utf-8").strip()), 15)
        pid_file.unlink()
