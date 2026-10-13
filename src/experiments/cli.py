"""Command line interface.

Typical workflow:

    harness init --workspace ~/code/my-repo --access high
    # launch the printed command yourself, then:
    harness watch latest
    harness note latest -t "told the agent to use a different port"
    harness score latest -i "creates a file" -v pass
    harness finalize latest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .agent import resolve_binary
from .harness import (
    append_note,
    build_metrics,
    create_run,
    finalize_run,
    find_repo_root,
    load_manifest,
    resolve_run,
    runs_dir,
    spawn_watcher,
    stop_watcher,
    summary_markdown,
    write_metrics,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Human-in-the-loop harness for LLM agent experiments.",
    )
    parser.add_argument("--version", action="version", version=f"harness {__version__}")
    parser.add_argument("--repo-root", default=None, help="Harness project root (auto-detected).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Prepare a run: workspace, hooks, manifest.")
    p.add_argument("--agent", choices=["claude", "codex"], default="claude")
    p.add_argument("--workspace", default=None, help="Repo/dir the agent works in.")
    p.add_argument(
        "--isolation", choices=["worktree", "copy", "none"], default="worktree"
    )
    p.add_argument("--access", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--count", type=int, default=1, help="Create N runs.")
    p.add_argument("--no-watch", action="store_true", help="Don't start the background watcher.")
    p.set_defaults(handler=_cmd_init)

    p = sub.add_parser("list", help="List runs.")
    p.set_defaults(handler=_cmd_list)

    p = sub.add_parser("show", help="Show a run manifest.")
    p.add_argument("run")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_show)

    p = sub.add_parser("path", help="Print a run's directory and workspace.")
    p.add_argument("run")
    p.set_defaults(handler=_cmd_path)

    p = sub.add_parser("status", help="Recompute and print metrics for a run.")
    p.add_argument("run")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_status)

    p = sub.add_parser("watch", help="Follow a run live and keep metrics updated.")
    p.add_argument("run")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(handler=_cmd_watch)

    p = sub.add_parser("note", help="Append a human note or intervention.")
    p.add_argument("run")
    p.add_argument("-t", "--text", required=True)
    p.add_argument(
        "-k",
        "--kind",
        default="note",
        choices=["note", "intervention", "approve", "deny", "task_result"],
    )
    p.set_defaults(handler=_cmd_note)

    p = sub.add_parser("score", help="Record a rubric score.")
    p.add_argument("run")
    p.add_argument("-i", "--item", required=True)
    p.add_argument("-v", "--value", required=True, help="e.g. pass / fail / 0.5")
    p.add_argument("-n", "--note", default=None)
    p.set_defaults(handler=_cmd_score)

    p = sub.add_parser("finalize", help="Compute final metrics and restore hooks.")
    p.add_argument("run")
    p.add_argument("--no-diff", action="store_true")
    p.set_defaults(handler=_cmd_finalize)

    p = sub.add_parser("doctor", help="Check the environment and agent CLIs.")
    p.set_defaults(handler=_cmd_doctor)

    return parser


def _repo_root(args) -> Path:
    return Path(args.repo_root) if args.repo_root else find_repo_root()


def _cmd_init(args) -> int:
    repo_root = _repo_root(args)
    created = []
    for _ in range(args.count):
        manifest = create_run(
            agent=args.agent,
            workspace=args.workspace,
            isolation=args.isolation,
            access=args.access,
            repo_root=repo_root,
        )
        if not args.no_watch:
            spawn_watcher(manifest.run_dir)
        created.append(manifest)

    for manifest in created:
        _print_run_banner(manifest)
    if len(created) > 1:
        print(f"Created {len(created)} runs.")
    return 0


def _print_run_banner(manifest) -> None:
    print()
    print(f"run        {manifest.run_id}")
    print(f"run dir    {manifest.run_dir}")
    print(f"workspace  {manifest.workspace}")
    print(f"hooks      {manifest.hook_settings_path}")
    print()
    print("Launch the agent yourself with:")
    print()
    print(f"    {manifest.launch_command}")
    print()
    print(f"or:  {manifest.launch_script}")
    print()
    print("Monitor with:")
    print(f"    harness watch {manifest.run_id}")
    print("When finished:")
    print(f"    harness finalize {manifest.run_id}")
    print()


def _cmd_list(args) -> int:
    root = runs_dir(_repo_root(args))
    rows = sorted(p for p in root.iterdir() if (p / "run.json").exists())
    if not rows:
        print("(no runs)")
        return 0
    print(f"{'run id':<34} {'status':<12} {'agent':<8} {'turns':>6} {'tools':>6}")
    for run_dir in rows:
        manifest = load_manifest(run_dir)
        turns = tools = ""
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            turns = data.get("turns", "")
            tools = len(data.get("tool_calls") or [])
        print(
            f"{manifest.run_id:<34} {manifest.status:<12} {manifest.agent:<8} "
            f"{turns!s:>6} {tools!s:>6}"
        )
    return 0


def _cmd_show(args) -> int:
    manifest = load_manifest(resolve_run(args.run, _repo_root(args)))
    if args.json:
        print(json.dumps(manifest.model_dump(mode="json"), indent=2))
        return 0
    print(f"run           {manifest.run_id}")
    print(f"name          {manifest.name}")
    print(f"agent         {manifest.agent} (access={manifest.access})")
    print(f"status        {manifest.status}")
    print(f"workspace     {manifest.workspace}")
    print(f"run dir       {manifest.run_dir}")
    print(f"hooks         {manifest.hook_settings_path}")
    print(f"launch        {manifest.launch_command}")
    return 0


def _cmd_path(args) -> int:
    manifest = load_manifest(resolve_run(args.run, _repo_root(args)))
    print(f"run_dir   {manifest.run_dir}")
    print(f"workspace {manifest.workspace}")
    return 0


def _cmd_status(args) -> int:
    run_dir = resolve_run(args.run, _repo_root(args))
    metrics = build_metrics(run_dir)
    write_metrics(run_dir, metrics)
    if args.json:
        print(json.dumps(metrics.model_dump(mode="json"), indent=2))
        return 0
    print(summary_markdown(metrics))
    return 0


def _cmd_watch(args) -> int:
    _watch(resolve_run(args.run, _repo_root(args)), interval=args.interval, quiet=args.quiet)
    return 0


def _cmd_note(args) -> int:
    run_dir = resolve_run(args.run, _repo_root(args))
    append_note(run_dir, args.kind, args.text)
    print(f"recorded {args.kind} on {run_dir.name}")
    return 0


def _cmd_score(args) -> int:
    run_dir = resolve_run(args.run, _repo_root(args))
    append_note(
        run_dir, "score", args.item, {"item": args.item, "value": args.value, "note": args.note}
    )
    print(f"scored {args.item}={args.value}")
    return 0


def _cmd_finalize(args) -> int:
    run_dir = resolve_run(args.run, _repo_root(args))
    stop_watcher(run_dir)
    metrics = finalize_run(
        run_dir.name, repo_root=_repo_root(args), capture_diff=not args.no_diff
    )
    print(f"finalized {metrics.run_id}")
    print(f"  metrics  {run_dir / 'metrics.json'}")
    print(f"  summary  {run_dir / 'summary.md'}")
    if metrics.diff and metrics.diff.patch_path:
        print(f"  diff     {metrics.diff.patch_path}")
    return 0


def _cmd_doctor(args) -> int:
    root = _repo_root(args)
    print(f"harness  {__version__}")
    print(f"python   {sys.executable}")
    print(f"repo     {root}")
    print(f"runs     {runs_dir(root)}")
    print()
    for name in ("claude", "codex"):
        print(f"{name:<8} {resolve_binary(name) or 'NOT FOUND'}")
    return 0


# --------------------------------------------------------------------------- #
# live view
# --------------------------------------------------------------------------- #
def _watch(run_dir: Path, *, interval: float, quiet: bool) -> None:
    if not quiet:
        print(f"watching {run_dir}")
        print("─" * 72)
    seen = 0
    while True:
        events_file = run_dir / "events.jsonl"
        events = []
        if events_file.exists():
            events = [
                json.loads(line)
                for line in events_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if not quiet:
            for raw in events[seen:]:
                line = _render_event(raw)
                if line:
                    print(line, flush=True)
        seen = len(events)

        metrics = build_metrics(run_dir)
        write_metrics(run_dir, metrics)
        if not quiet:
            print(_render_status(metrics), end="\r", flush=True)
        time.sleep(interval)


def _render_event(raw: dict) -> str | None:
    kind, payload = raw.get("kind"), raw.get("payload") or {}
    if raw.get("source") == "human":
        return f"👤 [{kind}] {raw.get('text') or ''}"
    if kind == "session_start":
        return f"🟢 session start (model={payload.get('model', '?')})"
    if kind == "session_end":
        return f"🔴 session end (reason={payload.get('reason', '?')})"
    if kind == "user_prompt":
        return f"▶️  prompt: {_clip(payload.get('prompt', ''))}"
    if kind == "pre_tool_use":
        return f"→ {raw.get('tool_name')}: {_clip(_tool_preview(payload))}"
    if kind == "post_tool_use":
        return f"✓ {raw.get('tool_name')}"
    if kind == "tool_failure":
        return f"✗ {raw.get('tool_name')}: {_clip(str(payload.get('error', '')))}"
    if kind == "permission_request":
        return f"🔐 permission requested: {raw.get('tool_name')}"
    if kind == "permission_denied":
        return f"🚫 permission denied: {raw.get('tool_name')}"
    if kind == "notification":
        return f"🔔 {payload.get('message', '')}"
    if kind == "stop":
        return "⏹️  turn complete"
    return None


def _tool_preview(payload: dict) -> str:
    tool_input = payload.get("tool_input") or {}
    for key in ("command", "file_path", "pattern", "url", "query", "description", "prompt"):
        if key in tool_input:
            return str(tool_input[key])
    return json.dumps(tool_input)


def _clip(text: str, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_status(m) -> str:
    return (
        f"[{m.status}] turns={m.turns} tools={len(m.tool_calls)} "
        f"fail={m.tool_failures} in={m.tokens.input_tokens} out={m.tokens.output_tokens}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
