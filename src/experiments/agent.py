from __future__ import annotations

import importlib.util
import shlex
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import RunManifest

HOOK_SCRIPT = Path(__file__).resolve().parent / "hook.py"

CLAUDE_PERMISSION = {
    "low": "default",
    "medium": "acceptEdits",
    "high": "bypassPermissions",
}
CODEX_SANDBOX = {
    "low": "read-only",
    "medium": "workspace-write",
    "high": "danger-full-access",
}

CLAUDE_HOOK_EVENTS = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "PermissionDenied",
    "Notification",
    "Stop",
    "StopFailure",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "PostCompact",
    "SessionEnd",
]
CLAUDE_TOOL_EVENTS = {
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "PermissionDenied",
}

CODEX_HOOK_EVENTS = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "Stop",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "PostCompact",
    "Interrupt",
    "SessionEnd",
]
CODEX_TOOL_EVENTS = {"PreToolUse", "PostToolUse", "PermissionRequest"}


def resolve_binary(name: str) -> str | None:
    """Locate an agent CLI on PATH, or in its bundled package in this venv."""
    found = shutil.which(name)
    if found:
        return found
    package = {"claude": "claude_agent_sdk", "codex": "codex_cli_bin"}.get(name)
    spec = importlib.util.find_spec(package) if package else None
    if not spec or not spec.submodule_search_locations:
        return None
    pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    candidates = (
        [pkg_dir / "_bundled" / name]
        if name == "claude"
        else [pkg_dir / "bin" / name, pkg_dir / name]
    )
    return next((str(c) for c in candidates if c.is_file()), None)


def hook_settings_path(agent: str, workspace: str | Path) -> Path:
    if agent == "codex":
        return Path(workspace) / ".codex" / "hooks.json"
    return Path(workspace) / ".claude" / "settings.local.json"


def hook_config(agent: str, run_dir: str | Path) -> dict:
    """Hook definitions the agent reads, pointing back at our forwarder."""
    events = CODEX_HOOK_EVENTS if agent == "codex" else CLAUDE_HOOK_EVENTS
    tool_events = CODEX_TOOL_EVENTS if agent == "codex" else CLAUDE_TOOL_EVENTS
    command = (
        f"python3 {shlex.quote(str(HOOK_SCRIPT))} "
        f"--run-dir {shlex.quote(str(run_dir))} --source {agent}"
    )
    hooks = {}
    for event in events:
        group = {"hooks": [{"type": "command", "command": command, "timeout": 30}]}
        if event in tool_events:
            group["matcher"] = "*"
        hooks[event] = [group]
    return {"hooks": hooks}


def launch_command(manifest: "RunManifest", workspace: str | Path) -> str:
    if manifest.agent == "codex":
        return _codex_launch(manifest, workspace)
    return _claude_launch(manifest, workspace)


def _claude_launch(manifest: "RunManifest", workspace: str | Path) -> str:
    parts = [resolve_binary("claude") or "claude"]
    parts += ["--permission-mode", CLAUDE_PERMISSION.get(manifest.access, "acceptEdits")]
    parts += ["--name", manifest.run_id]
    return " ".join(shlex.quote(p) for p in parts)


def _codex_launch(manifest: "RunManifest", workspace: str | Path) -> str:
    parts = [resolve_binary("codex") or "codex"]
    if manifest.access == "high":
        parts.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        parts += [
            "--sandbox",
            CODEX_SANDBOX.get(manifest.access, "workspace-write"),
            "--ask-for-approval",
            "on-request",
        ]
    parts += ["-C", str(workspace), "--dangerously-bypass-hook-trust"]
    return " ".join(shlex.quote(p) for p in parts)
