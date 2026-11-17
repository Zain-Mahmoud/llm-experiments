from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import RunManifest

HOOK_SCRIPT = Path(__file__).resolve().parent / "hook.py"


def _hook_command(run_dir: str | Path, agent: str) -> str:
    """Shell command the agent runs for each hook event.

    Uses ``sys.executable`` (the interpreter running the harness) rather than a
    bare ``python3``, which is often absent on Windows, and quotes every argument
    for the local shell: ``list2cmdline`` for cmd.exe, ``shlex.join`` for POSIX.
    """
    parts = [
        sys.executable,
        str(HOOK_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--source",
        agent,
    ]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)

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


def hook_settings_path(agent: str, workspace: str | Path) -> Path:
    if agent == "codex":
        return Path(workspace) / ".codex" / "hooks.json"
    return Path(workspace) / ".claude" / "settings.local.json"


def hook_config(agent: str, run_dir: str | Path) -> dict:
    """Hook definitions the agent reads, pointing back at our forwarder."""
    events = CODEX_HOOK_EVENTS if agent == "codex" else CLAUDE_HOOK_EVENTS
    tool_events = CODEX_TOOL_EVENTS if agent == "codex" else CLAUDE_TOOL_EVENTS
    command = _hook_command(run_dir, agent)
    hooks = {}
    for event in events:
        group = {"hooks": [{"type": "command", "command": command, "timeout": 30}]}
        if event in tool_events:
            group["matcher"] = "*"
        hooks[event] = [group]
    return {"hooks": hooks}


def codex_config_path(workspace: str | Path) -> Path:
    return Path(workspace) / ".codex" / "config.toml"


def codex_permission_lines(access: str) -> list[str]:
    """Top-level config.toml keys for the access level. The desktop app takes no
    flags, so these replace the CLI's --sandbox / --ask-for-approval."""
    approval = "never" if access == "high" else "on-request"
    return [
        f'sandbox_mode = "{CODEX_SANDBOX.get(access, "workspace-write")}"',
        f'approval_policy = "{approval}"',
    ]


# Bundle ids outlast app names: Codex ships inside the app named "ChatGPT".
DESKTOP_APPS = {"claude": "com.anthropic.claudefordesktop", "codex": "com.openai.codex"}


def find_app(bundle_id: str) -> str | None:
    """Path of an installed macOS app, found by bundle id through Spotlight."""
    try:
        out = subprocess.run(
            ["mdfind", f"kMDItemCFBundleIdentifier == '{bundle_id}'"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return next((line for line in out.splitlines() if line.endswith(".app")), None)


def launch_command(manifest: "RunManifest", workspace: str | Path) -> str:
    # The desktop apps take no flags: open the workspace folder in the app. The
    # access level is written to the project config next to the hooks instead.
    if manifest.agent == "codex":
        return f"open -b {DESKTOP_APPS['codex']}"
    return "open -a Claude"
