#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

KINDS = {
    "SessionStart": "session_start",
    "SessionEnd": "session_end",
    "UserPromptSubmit": "user_prompt",
    "PreToolUse": "pre_tool_use",
    "PostToolUse": "post_tool_use",
    "PostToolUseFailure": "tool_failure",
    "PermissionRequest": "permission_request",
    "PermissionDenied": "permission_denied",
    "Notification": "notification",
    "Stop": "stop",
    "StopFailure": "stop_failure",
    "SubagentStart": "subagent_start",
    "SubagentStop": "subagent_stop",
    "PreCompact": "pre_compact",
    "PostCompact": "post_compact",
    "Interrupt": "interrupt",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source", default="claude")
    args, _ = parser.parse_known_args(argv)

    raw = json.load(sys.stdin)
    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "run_id": os.path.basename(os.path.normpath(args.run_dir)),
        "source": args.source,
        "kind": KINDS.get(raw.get("hook_event_name"), "hook"),
        "session_id": raw.get("session_id"),
        "tool_name": raw.get("tool_name"),
        "tool_use_id": raw.get("tool_use_id"),
        "payload": raw,
    }
    line = json.dumps(record, separators=(",", ":")) + "\n"
    fd = os.open(
        os.path.join(args.run_dir, "events.jsonl"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    os.write(fd, line.encode("utf-8"))
    os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
