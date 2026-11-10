"""Parsing for agent session transcripts: Claude Code's and Codex's rollout files.

Claude Code writes one JSONL entry per conversation turn. Assistant entries
carry the API ``usage`` block, which is the source of token counts. Codex
writes typed events tagged with a turn id; see ``parse_codex_rollout``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_dict(cls, usage: dict[str, Any] | None) -> "TokenUsage":
        usage = usage or {}
        return cls(**{k: int(usage.get(k) or 0) for k in TOKEN_KEYS})

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(**{k: getattr(self, k) + getattr(other, k) for k in TOKEN_KEYS})

    def as_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in TOKEN_KEYS}

    def total(self) -> int:
        return sum(getattr(self, k) for k in TOKEN_KEYS)


@dataclass
class ToolUse:
    name: str
    input: dict[str, Any]


@dataclass
class Entry:
    kind: str
    timestamp: str | None = None
    model: str | None = None
    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_uses: list[ToolUse] = field(default_factory=list)
    is_prompt: bool = False
    turn_id: str | None = None  # Codex only; Claude Code transcripts have no turn ids


def _blocks(content: Any) -> list[dict[str, Any]]:
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        str(b.get("text", ""))
        for b in _blocks(content)
        if b.get("type") == "text"
    )


def parse_entry(obj: dict[str, Any]) -> Entry:
    message = obj.get("message") or {}
    content = message.get("content")
    blocks = _blocks(content)
    kind = obj.get("type") or "other"
    has_tool_result = any(b.get("type") == "tool_result" for b in blocks)
    text = _stringify(content)
    # A "user" entry is not always a human prompt: tool results, injected/meta
    # messages, subagent traffic and interrupt markers are all written as user entries.
    human = not (
        has_tool_result
        or obj.get("isMeta")
        or obj.get("isSidechain")
        or text.startswith("[Request interrupted")
    )
    entry = Entry(
        kind=kind,
        timestamp=obj.get("timestamp"),
        model=message.get("model"),
        text=text,
        usage=TokenUsage.from_dict(message.get("usage") or obj.get("usage")),
        is_prompt=kind == "user" and human,
    )
    for block in blocks:
        if block.get("type") == "tool_use":
            entry.tool_uses.append(
                ToolUse(name=str(block.get("name", "")), input=block.get("input") or {})
            )
    return entry


def parse_transcript(path: str | Path) -> list[Entry]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    objs = [json.loads(line) for line in lines if line.strip()]
    if objs and objs[0].get("type") == "session_meta":
        return parse_codex_rollout(objs)
    return [parse_entry(obj) for obj in objs]


def _codex_usage(usage: dict[str, Any] | None) -> TokenUsage:
    # OpenAI counts cached tokens inside input_tokens; Claude counts them apart.
    usage = usage or {}
    cached = int(usage.get("cached_input_tokens") or 0)
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or 0) - cached,
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=int(usage.get("cache_write_input_tokens") or 0),
    )


def _codex_text(content: Any) -> str:
    return "\n".join(str(b.get("text", "")) for b in _blocks(content) if b.get("text"))


def _codex_tool_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        return parsed if isinstance(parsed, dict) else {"input": raw}
    return raw if isinstance(raw, dict) else {}


def parse_codex_rollout(objs: list[dict[str, Any]]) -> list[Entry]:
    """Codex rollout -> entries. The human prompt and the replies come from the
    completed-item events (the model-facing ``message`` items also hold injected
    context); token counts from ``token_usage_record``, as ``usage`` entries so
    they are not counted as messages; the model from ``turn_context``."""
    entries: list[Entry] = []
    turn_id: str | None = None
    model: str | None = None
    for obj in objs:
        kind, payload, ts = obj.get("type"), obj.get("payload") or {}, obj.get("timestamp")
        ptype = payload.get("type")
        turn = payload.get("turn_id") or turn_id
        if kind == "turn_context":
            turn_id, model = payload.get("turn_id") or turn_id, payload.get("model") or model
        elif kind == "event_msg" and ptype == "task_started":
            turn_id = payload.get("turn_id") or turn_id
        elif kind == "event_msg" and ptype == "item_completed":
            item = payload.get("item") or {}
            if item.get("type") == "UserMessage":
                entries.append(Entry(kind="user", timestamp=ts, text=_codex_text(item.get("content")).strip(),
                                     is_prompt=True, turn_id=turn))
            elif item.get("type") == "AgentMessage":
                entries.append(Entry(kind="assistant", timestamp=ts, model=model,
                                     text=_codex_text(item.get("content")), turn_id=turn))
        elif kind == "token_usage_record":
            entries.append(Entry(kind="usage", timestamp=ts, model=model,
                                 usage=_codex_usage(payload.get("usage")), turn_id=turn))
        elif kind == "response_item" and ptype in {"function_call", "custom_tool_call", "local_shell_call"}:
            raw = payload.get("arguments") if ptype == "function_call" else payload.get("input") or payload.get("action")
            entries.append(Entry(kind="tool", timestamp=ts, model=model, turn_id=turn,
                                 tool_uses=[ToolUse(name=str(payload.get("name") or ptype), input=_codex_tool_input(raw))]))
    return entries


def summarize(entries: list[Entry]) -> dict[str, Any]:
    tokens = TokenUsage()
    per_model: dict[str, TokenUsage] = {}
    assistant_messages = 0
    for entry in entries:
        if entry.kind in {"assistant", "usage"}:
            assistant_messages += entry.kind == "assistant"
            tokens = tokens + entry.usage
            model = entry.model or "unknown"
            per_model[model] = per_model.get(model, TokenUsage()) + entry.usage
    return {
        "tokens": tokens,
        "tokens_by_model": per_model,
        "assistant_messages": assistant_messages,
    }
